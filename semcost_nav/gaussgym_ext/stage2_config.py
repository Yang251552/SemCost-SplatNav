"""Shared config assembly for Stage-2 GaussGym dry-runs."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


ARMS = ("depth", "rgb", "rgb_sem")
ALLOWED_OBS_DIFF_PATHS = (
    ("observations", "image_encoder", "observations"),
    ("observations", "critic", "observations"),
)


def load_flagship_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"expected mapping in {path}")
    return cfg


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def set_dotted(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    cur = cfg
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def parse_override_value(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def normalized_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def config_hash(cfg: dict[str, Any]) -> str:
    return hashlib.sha256(normalized_json(cfg).encode("utf-8")).hexdigest()


def strip_paths(data: Any, paths: tuple[tuple[str, ...], ...]) -> Any:
    out = copy.deepcopy(data)
    for path in paths:
        cur = out
        for part in path[:-1]:
            if not isinstance(cur, dict) or part not in cur:
                cur = None
                break
            cur = cur[part]
        if isinstance(cur, dict):
            cur.pop(path[-1], None)
    return out


def diff_paths(a: Any, b: Any, prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    if isinstance(a, dict) and isinstance(b, dict):
        keys = sorted(set(a) | set(b))
        diffs: list[tuple[str, ...]] = []
        for key in keys:
            if key not in a or key not in b:
                diffs.append(prefix + (str(key),))
            else:
                diffs.extend(diff_paths(a[key], b[key], prefix + (str(key),)))
        return diffs
    if isinstance(a, list) and isinstance(b, list):
        if a == b:
            return []
        return [prefix]
    return [] if a == b else [prefix]


def build_arm_config(
    raw_cfg: dict[str, Any],
    arm: str,
    dotted_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    cfg = deep_merge(raw_cfg.get("base_config", {}), raw_cfg.get("override", {}))
    cfg = deep_merge(cfg, raw_cfg["arms"][arm])
    for key, value in (dotted_overrides or {}).items():
        set_dotted(cfg, key, value)
    return cfg


def assert_fairness(configs: dict[str, dict[str, Any]]) -> None:
    stripped = {
        arm: strip_paths(cfg, ALLOWED_OBS_DIFF_PATHS)
        for arm, cfg in configs.items()
    }
    ref_arm = ARMS[0]
    for arm in ARMS[1:]:
        diffs = diff_paths(stripped[ref_arm], stripped[arm])
        if diffs:
            rendered = ["/".join(path) for path in diffs]
            raise AssertionError(
                "fairness violation outside observation allow-list: "
                f"{ref_arm} vs {arm}: {rendered}"
            )


def normalized_fairness_hash(cfg: dict[str, Any]) -> str:
    return config_hash(strip_paths(cfg, ALLOWED_OBS_DIFF_PATHS))


def mock_forward_one_step(cfg: dict[str, Any], arm: str) -> dict[str, Any]:
    import numpy as np

    obs_tokens = cfg["observations"]["image_encoder"]["observations"]
    batch = int(cfg.get("env", {}).get("num_envs", 1))
    outputs: dict[str, Any] = {"arm": arm, "batch": batch, "tokens": list(obs_tokens)}
    if "SEMANTIC_COST" in obs_tokens:
        from semcost_nav.gaussgym_ext.observation_tokens import semantic_cost_observation

        outputs["semantic_cost_shape"] = list(
            semantic_cost_observation(hazard_mask=np.zeros((8, 8), dtype=np.float32)).shape
        )
    outputs["latent_shape"] = [batch, 64]
    return outputs

