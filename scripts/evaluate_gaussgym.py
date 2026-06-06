"""Stage-2 GaussGym evaluation entrypoint and result template writer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from semcost_nav.gaussgym_ext.stage2_config import (  # noqa: E402
    ARMS,
    assert_fairness,
    build_arm_config,
    config_hash,
    load_flagship_config,
    mock_forward_one_step,
)

# Stage-2 eval output filenames. The depth arm uses a `gaussgym`-prefixed name so
# it does NOT overwrite the Stage-1 protected artifact results/eval_depth.json.
# rgb / rgb_sem keep the names required by CLAUDE.md Stage-2 artifacts.
EVAL_FILENAMES = {
    "depth": "eval_gaussgym_depth.json",
    "rgb": "eval_rgb.json",
    "rgb_sem": "eval_rgb_sem.json",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate all GaussGym Stage-2 arms.")
    p.add_argument("--all-modes", action="store_true", required=True)
    # Budget: spot quota denied -> on-demand $1.4/hr. 3->2 seeds saves ~1/3 of
    # training GPU-hours; disclosed honestly in REPORT.md (paired-eval framing).
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--config", default=str(ROOT / "configs" / "flagship.yaml"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--results-dir", default=str(ROOT / "results"))
    return p.parse_args()


def seed_list(n: int) -> list[int]:
    return list(range(n))


def write_template(path: Path, arm: str, cfg_hash: str, seeds: list[int]) -> None:
    payload = {
        "success_rate": None,
        "collision_rate": None,
        "bad_region_time": None,
        "path_length": None,
        "average_return": None,
        "config_hash": cfg_hash,
        "arm": arm,
        "seed_list": seeds,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    raw_cfg = load_flagship_config(args.config)
    arm_cfgs = {arm: build_arm_config(raw_cfg, arm) for arm in ARMS}
    assert_fairness(arm_cfgs)
    seeds = seed_list(args.seeds)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        for arm in ARMS:
            mock_forward_one_step(arm_cfgs[arm], arm)
            out_path = results_dir / EVAL_FILENAMES[arm]
            write_template(out_path, arm, config_hash(arm_cfgs[arm]), seeds)
            print(f"[gaussgym-eval] wrote {out_path}")
        return 0

    try:
        import isaacgym  # noqa: F401
        import gauss_gym  # noqa: F401
    except Exception as exc:
        print(f"# TODO: run on GPU instance ({exc})")
        return 2

    print("# TODO: connect GaussGym rollout evaluator on GPU instance")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

