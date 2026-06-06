from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from semcost_nav.gaussgym_ext.stage2_config import (  # noqa: E402
    ARMS,
    build_arm_config,
    load_flagship_config,
    normalized_fairness_hash,
)


def arm_configs() -> dict[str, dict]:
    raw = load_flagship_config(ROOT / "configs" / "flagship.yaml")
    return {arm: build_arm_config(raw, arm) for arm in ARMS}


def test_config_hash_three_arms_only_obs_diff() -> None:
    hashes = {arm: normalized_fairness_hash(cfg) for arm, cfg in arm_configs().items()}
    assert len(set(hashes.values())) == 1


def test_arm_obs_lists_correct() -> None:
    cfgs = arm_configs()
    assert cfgs["depth"]["observations"]["image_encoder"]["observations"] == [
        "RAY_CAST",
        "PROJECTED_GRAVITY",
    ]
    assert cfgs["rgb"]["observations"]["image_encoder"]["observations"] == [
        "CAMERA_IMAGE",
        "PROJECTED_GRAVITY",
    ]
    assert cfgs["rgb_sem"]["observations"]["image_encoder"]["observations"] == [
        "CAMERA_IMAGE",
        "SEMANTIC_COST",
        "PROJECTED_GRAVITY",
    ]


def test_critic_no_raycast_in_depth_arm() -> None:
    cfgs = arm_configs()
    assert "RAY_CAST" not in cfgs["depth"]["observations"]["critic"]["observations"]


def test_fairness_catches_non_obs_tampering() -> None:
    """Negative test: a diff OUTSIDE the obs allow-list must break fairness."""
    import pytest

    from semcost_nav.gaussgym_ext.stage2_config import assert_fairness

    cfgs = arm_configs()
    # Tamper a non-observation field on one arm (reward scale) -> must raise,
    # proving the fairness assertion is not vacuously passing.
    cfgs["rgb"]["rewards"]["semantic_cost_penalty"]["scale"] = -0.5
    with pytest.raises(AssertionError):
        assert_fairness(cfgs)

