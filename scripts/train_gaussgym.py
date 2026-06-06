"""Stage-2 GaussGym training entrypoint with CPU dry-run support."""

from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run GaussGym Stage-2 arm training.")
    # CLAUDE.md Required Commands use --obs; --arm kept as a compatible alias.
    p.add_argument("--obs", "--arm", dest="arm", choices=ARMS, required=True)
    p.add_argument("--config", default=str(ROOT / "configs" / "flagship.yaml"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--env.num_envs", dest="env_num_envs", type=int)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    raw_cfg = load_flagship_config(args.config)
    overrides = {}
    if args.env_num_envs is not None:
        overrides["env.num_envs"] = args.env_num_envs

    arm_cfgs = {arm: build_arm_config(raw_cfg, arm, overrides) for arm in ARMS}
    assert_fairness(arm_cfgs)
    selected = arm_cfgs[args.arm]
    selected_hash = config_hash(selected)

    print(f"[gaussgym-train] arm={args.arm} config_hash={selected_hash}")
    print("[gaussgym-train] fairness OK")

    if args.dry_run:
        out = mock_forward_one_step(selected, args.arm)
        print(
            "[gaussgym-train] dry-run forward OK "
            f"batch={out['batch']} latent_shape={out['latent_shape']}"
        )
        return 0

    try:
        import isaacgym  # noqa: F401
        from gauss_gym.scripts.train import main as gaussgym_main
    except Exception as exc:
        print(f"# TODO: run on GPU instance ({exc})")
        return 2

    sys.argv = [
        "train.py",
        "--task=a1_vision",
        f"--env.num_envs={selected['env']['num_envs']}",
        "--terrain.scenes.iphone_data.split_frac=0.0",
        "--terrain.scenes.grand_tour.split_frac=0.0",
        "--terrain.scenes.arkit.split_frac=1.0",
        # TODO(P2): per-arm obs are NOT injected yet (depth=RAY_CAST /
        # rgb=CAMERA_IMAGE / rgb_sem=+SEMANTIC_COST, plus depth-arm critic
        # RAY_CAST removal). Verify GaussGym Flags list-override on the GPU
        # instance, e.g.:
        #   f"--observations.image_encoder.observations="
        #   + ",".join(selected["observations"]["image_encoder"]["observations"]),
    ]
    # Refuse to run real training until per-arm obs injection is wired & verified;
    # running as-is would train all three arms with identical default a1_vision
    # observations, silently breaking the ablation (fake parity). See
    # docs/AWS_FLAGSHIP_PLAN.md and configs/flagship.yaml `arms:`.
    raise NotImplementedError(
        f"Stage-2 GPU training (arm={args.arm!r}) is not wired: per-arm "
        "observation injection into GaussGym is unverified on real source. "
        "See the TODO above and docs/AWS_FLAGSHIP_PLAN.md before running on GPU."
    )
    gaussgym_main()  # unreachable until obs injection is wired (kept for P2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

