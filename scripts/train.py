"""Train a PPO policy on SemCostNav for one observation mode.

Usage:
    python scripts/train.py --obs depth
    python scripts/train.py --obs depth_semantic

Both modes use identical reward, dynamics, map distribution, horizon, PPO
hyperparameters and training seed (see semcost_nav/utils/experiment.py). Only
the observation channels differ, keeping the comparison fair.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

# Allow running as a script without installing the package.
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from semcost_nav.utils import experiment as ex  # noqa: E402
from semcost_nav.utils.seeding import set_global_seeds  # noqa: E402
from semcost_nav.envs.semcost_nav_env import VALID_OBS_MODES  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO on SemCostNav.")
    p.add_argument("--obs", choices=VALID_OBS_MODES, required=True)
    p.add_argument("--timesteps", type=int, default=ex.DEFAULT_TIMESTEPS)
    p.add_argument("--seed", type=int, default=ex.TRAIN_SEED)
    p.add_argument("--n-envs", type=int, default=ex.N_ENVS)
    p.add_argument("--device", default="cpu")
    p.add_argument("--checkpoints-dir", default=str(ROOT / "checkpoints"))
    p.add_argument("--results-dir", default=str(ROOT / "results"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seeds(args.seed)

    ckpt_dir = Path(args.checkpoints_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    env = ex.make_training_env(args.obs, n_envs=args.n_envs, seed=args.seed)
    model = ex.build_ppo(env, seed=args.seed, device=args.device)

    print(
        f"[train] obs={args.obs} timesteps={args.timesteps} "
        f"n_envs={args.n_envs} seed={args.seed} device={args.device}"
    )
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    elapsed = time.time() - t0

    model_path = ckpt_dir / f"ppo_{args.obs}.zip"
    model.save(model_path)
    env.close()

    meta = {
        "obs_mode": args.obs,
        "timesteps": args.timesteps,
        "seed": args.seed,
        "n_envs": args.n_envs,
        "device": args.device,
        "train_seconds": round(elapsed, 2),
        "model_path": str(model_path),
        "config": ex.config_snapshot(),
        "dependency_versions": ex.dependency_versions(),
    }
    meta_path = results_dir / f"train_{args.obs}.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[train] done in {elapsed:.1f}s -> {model_path}")
    print(f"[train] metadata -> {meta_path}")


if __name__ == "__main__":
    main()
