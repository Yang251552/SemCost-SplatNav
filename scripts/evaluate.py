"""Evaluate a trained PPO policy on a fixed, shared set of SemCostNav maps.

Usage:
    python scripts/evaluate.py --obs depth
    python scripts/evaluate.py --obs depth_semantic

Both observation modes are evaluated on exactly the same fixed seed list
(see semcost_nav/utils/experiment.eval_seeds), so the maps, start columns and
hazard gaps are identical between the two policies. Any metric difference is
attributable to the policy, not to different scene draws.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO  # noqa: E402

from semcost_nav.utils import experiment as ex  # noqa: E402
from semcost_nav.utils.metrics import summarize_metrics  # noqa: E402
from semcost_nav.envs.semcost_nav_env import VALID_OBS_MODES  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate PPO on SemCostNav.")
    p.add_argument("--obs", choices=VALID_OBS_MODES, required=True)
    p.add_argument("--episodes", type=int, default=ex.N_EVAL_EPISODES)
    p.add_argument("--deterministic", action="store_true", default=True)
    p.add_argument("--stochastic", dest="deterministic", action="store_false")
    p.add_argument("--checkpoints-dir", default=str(ROOT / "checkpoints"))
    p.add_argument("--results-dir", default=str(ROOT / "results"))
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def run_episode(model: PPO, env, seed: int, deterministic: bool) -> dict:
    obs, info = env.reset(seed=seed)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
    return {
        "seed": seed,
        "success": bool(info["success"]),
        "collisions": int(info["collisions"]),
        "bad_region_time": int(info["bad_region_time"]),
        "path_length": int(info["path_length"]),
        "ep_return": float(info["ep_return"]),
    }


def main() -> None:
    args = parse_args()
    model_path = Path(args.checkpoints_dir) / f"ppo_{args.obs}.zip"
    if not model_path.exists():
        raise FileNotFoundError(
            f"missing model {model_path}; run train.py --obs {args.obs} first"
        )
    model = PPO.load(model_path, device=args.device)
    env = ex.make_single_env(args.obs)

    seeds = ex.eval_seeds(args.episodes)
    episodes = [run_episode(model, env, s, args.deterministic) for s in seeds]
    env.close()

    summary = summarize_metrics(episodes)
    out = {
        "obs_mode": args.obs,
        "n_episodes": len(episodes),
        "deterministic": args.deterministic,
        "eval_seed_base": ex.EVAL_SEED_BASE,
        "metrics": summary,
        "episodes": episodes,
    }
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"eval_{args.obs}.json"
    out_path.write_text(json.dumps(out, indent=2))

    print(f"[eval] obs={args.obs} episodes={len(episodes)}")
    for k in (
        "success_rate",
        "collision_rate",
        "bad_region_time",
        "path_length",
        "average_return",
    ):
        print(f"  {k:16s}: {summary[k]:.4f}")
    print(f"[eval] -> {out_path}")


if __name__ == "__main__":
    main()
