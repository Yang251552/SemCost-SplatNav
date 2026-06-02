"""Paired, fairness-enforcing experiment runner.

Trains and evaluates BOTH observation modes with the same shared configuration
(semcost_nav/utils/experiment.py), then asserts the two runs are fair before
declaring success. This is the canonical, reproducible entry point:

    python scripts/run_experiment.py

It deliberately exposes only knobs that are applied identically to both modes
(timesteps, seed), so a single invocation cannot produce an unfair pairing.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from semcost_nav.utils import experiment as ex  # noqa: E402

MODES = ("depth", "depth_semantic")


def run(cmd: list[str]) -> None:
    print(f"[run] $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Paired depth vs depth_semantic run.")
    p.add_argument("--timesteps", type=int, default=ex.DEFAULT_TIMESTEPS)
    p.add_argument("--seed", type=int, default=ex.TRAIN_SEED)
    p.add_argument("--n-envs", type=int, default=ex.N_ENVS)
    p.add_argument("--episodes", type=int, default=ex.N_EVAL_EPISODES)
    p.add_argument("--skip-train", action="store_true")
    args = p.parse_args()

    py = sys.executable
    for mode in MODES:
        if not args.skip_train:
            run([
                py, str(ROOT / "scripts" / "train.py"),
                "--obs", mode,
                "--timesteps", str(args.timesteps),
                "--seed", str(args.seed),
                "--n-envs", str(args.n_envs),
            ])
        run([
            py, str(ROOT / "scripts" / "evaluate.py"),
            "--obs", mode,
            "--episodes", str(args.episodes),
        ])

    # Post-hoc fairness assertion across the two runs.
    results = ROOT / "results"
    meta_d = json.loads((results / "train_depth.json").read_text())
    meta_ds = json.loads((results / "train_depth_semantic.json").read_text())
    ex.assert_fair(meta_d, meta_ds)
    print("[run] fairness check passed: both modes share identical "
          "fairness-critical configuration.")

    for mode in MODES:
        m = json.loads((results / f"eval_{mode}.json").read_text())["metrics"]
        print(
            f"[run] {mode:14s} success={m['success_rate']:.3f} "
            f"bad_region_time={m['bad_region_time']:.3f} "
            f"collision_rate={m['collision_rate']:.3f} "
            f"path_len={m['path_length']:.2f} "
            f"return={m['average_return']:.3f}"
        )


if __name__ == "__main__":
    main()
