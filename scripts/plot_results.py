"""Plot the depth vs depth_semantic metric comparison.

Reads results/eval_depth.json and results/eval_depth_semantic.json and writes
figures/metrics_comparison.png (one panel per primary metric).

Usage:
    python scripts/plot_results.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

METRICS = [
    ("bad_region_time", "Bad-region time\n(steps in hazard)", "lower is better"),
    ("collision_rate", "Collision rate", "lower is better"),
    ("success_rate", "Success rate", "higher is better"),
    ("path_length", "Path length", "lower is better"),
    ("average_return", "Average return", "higher is better"),
]

COLORS = {"depth": "#9aa7b5", "depth_semantic": "#2c7fb8"}
LABELS = {"depth": "depth-only", "depth_semantic": "depth + semantic_cost"}


def load(results_dir: Path, obs: str) -> dict:
    path = results_dir / f"eval_{obs}.json"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run evaluate.py --obs {obs} first")
    return json.loads(path.read_text())["metrics"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default=str(ROOT / "results"))
    p.add_argument("--figures-dir", default=str(ROOT / "figures"))
    args = p.parse_args()

    results_dir = Path(args.results_dir)

    # Fairness guard: refuse to plot if the two runs used different
    # fairness-critical settings (steps, seed, reward, map, hyperparameters...).
    from semcost_nav.utils import experiment as ex  # noqa: E402

    try:
        meta_d = json.loads((results_dir / "train_depth.json").read_text())
        meta_ds = json.loads((results_dir / "train_depth_semantic.json").read_text())
        ex.assert_fair(meta_d, meta_ds)
    except FileNotFoundError:
        print("[plot] warning: train metadata missing; skipping fairness check")

    d = load(results_dir, "depth")
    ds = load(results_dir, "depth_semantic")

    fig, axes = plt.subplots(1, len(METRICS), figsize=(3.0 * len(METRICS), 3.6))
    for ax, (key, title, hint) in zip(axes, METRICS):
        vals = [d[key], ds[key]]
        bars = ax.bar(
            ["depth-only", "depth+sem"],
            vals,
            color=[COLORS["depth"], COLORS["depth_semantic"]],
            width=0.6,
        )
        ax.set_title(f"{title}\n({hint})", fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
        top = max(vals) if max(vals) > 0 else 1.0
        ax.set_ylim(min(0, min(vals)) * 1.1, top * 1.25 if top > 0 else 1.0)
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height(),
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    n = json.loads((results_dir / "eval_depth.json").read_text())["n_episodes"]
    fig.suptitle(
        f"SemCost-SplatNav: depth-only vs depth+semantic_cost  "
        f"(PPO, {n} fixed eval maps)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    out = figures_dir / "metrics_comparison.png"
    fig.savefig(out, dpi=150)
    print(f"[plot] -> {out}")


if __name__ == "__main__":
    main()
