"""Render rollout visualizations comparing the two policies.

Produces:
  * figures/rollout_comparison.gif (and .mp4 if ffmpeg is available):
    a side-by-side animation of the depth-only vs depth+semantic policy on the
    SAME map/seed.
  * figures/scene_depth_cost_trajectory.png: a static 4-panel figure showing
    the scene, the geometry/depth channel, the semantic-cost channel, and the
    two trajectories overlaid.

Usage:
    python scripts/render_rollout.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO  # noqa: E402

from semcost_nav.utils import experiment as ex  # noqa: E402


def load_model(obs: str, device: str = "cpu") -> PPO:
    path = ROOT / "checkpoints" / f"ppo_{obs}.zip"
    if not path.exists():
        raise FileNotFoundError(f"missing model {path}; train first")
    return PPO.load(path, device=device)


def rollout(model: PPO, obs_mode: str, seed: int):
    """Run one deterministic episode; return frames, trajectory and summary."""
    env = ex.make_single_env(obs_mode)
    obs, info = env.reset(seed=seed)
    frames = [env.render()]
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(int(action))
        frames.append(env.render())
        done = term or trunc
    summary = {
        "success": bool(info["success"]),
        "bad_region_time": int(info["bad_region_time"]),
        "path_length": int(info["path_length"]),
        "collisions": int(info["collisions"]),
        "trajectory": list(env.trajectory),
        "layout": env.layout,
        "geo_abs": env._geo_abs.copy(),
    }
    env.close()
    return frames, summary


def pick_seed(m_depth, m_sem, candidate_seeds) -> int:
    """Pick a seed that best illustrates the effect: depth enters hazard, the
    semantic policy avoids it (and ideally both succeed). Falls back to first."""
    best = candidate_seeds[0]
    for s in candidate_seeds:
        _, sd = rollout(m_depth, "depth", s)
        _, ss = rollout(m_sem, "depth_semantic", s)
        if sd["success"] and ss["success"] and sd["bad_region_time"] > 0 and ss[
            "bad_region_time"
        ] == 0:
            return s
    return best


def _label_frame(frame: np.ndarray, text: str) -> np.ndarray:
    """Add a small text banner above a frame using matplotlib rasterization."""
    h, w, _ = frame.shape
    fig = plt.figure(figsize=(w / 100, (h + 28) / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    canvas = np.full((h + 28, w, 3), 255, dtype=np.uint8)
    canvas[28:, :, :] = frame
    ax.imshow(canvas)
    ax.text(6, 18, text, fontsize=9, color="black", va="center")
    fig.canvas.draw()
    out = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return out


def make_side_by_side(frames_d, frames_s, sum_d, sum_s):
    n = max(len(frames_d), len(frames_s))

    def pad(frames):
        return frames + [frames[-1]] * (n - len(frames))

    fd, fs = pad(frames_d), pad(frames_s)
    combined = []
    gap = np.full((fd[0].shape[0], 8, 3), 255, dtype=np.uint8)
    for i in range(n):
        left = _label_frame(fd[i], f"depth-only  bad-region so far: see end={sum_d['bad_region_time']}")
        right = _label_frame(fs[i], f"depth+semantic  end={sum_s['bad_region_time']}")
        gap2 = np.full((left.shape[0], 8, 3), 255, dtype=np.uint8)
        combined.append(np.concatenate([left, gap2, right], axis=1))
    return combined


def save_scene_figure_with_traj(sum_d, sum_s, out_path: Path) -> None:
    layout = sum_s["layout"]
    geo = sum_s["geo_abs"]
    hazard = layout.hazard

    scene = np.full((*geo.shape, 3), 0.96)
    scene[hazard > 0] = [0.86, 0.45, 0.45]
    scene[layout.walls == 1] = [0.25, 0.25, 0.25]
    sr, sc = layout.start
    gr, gc = layout.goal

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))

    axes[0].imshow(scene)
    axes[0].plot(sc, sr, "o", color="#1f4fdb", ms=8)
    axes[0].plot(gc, gr, "*", color="#28b347", ms=14)
    axes[0].set_title("(a) Scene\nhazard band + safe gap")

    axes[1].imshow(geo, cmap="bone", vmin=0, vmax=1)
    axes[1].set_title("(b) Geometry/depth channel\nhazard is INVISIBLE")

    im = axes[2].imshow(hazard, cmap="Reds", vmin=0, vmax=1)
    axes[2].set_title("(c) Semantic-cost channel\nreveals hazard + gap")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(scene)
    for sm, color, lab in (
        (sum_d, "#444444", f"depth-only (bad={sum_d['bad_region_time']})"),
        (sum_s, "#1f4fdb", f"depth+sem (bad={sum_s['bad_region_time']})"),
    ):
        traj = np.array(sm["trajectory"])
        axes[3].plot(traj[:, 1], traj[:, 0], "-", color=color, lw=2, label=lab, alpha=0.9)
    axes[3].plot(sc, sr, "o", color="#1f4fdb", ms=8)
    axes[3].plot(gc, gr, "*", color="#28b347", ms=14)
    axes[3].legend(fontsize=8, loc="lower center")
    axes[3].set_title("(d) Trajectories\ndepth-only crosses hazard, semantic detours")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=None, help="fixed map seed to render")
    p.add_argument("--figures-dir", default=str(ROOT / "figures"))
    p.add_argument("--fps", type=int, default=6)
    args = p.parse_args()

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    m_depth = load_model("depth")
    m_sem = load_model("depth_semantic")

    if args.seed is None:
        seed = pick_seed(m_depth, m_sem, ex.eval_seeds(40))
    else:
        seed = args.seed
    print(f"[render] using seed={seed}")

    frames_d, sum_d = rollout(m_depth, "depth", seed)
    frames_s, sum_s = rollout(m_sem, "depth_semantic", seed)

    combined = make_side_by_side(frames_d, frames_s, sum_d, sum_s)
    gif_path = figures_dir / "rollout_comparison.gif"
    imageio.mimsave(gif_path, combined, duration=1.0 / args.fps, loop=0)
    print(f"[render] -> {gif_path}")

    try:
        mp4_path = figures_dir / "rollout_comparison.mp4"
        imageio.mimsave(mp4_path, combined, fps=args.fps)
        print(f"[render] -> {mp4_path}")
    except Exception as e:  # noqa: BLE001 - ffmpeg optional
        print(f"[render] mp4 skipped ({e})")

    scene_path = figures_dir / "scene_depth_cost_trajectory.png"
    save_scene_figure_with_traj(sum_d, sum_s, scene_path)
    print(f"[render] -> {scene_path}")

    print(
        f"[render] depth bad_region_time={sum_d['bad_region_time']} "
        f"success={sum_d['success']} | "
        f"semantic bad_region_time={sum_s['bad_region_time']} "
        f"success={sum_s['success']}"
    )


if __name__ == "__main__":
    main()
