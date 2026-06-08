"""Make a 2x2 RGB / depth / semantic-cost / avoid-mask figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make RGB/depth/cost/avoid quad figure.")
    parser.add_argument("--frame", type=int, default=None)
    parser.add_argument("--cost-maps", default=str(ROOT / "results" / "semantic_cost_maps.npz"))
    parser.add_argument("--rgb-dir", default=str(ROOT / "assets" / "p2_arkit_render" / "rgb"))
    parser.add_argument("--depth-dir", default=str(ROOT / "assets" / "p2_arkit_render" / "depth"))
    parser.add_argument("--masks-dir", default=str(ROOT / "assets" / "p2_arkit_render" / "masks"))
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--out", default=str(ROOT / "figures" / "rgbd_cost_avoid_quad.png"))
    return parser.parse_args()


def choose_frame(args: argparse.Namespace) -> tuple[str, float | None]:
    if args.frame is not None:
        return f"{args.frame:04d}", None
    report_path = ROOT / "results" / "parity_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        per_frame = report.get("per_frame_IoU") or {}
        numeric = {key: value for key, value in per_frame.items() if value is not None}
        if numeric:
            frame_id = max(numeric, key=numeric.get)
            return frame_id, float(numeric[frame_id])
    masks = sorted(Path(args.masks_dir).glob("*.npy"))
    if masks:
        return masks[0].stem, None
    raise SystemExit("No --frame, parity report, or masks available to choose a frame.")


def main() -> int:
    args = parse_args()
    frame_id, reported_iou = choose_frame(args)
    if not Path(args.cost_maps).exists():
        raise SystemExit(f"Cost maps not found: {args.cost_maps}")
    data = np.load(args.cost_maps)
    cost_maps = data["cost_maps"]
    raw_names = data["frame_names"] if "frame_names" in data.files else np.arange(cost_maps.shape[0])
    frame_names = [str(name) for name in raw_names]
    frame_to_index = {Path(name).stem: idx for idx, name in enumerate(frame_names)}
    if frame_id not in frame_to_index:
        raise SystemExit(f"Frame {frame_id} not found in cost maps.")

    rgb = np.asarray(Image.open(Path(args.rgb_dir) / f"{frame_id}.png").convert("RGB"))
    depth = np.load(Path(args.depth_dir) / f"{frame_id}.npy")
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    cost = np.asarray(cost_maps[frame_to_index[frame_id], 0], dtype=np.float32)
    avoid = cost >= args.threshold
    mask_path = Path(args.masks_dir) / f"{frame_id}.npy"
    gt = np.load(mask_path).astype(bool) if mask_path.exists() else None
    if reported_iou is None and gt is not None:
        union = np.logical_or(avoid, gt).sum()
        reported_iou = float(np.logical_and(avoid, gt).sum() / union) if union else None

    fig, axes = plt.subplots(2, 2, figsize=(8, 6), dpi=150)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title(f"RGB frame {frame_id}")
    im_depth = axes[0, 1].imshow(depth, cmap="jet")
    axes[0, 1].set_title("Depth (m)")
    fig.colorbar(im_depth, ax=axes[0, 1], fraction=0.046, pad=0.04)
    im_cost = axes[1, 0].imshow(cost, cmap="viridis", vmin=0.0, vmax=max(1.0, float(cost.max())))
    axes[1, 0].set_title("DINO semantic cost")
    fig.colorbar(im_cost, ax=axes[1, 0], fraction=0.046, pad=0.04)
    axes[1, 1].imshow(avoid, cmap="gray", vmin=0, vmax=1)
    if gt is not None:
        axes[1, 1].contour(gt.astype(float), levels=[0.5], colors="red", linewidths=1.0)
    title = f"Avoid mask tau={args.threshold:g}"
    if reported_iou is not None:
        title += f", IoU={reported_iou:.3f}"
    axes[1, 1].set_title(title)
    for ax in axes.ravel():
        ax.set_axis_off()
    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[quad-figure] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
