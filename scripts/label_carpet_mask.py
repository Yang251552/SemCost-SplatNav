"""Label one carpet mask per frame.

Run this once per frame with a GUI to draw a polygon. On SSH/headless machines,
use --auto-stub only to exercise the pipeline; it is not a real hazard label.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactively label carpet masks.")
    parser.add_argument("--frame", type=int, required=True, help="Frame id in [0, 79].")
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "assets" / "p2_arkit_render" / "masks"),
        help="Directory for <frame_id>.npy boolean masks.",
    )
    parser.add_argument(
        "--auto-stub",
        action="store_true",
        help="Write a synthetic lower-image rectangle for pipeline smoke tests only.",
    )
    return parser.parse_args()


def rasterize_polygon(vertices: list[tuple[float, float]], width: int, height: int) -> np.ndarray:
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    draw.polygon(vertices, outline=1, fill=1)
    return np.asarray(image, dtype=bool)


def label_interactively(rgb_path: Path) -> np.ndarray:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import PolygonSelector

    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    height, width = rgb.shape[:2]
    fig, ax = plt.subplots()
    ax.imshow(rgb, interpolation="nearest")
    ax.set_title("Draw carpet polygon, then press Enter")
    ax.set_axis_off()
    vertices: list[tuple[float, float]] = []

    def onselect(points: list[tuple[float, float]]) -> None:
        vertices[:] = points

    selector = PolygonSelector(ax, onselect, useblit=True)

    def onkey(event: object) -> None:
        if getattr(event, "key", None) == "enter":
            if len(vertices) < 3:
                # Fallback: read live vertices even if the loop was not closed,
                # so pressing Enter after >=3 clicks just works.
                try:
                    pts = [(float(x), float(y)) for x, y in selector.verts]
                except Exception:
                    pts = []
                if len(pts) >= 3:
                    vertices[:] = pts
            selector.disconnect_events()
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", onkey)
    plt.show()
    if len(vertices) < 3:
        raise SystemExit("Need at least 3 polygon vertices; no mask written.")
    return rasterize_polygon(vertices, width=width, height=height)


def make_auto_stub(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    y0, y1 = int(height * 0.58), int(height * 0.88)
    x0, x1 = int(width * 0.22), int(width * 0.78)
    mask[y0:y1, x0:x1] = True
    return mask


def main() -> int:
    args = parse_args()
    if args.frame < 0 or args.frame > 79:
        raise SystemExit("--frame must be in [0, 79]")

    frame_id = f"{args.frame:04d}"
    rgb_path = ROOT / "assets" / "p2_arkit_render" / "rgb" / f"{frame_id}.png"
    if not rgb_path.exists():
        raise SystemExit(f"RGB frame not found: {rgb_path}")

    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    height, width = rgb.shape[:2]
    if args.auto_stub:
        mask = make_auto_stub(height, width)
    else:
        mask = label_interactively(rgb_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{frame_id}.npy"
    np.save(out_path, mask.astype(bool))
    marker_path = out_dir / f"{frame_id}.auto_stub"
    if args.auto_stub:
        marker_path.write_text(
            "Synthetic mask for pipeline connectivity only; replace with human polygon label.\n",
            encoding="utf-8",
        )
    elif marker_path.exists():
        marker_path.unlink()
    pixels = int(mask.sum())
    ratio = pixels / float(mask.size)
    print(f"[label-mask] wrote {out_path}")
    print(f"[label-mask] pixels={pixels} ratio={ratio:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
