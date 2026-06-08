"""Evaluate DINO semantic cost maps against human carpet masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate cost map / mask IoU.")
    parser.add_argument("--cost-maps", default=str(ROOT / "results" / "semantic_cost_maps.npz"))
    parser.add_argument("--masks-dir", default=str(ROOT / "assets" / "p2_arkit_render" / "masks"))
    parser.add_argument("--out", default=str(ROOT / "results" / "parity_report.json"))
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--threshold-scan", action="store_true")
    return parser.parse_args()


def iou_at(cost: np.ndarray, gt: np.ndarray, threshold: float) -> float | None:
    pred = cost >= threshold
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return None
    inter = np.logical_and(pred, gt).sum()
    return float(inter / union)


def mean_iou(per_frame: dict[str, float | None]) -> float | None:
    values = [value for value in per_frame.values() if value is not None]
    if not values:
        return None
    return float(np.mean(values))


def main() -> int:
    args = parse_args()
    cost_path = Path(args.cost_maps)
    if not cost_path.exists():
        raise SystemExit(f"Cost maps not found: {cost_path}")
    mask_paths = sorted(Path(args.masks_dir).glob("*.npy"))
    if not mask_paths:
        raise SystemExit(
            "No masks found. First run: python scripts/label_carpet_mask.py --frame <X>"
        )

    data = np.load(cost_path)
    cost_maps = data["cost_maps"]
    raw_names = data["frame_names"] if "frame_names" in data.files else np.arange(cost_maps.shape[0])
    frame_names = [str(name) for name in raw_names]
    frame_to_index = {Path(name).stem: idx for idx, name in enumerate(frame_names)}

    per_frame: dict[str, float | None] = {}
    for mask_path in mask_paths:
        frame_id = mask_path.stem
        if frame_id not in frame_to_index:
            print(f"[eval-parity] skipping {frame_id}: no matching cost map")
            continue
        gt = np.load(mask_path).astype(bool)
        cost = np.asarray(cost_maps[frame_to_index[frame_id], 0], dtype=np.float32)
        if cost.shape != gt.shape:
            raise SystemExit(f"Shape mismatch for {frame_id}: cost={cost.shape} gt={gt.shape}")
        per_frame[frame_id] = iou_at(cost, gt, args.threshold)

    if not per_frame:
        raise SystemExit("No labeled frames matched cost maps.")

    scan = []
    best_threshold = None
    best_iou = None
    if args.threshold_scan:
        for tau in np.linspace(0.0, 0.5, 21):
            tau_per_frame = {
                frame_id: iou_at(
                    np.asarray(cost_maps[frame_to_index[frame_id], 0], dtype=np.float32),
                    np.load(Path(args.masks_dir) / f"{frame_id}.npy").astype(bool),
                    float(tau),
                )
                for frame_id in per_frame
            }
            tau_iou = mean_iou(tau_per_frame)
            scan.append({"tau": float(tau), "IoU": tau_iou})
            if tau_iou is not None and (best_iou is None or tau_iou > best_iou):
                best_iou = tau_iou
                best_threshold = float(tau)

    macro = mean_iou(per_frame)
    source = "few-shot mask-pooled DINOv2 ViT-S patches"
    has_stub_marker = any((Path(args.masks_dir) / f"{path.stem}.auto_stub").exists() for path in mask_paths)
    if has_stub_marker:
        source = "AUTO_STUB - replace by real human labels; few-shot mask-pooled DINOv2 ViT-S patches"
    payload = {
        "hazard_class": "carpet",
        "num_labeled": len(per_frame),
        "threshold": args.threshold,
        "IoU": macro,
        "per_frame_IoU": per_frame,
        "best_threshold": best_threshold,
        "best_IoU": best_iou,
        "threshold_scan": scan,
        "bad_proto_source": source,
        "pca_components": 64,
        "model": "facebook/dinov2-small",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    verdict = macro is not None and macro >= 0.5
    slow = sorted(
        ((frame_id, value) for frame_id, value in per_frame.items() if value is not None),
        key=lambda item: item[1],
    )[:3]
    print(f"[eval-parity] IoU={macro} verdict={'>=0.5' if verdict else '<0.5'}")
    print(f"[eval-parity] lowest_frames={slow}")
    print(f"[eval-parity] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
