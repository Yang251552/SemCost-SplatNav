"""Leave-one-out parity for the DINO carpet cost.

The standard parity (eval_parity.py) builds the bad prototype from all labeled
frames and evaluates IoU on those same frames -- a few-shot train==test setting.
This script instead, for each labeled frame, builds the prototype from the OTHER
N-1 frames and evaluates IoU on the held-out frame. A LOO IoU close to the full
parity shows the carpet cost generalizes across frames/viewpoints rather than
overfitting the exact masks used to build the prototype.

Feature flow matches dino_cost.DinoCostMapper / build_bad_proto.py exactly:
L2-norm patch features -> subtract PCA mean -> @ pca -> L2-norm.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leave-one-out DINO carpet parity.")
    parser.add_argument("--masks-dir", default=str(ROOT / "assets" / "p2_arkit_render" / "masks"))
    parser.add_argument("--rgb-dir", default=str(ROOT / "assets" / "p2_arkit_render" / "rgb"))
    parser.add_argument("--pca", default=str(ROOT / "results" / "dino_pca64.npy"))
    parser.add_argument("--out", default=str(ROOT / "results" / "parity_loo.json"))
    parser.add_argument("--model", default="facebook/dinov2-small")
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--patch-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoImageProcessor, AutoModel
    except Exception as exc:  # pragma: no cover - env guard
        print(f"[loo-parity] missing dependency: {exc}")
        return 2

    mask_paths = sorted(Path(args.masks_dir).glob("*.npy"))
    if len(mask_paths) < 3:
        raise SystemExit("need >=3 labeled frames for leave-one-out")
    if not Path(args.pca).exists():
        raise SystemExit(f"PCA projection not found: {args.pca}")

    pca = torch.as_tensor(np.load(args.pca), dtype=torch.float32)
    mean_sib = Path(args.pca).parent / f"{Path(args.pca).stem}_mean.npy"
    pca_mean = (
        torch.as_tensor(np.load(mean_sib), dtype=torch.float32) if mean_sib.exists() else None
    )
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to("cpu").eval()
    for param in model.parameters():
        param.requires_grad_(False)

    def load_224(fid: str) -> np.ndarray:
        im = Image.open(Path(args.rgb_dir) / f"{fid}.png").convert("RGB").resize((224, 224), Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8)

    def downsample(mask: np.ndarray) -> np.ndarray:
        im = Image.fromarray(mask.astype(np.uint8) * 255).resize((16, 16), Image.BILINEAR)
        return np.asarray(im, dtype=np.float32) / 255.0

    # Cache reduced patch features + patch-level mask for each labeled frame.
    cache: dict[str, dict] = {}
    with torch.no_grad():
        for mp in mask_paths:
            fid = mp.stem
            mask = np.load(mp).astype(bool)
            inputs = processor(images=[load_224(fid)], return_tensors="pt")
            outputs = model(pixel_values=inputs["pixel_values"])
            patch = outputs.last_hidden_state[:, 1:, :]
            feats = F.normalize(patch, dim=-1)
            if pca_mean is not None:
                feats = feats - pca_mean
            reduced = F.normalize(feats @ pca, dim=-1).reshape(16, 16, -1)
            patch_mask = torch.as_tensor(downsample(mask) >= args.patch_threshold)
            cache[fid] = {"reduced": reduced, "patch_mask": patch_mask, "gt": mask, "hw": mask.shape}

    def iou(cost: np.ndarray, gt: np.ndarray, tau: float):
        pred = cost >= tau
        union = np.logical_or(pred, gt).sum()
        return float(np.logical_and(pred, gt).sum() / union) if union else None

    per_frame: dict[str, float | None] = {}
    fids = list(cache)
    with torch.no_grad():
        for held in fids:
            train = [f for f in fids if f != held]
            pooled = torch.cat([cache[f]["reduced"][cache[f]["patch_mask"]] for f in train], dim=0)
            proto = F.normalize(pooled.mean(dim=0), dim=0)
            reduced = cache[held]["reduced"]
            cost = torch.relu((reduced * proto).sum(dim=-1) - args.margin).reshape(1, 1, 16, 16)
            cost_up = F.interpolate(cost, size=cache[held]["hw"], mode="bilinear", align_corners=False)
            val = iou(cost_up[0, 0].numpy(), cache[held]["gt"], args.threshold)
            per_frame[held] = val
            print(f"[loo-parity] held={held} IoU={val:.3f} (proto from {len(train)} frames)")

    vals = [v for v in per_frame.values() if v is not None]
    macro = float(np.mean(vals)) if vals else None
    payload = {
        "hazard_class": "carpet",
        "mode": "leave-one-out",
        "threshold": args.threshold,
        "margin": args.margin,
        "num_frames": len(fids),
        "per_frame_IoU": per_frame,
        "LOO_IoU": macro,
        "model": args.model,
        "note": "prototype built from N-1 frames, evaluated on the held-out frame; with only 5 labeled frames this shows small-sample cross-frame consistency (not strong generalization), complementing the full-proto train==test parity in parity_report.json",
    }
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[loo-parity] LOO_IoU={macro:.4f} wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
