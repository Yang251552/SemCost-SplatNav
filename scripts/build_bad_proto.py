"""Build a few-shot carpet hazard prototype from labeled masks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build bad DINO prototype from masks.")
    parser.add_argument("--masks-dir", default=str(ROOT / "assets" / "p2_arkit_render" / "masks"))
    parser.add_argument("--rgb-dir", default=str(ROOT / "assets" / "p2_arkit_render" / "rgb"))
    parser.add_argument("--pca", default=str(ROOT / "results" / "dino_pca64.npy"))
    parser.add_argument("--out", default=str(ROOT / "results" / "bad_proto.npy"))
    parser.add_argument("--model", default="facebook/dinov2-small")
    parser.add_argument("--patch-threshold", type=float, default=0.5)
    return parser.parse_args()


def downsample_mask(mask: np.ndarray) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    small = image.resize((16, 16), Image.BILINEAR)
    return np.asarray(small, dtype=np.float32) / 255.0


def load_rgb_224(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def main() -> int:
    args = parse_args()
    mask_paths = sorted(Path(args.masks_dir).glob("*.npy"))
    if not mask_paths:
        print(
            "No masks found. First run: python scripts/label_carpet_mask.py --frame <X>",
            file=sys.stderr,
        )
        return 2
    if not Path(args.pca).exists():
        raise SystemExit(f"PCA projection not found: {args.pca}")

    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoImageProcessor, AutoModel
    except Exception as exc:
        print(f"[build-bad-proto] missing dependency: {exc}", file=sys.stderr)
        return 2

    pca = torch.as_tensor(np.load(args.pca), dtype=torch.float32)
    pca_path_obj = Path(args.pca)
    mean_sibling = pca_path_obj.parent / f"{pca_path_obj.stem}_mean.npy"
    pca_mean = (
        torch.as_tensor(np.load(mean_sibling), dtype=torch.float32)
        if mean_sibling.exists()
        else None
    )
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to("cpu").eval()
    for param in model.parameters():
        param.requires_grad_(False)

    selected = []
    used_frames = 0
    with torch.no_grad():
        for mask_path in mask_paths:
            frame_id = mask_path.stem
            rgb_path = Path(args.rgb_dir) / f"{frame_id}.png"
            if not rgb_path.exists():
                print(f"[build-bad-proto] skipping {frame_id}: missing RGB")
                continue
            mask = np.load(mask_path).astype(bool)
            patch_mask = downsample_mask(mask) >= args.patch_threshold
            count = int(patch_mask.sum())
            if count == 0:
                print(f"[build-bad-proto] {frame_id}: 0 hazard patches")
                continue
            inputs = processor(images=[load_rgb_224(rgb_path)], return_tensors="pt")
            outputs = model(pixel_values=inputs["pixel_values"].to("cpu"))
            patch = outputs.last_hidden_state[:, 1:, :]
            features = F.normalize(patch, dim=-1)
            if pca_mean is not None:
                features = features - pca_mean
            features = F.normalize(features @ pca, dim=-1).reshape(16, 16, -1)
            selected.append(features[torch.as_tensor(patch_mask)].cpu().numpy())
            used_frames += 1
            print(f"[build-bad-proto] {frame_id}: hazard_patches={count}")

    if not selected:
        raise SystemExit("No hazard patches selected from masks; lower --patch-threshold or relabel.")
    stacked = np.concatenate(selected, axis=0)
    proto = stacked.mean(axis=0)
    norm = float(np.linalg.norm(proto))
    if norm <= 0.0:
        raise SystemExit("Prototype norm is zero.")
    proto = (proto / norm).astype(np.float32)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, proto)
    print(f"[build-bad-proto] used_frames={used_frames} patches={stacked.shape[0]}")
    print(f"[build-bad-proto] wrote {out_path}")
    print(f"[build-bad-proto] proto_l2={np.linalg.norm(proto):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
