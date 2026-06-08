"""Fit a PCA projection for DINOv2 patch features over rendered RGB frames."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit PCA64 for DINOv2 patch features.")
    parser.add_argument("--rgb-dir", default=str(ROOT / "assets" / "p2_arkit_render" / "rgb"))
    parser.add_argument("--model", default="facebook/dinov2-small")
    parser.add_argument("--n-components", type=int, default=64)
    parser.add_argument("--out", default=str(ROOT / "results" / "dino_pca64.npy"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_rgb_batch(paths: list[Path]) -> list[np.ndarray]:
    images = []
    for path in paths:
        image = Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR)
        images.append(np.asarray(image, dtype=np.uint8))
    return images


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)
    try:
        import torch
        import torch.nn.functional as F
        from sklearn.decomposition import PCA
        from transformers import AutoImageProcessor, AutoModel
    except Exception as exc:
        print(f"[fit-dino-pca] missing dependency: {exc}", file=sys.stderr)
        return 2

    torch.manual_seed(args.seed)
    rgb_paths = sorted(Path(args.rgb_dir).glob("*.png"))
    if not rgb_paths:
        raise SystemExit(f"No PNG frames found in {args.rgb_dir}")

    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to("cpu").eval()
    for param in model.parameters():
        param.requires_grad_(False)

    all_features = []
    with torch.no_grad():
        for start in range(0, len(rgb_paths), args.batch_size):
            batch_paths = rgb_paths[start : start + args.batch_size]
            inputs = processor(images=load_rgb_batch(batch_paths), return_tensors="pt")
            outputs = model(pixel_values=inputs["pixel_values"].to("cpu"))
            patch = outputs.last_hidden_state[:, 1:, :]
            features = F.normalize(patch, dim=-1).cpu().numpy()
            all_features.append(features.reshape(-1, features.shape[-1]))
            done = min(start + len(batch_paths), len(rgb_paths))
            if done == len(rgb_paths) or done % 10 == 0:
                print(f"[fit-dino-pca] processed {done}/{len(rgb_paths)} frames")

    matrix = np.concatenate(all_features, axis=0)
    pca = PCA(n_components=args.n_components, random_state=args.seed)
    pca.fit(matrix)
    projection = pca.components_.T.astype(np.float32)
    mean = pca.mean_.astype(np.float32)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, projection)
    mean_path = out_path.parent / f"{out_path.stem}_mean.npy"
    np.save(mean_path, mean)
    loaded = np.load(out_path)
    assert loaded.shape == (384, args.n_components), loaded.shape
    print(f"[fit-dino-pca] wrote {out_path}")
    print(f"[fit-dino-pca] wrote {mean_path}")
    print(f"[fit-dino-pca] projection_shape={loaded.shape}")
    print(f"[fit-dino-pca] explained_variance_sum={pca.explained_variance_ratio_.sum():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
