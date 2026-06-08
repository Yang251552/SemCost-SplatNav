"""Post-hoc DINOv2 semantic cost maps from RGB frames."""

from __future__ import annotations

from pathlib import Path
from typing import Any


DINO_DEPENDENCIES_REQUIRED = "需要安装 transformers 和 safetensors 才能本机 CPU 跑 DINOv2"


class DinoCostMapper:
    """Map RGB frames to soft hazard-cost maps with DINOv2 patch features."""

    def __init__(
        self,
        pca_path: str | Path,
        bad_proto_path: str | Path,
        margin: float = 0.2,
        model_name: str = "facebook/dinov2-small",
    ) -> None:
        try:
            import torch
            import torch.nn.functional as F
            from transformers import AutoImageProcessor, AutoModel
        except Exception as exc:
            raise NotImplementedError(DINO_DEPENDENCIES_REQUIRED) from exc

        self.torch = torch
        self.F = F
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to("cpu").eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.margin = float(margin)
        self.pca = torch.as_tensor(self._load_npy(pca_path), dtype=torch.float32)
        pca_path_obj = Path(pca_path)
        mean_sibling = pca_path_obj.parent / f"{pca_path_obj.stem}_mean.npy"
        if mean_sibling.exists():
            self.pca_mean = torch.as_tensor(self._load_npy(mean_sibling), dtype=torch.float32)
        else:
            self.pca_mean = None
        self.bad_proto = F.normalize(
            torch.as_tensor(self._load_npy(bad_proto_path), dtype=torch.float32),
            dim=0,
        )

    @staticmethod
    def _load_npy(path: str | Path) -> Any:
        try:
            import numpy as np
        except Exception as exc:
            raise NotImplementedError(DINO_DEPENDENCIES_REQUIRED) from exc
        return np.load(path)

    def __call__(self, rgb: Any) -> Any:
        torch = self.torch
        F = self.F
        if not torch.is_tensor(rgb):
            rgb = torch.as_tensor(rgb, dtype=torch.float32)
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("expected RGB tensor shaped (N,3,H,W)")

        output_hw = rgb.shape[-2:]
        resized = F.interpolate(
            rgb.to(dtype=torch.float32, device="cpu"),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )
        images = (
            resized.clamp(0.0, 1.0)
            .mul(255.0)
            .round()
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .numpy()
        )
        inputs = self.processor(images=list(images), return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            patch = outputs.last_hidden_state[:, 1:, :]
            feats = F.normalize(patch, dim=-1)
            if self.pca_mean is not None:
                feats = feats - self.pca_mean.to(feats.device)
            reduced = F.normalize(feats @ self.pca.to(feats.device), dim=-1)
            proto = self.bad_proto.to(reduced.device)
            cost = torch.relu((reduced * proto).sum(dim=-1) - self.margin)
            cost = cost.reshape(rgb.shape[0], 1, 16, 16)
            return F.interpolate(cost, size=output_hw, mode="bilinear", align_corners=False)
