"""Post-hoc DINOv2 semantic cost map skeleton."""

from __future__ import annotations

from pathlib import Path
from typing import Any


GPU_REQUIRED = "需要 GPU 实例,见 docs/AWS_FLAGSHIP_PLAN.md"


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
            raise NotImplementedError(GPU_REQUIRED) from exc

        self.torch = torch
        self.F = F
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.margin = float(margin)
        self.pca = torch.as_tensor(self._load_npy(pca_path), dtype=torch.float32)
        self.bad_proto = F.normalize(
            torch.as_tensor(self._load_npy(bad_proto_path), dtype=torch.float32),
            dim=0,
        )

    @staticmethod
    def _load_npy(path: str | Path) -> Any:
        try:
            import numpy as np
        except Exception as exc:
            raise NotImplementedError(GPU_REQUIRED) from exc
        return np.load(path)

    def __call__(self, rgb: Any) -> Any:
        torch = self.torch
        F = self.F
        if not torch.is_tensor(rgb):
            rgb = torch.as_tensor(rgb, dtype=torch.float32)
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("expected RGB tensor shaped (N,3,H,W)")

        target_hw = rgb.shape[-2:]
        resized = F.interpolate(rgb, size=(224, 224), mode="bilinear", align_corners=False)
        inputs = self.processor(images=list(resized), return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            patch = outputs.last_hidden_state[:, 1:, :]
            feats = F.normalize(patch, dim=-1)
            reduced = F.normalize(feats @ self.pca.to(feats.device), dim=-1)
            proto = self.bad_proto.to(reduced.device)
            cost = torch.relu((reduced * proto).sum(dim=-1) - self.margin)
            cost = cost.reshape(rgb.shape[0], 1, 16, 16)
            return F.interpolate(cost, size=target_hw, mode="bilinear", align_corners=False)

