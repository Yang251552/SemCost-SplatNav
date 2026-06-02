"""Seeding helpers for reproducible training and evaluation."""

from __future__ import annotations

import random

import numpy as np


def set_global_seeds(seed: int) -> None:
    """Seed Python, NumPy and (if available) PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
