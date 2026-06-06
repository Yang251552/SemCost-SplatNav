"""Local GaussGym observation token extensions.

GaussGym upstream does not define SEMANTIC_COST. This module keeps the
registration lazy so CPU dry-runs never import IsaacGym or GaussGym.
"""

from __future__ import annotations

from typing import Any


SEMANTIC_COST_TOKEN = "SEMANTIC_COST"


def semantic_cost_observation(hazard_mask: Any = None, rgb_latent: Any = None) -> Any:
    """Mock semantic-cost observation for Stage 2 dry-runs.

    The function intentionally accepts only hazard-mask or RGB-latent inputs.
    It must not depend on target, goal, agent pose, or command state.
    """
    source = hazard_mask if hazard_mask is not None else rgb_latent
    if source is None:
        try:
            import numpy as np

            return np.zeros((1, 1), dtype=np.float32)
        except Exception:
            return [[0.0]]

    try:
        import numpy as np

        arr = np.asarray(source)
        if arr.ndim >= 2:
            return np.zeros(arr.shape[-2:], dtype=np.float32)
        return np.zeros_like(arr, dtype=np.float32)
    except Exception:
        return source * 0


def register_semantic_cost_token(observation_groups_module: Any) -> Any:
    """Attach SEMANTIC_COST to a loaded GaussGym observation_groups module."""
    observation_cls = getattr(observation_groups_module, "Observation")
    token = observation_cls(
        name="semantic_cost",
        func=semantic_cost_observation,
    )
    setattr(observation_groups_module, SEMANTIC_COST_TOKEN, token)
    return token

