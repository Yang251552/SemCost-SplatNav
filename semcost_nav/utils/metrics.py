"""Episode metric aggregation shared by evaluation and plotting."""

from __future__ import annotations

from typing import Any

import numpy as np

PRIMARY_METRICS = (
    "success_rate",
    "collision_rate",
    "bad_region_time",
    "path_length",
    "average_return",
)


def aggregate_episode_metrics(episodes: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate per-episode records into the five primary metrics.

    Args:
        episodes: list of per-episode dicts with keys ``success`` (bool),
            ``collisions`` (int), ``bad_region_time`` (int), ``path_length``
            (int), ``ep_return`` (float).

    Returns:
        Dict with the five primary metrics plus episode count.
    """
    n = len(episodes)
    if n == 0:
        return {k: 0.0 for k in PRIMARY_METRICS} | {"n_episodes": 0}
    success = np.array([float(bool(e["success"])) for e in episodes])
    collided = np.array([1.0 if int(e["collisions"]) > 0 else 0.0 for e in episodes])
    bad = np.array([float(e["bad_region_time"]) for e in episodes])
    plen = np.array([float(e["path_length"]) for e in episodes])
    ret = np.array([float(e["ep_return"]) for e in episodes])
    # path_length over successful episodes only (efficiency of paths that
    # actually reached the goal); falls back to all-episode mean if none succeed.
    succ_mask = success > 0.5
    plen_success = float(plen[succ_mask].mean()) if succ_mask.any() else float("nan")
    return {
        # collision_rate = fraction of episodes with >=1 collision (not count).
        # bad_region_time = mean steps spent on hazard cells per episode.
        # path_length = mean number of forward moves over ALL episodes.
        "success_rate": float(success.mean()),
        "collision_rate": float(collided.mean()),
        "bad_region_time": float(bad.mean()),
        "path_length": float(plen.mean()),
        "path_length_success": plen_success,
        "average_return": float(ret.mean()),
        "n_episodes": int(n),
    }


def summarize_metrics(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate metrics and attach simple dispersion (std) for the primaries."""
    agg = aggregate_episode_metrics(episodes)
    if episodes:
        agg["bad_region_time_std"] = float(
            np.std([float(e["bad_region_time"]) for e in episodes])
        )
        agg["path_length_std"] = float(np.std([float(e["path_length"]) for e in episodes]))
        agg["return_std"] = float(np.std([float(e["ep_return"]) for e in episodes]))
    return agg
