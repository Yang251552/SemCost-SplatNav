"""Procedural map generation for SemCost-SplatNav.

The map is a small square arena enclosed by walls. A horizontal *hazard band*
crosses the arena with a single safe *gap* whose column is randomized every
episode. The band is NOT a wall: it does not block motion and leaves no
geometric trace, so an agent that only observes geometry/depth cannot tell
where the safe gap is. The hazard location is only revealed through the
semantic-cost channel.

This is the core of the experiment: geometry alone is ambiguous about where the
hazard is; semantic cost resolves that ambiguity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Cell codes for the geometry grid.
FREE = 0
WALL = 1


@dataclass
class MapLayout:
    """A single generated arena instance.

    Attributes:
        walls: (H, W) int array, 1 where a wall blocks motion, else 0.
        hazard: (H, W) float array in [0, 1], semantic hazard intensity.
        start: (row, col) start cell of the agent.
        goal: (row, col) goal cell.
        band_rows: tuple of rows occupied by the hazard band.
        gap_cols: tuple of columns that form the safe gap in the band.
    """

    walls: np.ndarray
    hazard: np.ndarray
    start: tuple[int, int]
    goal: tuple[int, int]
    band_rows: tuple[int, ...]
    gap_cols: tuple[int, ...]

    @property
    def height(self) -> int:
        return int(self.walls.shape[0])

    @property
    def width(self) -> int:
        return int(self.walls.shape[1])


@dataclass
class MapConfig:
    """Configuration for :func:`generate_map`.

    Defaults describe an 11x11 arena with a 3-row hazard band and a 3-wide gap.
    """

    size: int = 11
    band_rows: tuple[int, ...] = (4, 5, 6)
    gap_width: int = 3
    hazard_value: float = 1.0
    goal_col: int | None = None  # default: horizontal centre
    randomize_start_col: bool = True


def generate_map(rng: np.random.Generator, config: MapConfig | None = None) -> MapLayout:
    """Generate one arena instance.

    Args:
        rng: NumPy random generator controlling the gap column and start column.
        config: Map configuration; defaults to :class:`MapConfig`.

    Returns:
        A :class:`MapLayout`. The hazard band and gap are the only sources of
        per-episode variation; walls, goal and band rows are fixed by config.
    """
    if config is None:
        config = MapConfig()

    size = config.size
    walls = np.zeros((size, size), dtype=np.int8)
    # Enclosing wall border.
    walls[0, :] = WALL
    walls[-1, :] = WALL
    walls[:, 0] = WALL
    walls[:, -1] = WALL

    interior_lo, interior_hi = 1, size - 2  # inclusive interior range

    goal_col = config.goal_col if config.goal_col is not None else size // 2
    goal = (interior_lo, int(goal_col))

    if config.randomize_start_col:
        start_col = int(rng.integers(interior_lo, interior_hi + 1))
    else:
        start_col = size // 2
    start = (interior_hi, start_col)

    # Hazard band across the full interior width, minus a contiguous safe gap.
    hazard = np.zeros((size, size), dtype=np.float32)
    n_interior_cols = interior_hi - interior_lo + 1
    gap_width = min(config.gap_width, n_interior_cols)
    gap_start = int(rng.integers(interior_lo, interior_hi - gap_width + 2))
    gap_cols = tuple(range(gap_start, gap_start + gap_width))

    band_rows = tuple(r for r in config.band_rows if interior_lo <= r <= interior_hi)
    # Leakage guard: the goal must never sit on a band row. Otherwise the forced
    # zero at the goal cell (below) would punch a goal-shaped hole in the hazard
    # field, leaking the target location through the semantic-cost channel.
    if goal[0] in band_rows:
        raise ValueError(
            "goal row must not be a hazard band row (would leak goal via "
            f"semantic_cost): goal_row={goal[0]}, band_rows={band_rows}"
        )
    for r in band_rows:
        for c in range(interior_lo, interior_hi + 1):
            if c not in gap_cols:
                hazard[r, c] = config.hazard_value

    # Safety: never place hazard on start or goal cells.
    hazard[start] = 0.0
    hazard[goal] = 0.0

    return MapLayout(
        walls=walls,
        hazard=hazard,
        start=start,
        goal=goal,
        band_rows=band_rows,
        gap_cols=gap_cols,
    )
