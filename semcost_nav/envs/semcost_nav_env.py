"""SemCost-SplatNav Gymnasium environment.

A lightweight 2D navigation task used to study whether a semantic-cost
observation channel helps a policy avoid hazardous regions that are
geometrically invisible.

Observation modes
-----------------
* ``depth``           -> shape (1, S, S): geometry/occupancy channel only.
* ``depth_semantic``  -> shape (2, S, S): geometry + semantic-cost channels.

Both modes share an identical geometry channel, reward, dynamics, seeds and
episode structure. The ONLY difference is whether the semantic-cost channel is
present. This keeps the depth-only vs depth+semantic comparison fair.

The observation is an egocentric, heading-aligned view of the whole arena:
the agent sits at the centre and always "faces up", so the agent's heading is
encoded by the rotation of the view (this keeps the task Markovian without an
extra channel). The goal is encoded in the geometry channel; the hazard band is
encoded only in the semantic-cost channel and never reveals the goal.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from semcost_nav.maps.map_gen import MapConfig, MapLayout, generate_map

# Discrete actions.
FORWARD, TURN_LEFT, TURN_RIGHT, STOP = 0, 1, 2, 3

# Heading -> forward unit vector (row, col). Row increases downward.
_FWD = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
# Heading -> "to the agent's right" unit vector.
_RIGHT = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}

# Geometry channel encoding.
_GEO_FREE = 0.0
_GEO_GOAL = 0.5
_GEO_WALL = 1.0

VALID_OBS_MODES = ("depth", "depth_semantic")


class SemCostNavEnv(gym.Env):
    """Semantic-cost navigation environment.

    Args:
        obs_mode: ``"depth"`` or ``"depth_semantic"``.
        max_steps: episode truncation horizon.
        map_config: optional :class:`MapConfig` override.
        reward_config: optional dict overriding reward coefficients.
        render_scale: pixels per cell for ``rgb_array`` rendering.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 8}

    DEFAULT_REWARD = {
        "step_penalty": -0.01,
        "collision_penalty": -0.1,
        "hazard_penalty": -0.1,
        "goal_reward": 1.0,
        "shaping_coef": 0.05,
    }

    def __init__(
        self,
        obs_mode: str = "depth",
        max_steps: int = 80,
        map_config: MapConfig | None = None,
        reward_config: dict | None = None,
        render_mode: str | None = None,
        render_scale: int = 24,
    ) -> None:
        super().__init__()
        if obs_mode not in VALID_OBS_MODES:
            raise ValueError(f"obs_mode must be one of {VALID_OBS_MODES}, got {obs_mode!r}")
        self.obs_mode = obs_mode
        self.max_steps = int(max_steps)
        self.map_config = map_config or MapConfig()
        self.reward = dict(self.DEFAULT_REWARD)
        if reward_config:
            self.reward.update(reward_config)
        self.render_mode = render_mode
        self.render_scale = int(render_scale)

        self.size = self.map_config.size
        self.radius = self.size - 1
        self.view = 2 * self.radius + 1  # S

        n_channels = 1 if obs_mode == "depth" else 2
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(n_channels, self.view, self.view), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

        # Episode state (populated in reset).
        self.layout: MapLayout | None = None
        self._geo_abs: np.ndarray | None = None
        self.agent: tuple[int, int] = (0, 0)
        self.heading: int = 0
        self.steps: int = 0
        self.collisions: int = 0
        self.bad_region_time: int = 0
        self.path_length: int = 0
        self.ep_return: float = 0.0
        self._prev_dist: int = 0
        self._succeeded: bool = False
        self.trajectory: list[tuple[int, int]] = []

    # ------------------------------------------------------------------ utils
    def _goal_dist(self, cell: tuple[int, int]) -> int:
        gr, gc = self.layout.goal
        return abs(cell[0] - gr) + abs(cell[1] - gc)

    def _build_geo_abs(self) -> np.ndarray:
        geo = np.where(self.layout.walls == 1, _GEO_WALL, _GEO_FREE).astype(np.float32)
        geo[self.layout.goal] = _GEO_GOAL
        return geo

    def _egocentric(self, abs_field: np.ndarray, fill: float) -> np.ndarray:
        """Return an (S, S) heading-aligned, agent-centred crop of ``abs_field``.

        Forward (agent heading) maps to "up" in the returned view; the agent's
        right maps to "right". Out-of-map pixels are set to ``fill``.
        """
        ar, ac = self.agent
        fwd_r, fwd_c = _FWD[self.heading]
        right_r, right_c = _RIGHT[self.heading]
        idx = np.arange(self.view)
        ii, jj = np.meshgrid(idx, idx, indexing="ij")
        f = self.radius - ii  # forward offset (top of image = +forward)
        s = jj - self.radius  # lateral offset (right = +)
        abs_r = ar + f * fwd_r + s * right_r
        abs_c = ac + f * fwd_c + s * right_c
        out = np.full((self.view, self.view), fill, dtype=np.float32)
        valid = (abs_r >= 0) & (abs_r < self.size) & (abs_c >= 0) & (abs_c < self.size)
        out[valid] = abs_field[abs_r[valid], abs_c[valid]]
        return out

    def _get_obs(self) -> np.ndarray:
        geo = self._egocentric(self._geo_abs, fill=_GEO_WALL)
        if self.obs_mode == "depth":
            return geo[None, :, :]
        haz = self._egocentric(self.layout.hazard, fill=0.0)
        return np.stack([geo, haz], axis=0)

    def _info(self, *, collision: bool, bad_region: bool) -> dict[str, Any]:
        return {
            "success": self._succeeded,
            "collision": collision,
            "bad_region": bad_region,
            "path_length": self.path_length,
            "collisions": self.collisions,
            "bad_region_time": self.bad_region_time,
            "ep_return": self.ep_return,
            "is_success": self._succeeded,
        }

    # ------------------------------------------------------------------ gym API
    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.layout = generate_map(self.np_random, self.map_config)
        self._geo_abs = self._build_geo_abs()
        self.agent = self.layout.start
        # Initial heading points toward the goal side (up) for both modes.
        self.heading = 0
        self.steps = 0
        self.collisions = 0
        self.bad_region_time = 0
        self.path_length = 0
        self.ep_return = 0.0
        self._succeeded = False
        self._prev_dist = self._goal_dist(self.agent)
        self.trajectory = [self.agent]
        return self._get_obs(), self._info(collision=False, bad_region=False)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = int(action)
        if not 0 <= action < 4:
            raise ValueError(f"action must be in [0, 4), got {action}")
        self.steps += 1
        collision = False
        reward = self.reward["step_penalty"]

        if action == FORWARD:
            dr, dc = _FWD[self.heading]
            nr, nc = self.agent[0] + dr, self.agent[1] + dc
            if 0 <= nr < self.size and 0 <= nc < self.size and self.layout.walls[nr, nc] == 0:
                self.agent = (nr, nc)
                self.path_length += 1
            else:
                collision = True
                self.collisions += 1
                reward += self.reward["collision_penalty"]
        elif action == TURN_LEFT:
            self.heading = (self.heading - 1) % 4
        elif action == TURN_RIGHT:
            self.heading = (self.heading + 1) % 4
        # STOP: no-op (still incurs step penalty).

        # Distance shaping (potential-based, identical for both obs modes).
        new_dist = self._goal_dist(self.agent)
        reward += self.reward["shaping_coef"] * (self._prev_dist - new_dist)
        self._prev_dist = new_dist

        # Hazard cost for the cell currently occupied.
        haz_here = float(self.layout.hazard[self.agent])
        bad_region = haz_here > 0.0
        if bad_region:
            self.bad_region_time += 1
            reward += self.reward["hazard_penalty"] * haz_here

        terminated = False
        if self.agent == self.layout.goal:
            reward += self.reward["goal_reward"]
            self._succeeded = True
            terminated = True

        truncated = self.steps >= self.max_steps and not terminated
        self.ep_return += reward
        self.trajectory.append(self.agent)
        return (
            self._get_obs(),
            float(reward),
            terminated,
            truncated,
            self._info(collision=collision, bad_region=bad_region),
        )

    # ------------------------------------------------------------------ render
    def render(self):  # noqa: D401 - simple renderer
        if self.layout is None:
            raise RuntimeError("Call reset() before render().")
        scale = self.render_scale
        h, w = self.size, self.size
        img = np.full((h, w, 3), 245, dtype=np.uint8)  # free = near-white
        # Hazard band -> red intensity.
        haz = self.layout.hazard
        red = (haz > 0)
        img[red] = np.stack(
            [
                np.full(red.sum(), 220),
                (245 - 180 * haz[red]).astype(np.uint8),
                (245 - 180 * haz[red]).astype(np.uint8),
            ],
            axis=1,
        )
        # Walls -> dark grey.
        img[self.layout.walls == 1] = (60, 60, 60)
        # Trajectory -> light blue.
        for (r, c) in self.trajectory[:-1]:
            img[r, c] = (150, 190, 255)
        # Goal -> green.
        img[self.layout.goal] = (40, 180, 70)
        # Agent -> blue.
        img[self.agent] = (30, 80, 220)
        big = np.kron(img, np.ones((scale, scale, 1), dtype=np.uint8))
        return big
