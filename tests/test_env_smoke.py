"""Smoke tests for SemCostNavEnv: API, shapes, leakage and orientation.

Run directly (``python tests/test_env_smoke.py``) or via pytest.
"""

from __future__ import annotations

import numpy as np

from semcost_nav.envs.semcost_nav_env import SemCostNavEnv


def test_obs_shapes() -> None:
    depth = SemCostNavEnv(obs_mode="depth")
    obs, info = depth.reset(seed=0)
    assert obs.shape == depth.observation_space.shape
    assert obs.shape[0] == 1
    ds = SemCostNavEnv(obs_mode="depth_semantic")
    obs2, _ = ds.reset(seed=0)
    assert obs2.shape[0] == 2
    assert obs2.shape[1:] == obs.shape[1:]
    # Geometry channel must be identical across the two modes (fairness).
    assert np.allclose(obs[0], obs2[0]), "geometry channel differs between modes"


def test_reset_seed_determinism() -> None:
    e1 = SemCostNavEnv(obs_mode="depth_semantic")
    e2 = SemCostNavEnv(obs_mode="depth_semantic")
    o1, _ = e1.reset(seed=123)
    o2, _ = e2.reset(seed=123)
    assert np.allclose(o1, o2)
    assert e1.layout.gap_cols == e2.layout.gap_cols
    assert e1.layout.start == e2.layout.start


def test_random_rollout_runs() -> None:
    env = SemCostNavEnv(obs_mode="depth_semantic")
    obs, info = env.reset(seed=7)
    rng = np.random.default_rng(7)
    steps = 0
    done = False
    while not done and steps < env.max_steps + 5:
        a = int(rng.integers(0, 4))
        obs, r, term, trunc, info = env.step(a)
        assert np.isfinite(r)
        assert obs.shape == env.observation_space.shape
        for key in ("success", "collision", "bad_region", "path_length"):
            assert key in info
        done = term or trunc
        steps += 1
    assert done, "episode did not terminate/truncate within horizon"


def test_semantic_does_not_leak_goal() -> None:
    """The semantic-cost channel must not encode the goal location.

    Concretely: hazard values are determined solely by the band/gap, never by
    the goal cell, and the goal cell carries zero hazard.
    """
    env = SemCostNavEnv(obs_mode="depth_semantic")
    for seed in range(20):
        env.reset(seed=seed)
        layout = env.layout
        assert layout.hazard[layout.goal] == 0.0
        # Hazard is non-zero only on band rows (never on the goal row unless the
        # goal row is a band row, which the default config avoids).
        nonzero_rows = set(np.argwhere(layout.hazard > 0)[:, 0].tolist())
        assert nonzero_rows.issubset(set(layout.band_rows))
        assert layout.goal[0] not in layout.band_rows


def test_forward_faces_up_in_view() -> None:
    """When facing the goal, the goal marker should sit above the centre.

    This verifies the egocentric rotation puts "forward" at the top of the view.
    """
    env = SemCostNavEnv(obs_mode="depth")
    env.reset(seed=1)
    # Place agent directly below the goal, facing up (heading 0).
    gr, gc = env.layout.goal
    env.agent = (gr + 2, gc)
    env.heading = 0
    env._prev_dist = env._goal_dist(env.agent)
    obs = env._get_obs()[0]
    center = env.radius
    # Goal marker (~0.5) must appear above centre along the centre column.
    col = obs[:, center]
    goal_rows = np.argwhere(np.isclose(col, 0.5)).ravel()
    assert goal_rows.size >= 1, "goal not visible ahead"
    assert goal_rows.min() < center, "goal not rendered above centre when faced"


def test_goal_row_in_band_is_rejected() -> None:
    """A config placing the goal on a band row must be rejected (leakage guard)."""
    import numpy as np  # local import to keep top clean
    from semcost_nav.maps.map_gen import MapConfig, generate_map

    rng = np.random.default_rng(0)
    bad = MapConfig(band_rows=(1,))  # goal defaults to interior row 1
    try:
        generate_map(rng, bad)
    except ValueError:
        return
    raise AssertionError("expected ValueError for goal row inside hazard band")


def test_illegal_action_rejected() -> None:
    env = SemCostNavEnv(obs_mode="depth")
    env.reset(seed=0)
    try:
        env.step(4)
    except ValueError:
        return
    raise AssertionError("expected ValueError for out-of-range action")


def test_collision_on_wall() -> None:
    env = SemCostNavEnv(obs_mode="depth")
    env.reset(seed=2)
    # Put agent against the top wall facing up, then step forward.
    env.agent = (1, env.layout.goal[1] + 0)  # row 1 interior; row 0 is wall
    # Move to a non-goal interior cell adjacent to a wall to test collision.
    env.agent = (1, 1)
    env.heading = 3  # facing left toward the wall at col 0
    env._prev_dist = env._goal_dist(env.agent)
    _, r, term, trunc, info = env.step(0)  # forward into wall
    assert info["collision"] is True
    assert env.agent == (1, 1)


if __name__ == "__main__":
    test_obs_shapes()
    test_reset_seed_determinism()
    test_random_rollout_runs()
    test_semantic_does_not_leak_goal()
    test_forward_faces_up_in_view()
    test_goal_row_in_band_is_rejected()
    test_illegal_action_rejected()
    test_collision_on_wall()
    print("All smoke tests passed.")
