from __future__ import annotations

import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from semcost_nav.gaussgym_ext.observation_tokens import (  # noqa: E402
    semantic_cost_observation,
)


def test_semantic_cost_observation_has_no_target_or_pose_inputs() -> None:
    banned = {"goal", "target", "agent_pos", "xyz", "pos_command"}
    sig_names = set(inspect.signature(semantic_cost_observation).parameters)
    closure = inspect.getclosurevars(semantic_cost_observation)
    referenced = (
        set(semantic_cost_observation.__code__.co_names)
        | set(semantic_cost_observation.__code__.co_varnames)
        | set(closure.nonlocals)
        | set(closure.globals)
    )
    assert not (sig_names & banned)
    assert not (referenced & banned)
    assert sig_names <= {"hazard_mask", "rgb_latent"}
