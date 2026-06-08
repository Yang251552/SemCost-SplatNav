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

import pytest  # noqa: E402


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


def _load_dino_cost_mapper():
    pytest.importorskip("transformers", reason="transformers not installed")
    from semcost_nav.semantic.dino_cost import DinoCostMapper

    return DinoCostMapper


def test_dino_cost_mapper_signature_has_no_target_inputs() -> None:
    DinoCostMapper = _load_dino_cost_mapper()
    banned = {"goal", "target", "agent_pos", "xyz", "pos_command"}
    sig_names = set(inspect.signature(DinoCostMapper.__init__).parameters)
    assert not (sig_names & banned)
    assert sig_names <= {"self", "pca_path", "bad_proto_path", "margin", "model_name"}


def test_dino_cost_mapper_call_signature_has_no_target_inputs() -> None:
    DinoCostMapper = _load_dino_cost_mapper()
    banned = {"goal", "target", "agent_pos", "xyz", "pos_command"}
    sig_names = set(inspect.signature(DinoCostMapper.__call__).parameters)
    assert not (sig_names & banned)
    assert sig_names <= {"self", "rgb"}


def test_dino_cost_mapper_call_closure_has_no_target_refs() -> None:
    DinoCostMapper = _load_dino_cost_mapper()
    banned = {"goal", "target", "agent_pos", "xyz", "pos_command"}
    referenced = set(DinoCostMapper.__call__.__code__.co_names) | set(
        DinoCostMapper.__call__.__code__.co_varnames
    )
    assert not (referenced & banned)
