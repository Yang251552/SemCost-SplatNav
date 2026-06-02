"""Shared experiment configuration for fair depth vs depth_semantic comparison.

Everything that must be identical across the two observation modes lives here so
that ``train.py`` and ``evaluate.py`` cannot drift: reward, dynamics, map
distribution, horizon, PPO hyperparameters, training seed and the fixed
evaluation seed list. The ONLY thing that varies between runs is ``obs_mode``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from semcost_nav.envs.semcost_nav_env import SemCostNavEnv
from semcost_nav.maps.map_gen import MapConfig
from semcost_nav.policies.extractors import SmallNavCNN

# --- Fixed task definition (identical for both obs modes) --------------------
MAX_STEPS = 90
MAP_CONFIG = MapConfig()  # default 11x11 arena, band rows (4,5,6), gap width 3
# Reward overrides shared by both modes. shaping_coef is kept small so the
# potential-based distance shaping does not over-penalise the lateral detour to
# the safe gap (see EXPERIMENT_LOG iteration 1 diagnosis).
REWARD_CONFIG = {"shaping_coef": 0.02}

# --- Training configuration --------------------------------------------------
DEFAULT_TIMESTEPS = 500_000
N_ENVS = 8
TRAIN_SEED = 0

PPO_KWARGS = dict(
    n_steps=512,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.02,
    learning_rate=3.0e-4,
    vf_coef=0.5,
    max_grad_norm=0.5,
)

POLICY_KWARGS = dict(
    features_extractor_class=SmallNavCNN,
    features_extractor_kwargs=dict(features_dim=128),
    net_arch=dict(pi=[128], vf=[128]),
)

# --- Evaluation configuration ------------------------------------------------
# A FIXED, shared seed list. Both policies are evaluated on exactly these maps,
# so any difference in bad_region_time/collision/etc. comes from the policy, not
# from different scene draws.
N_EVAL_EPISODES = 200
EVAL_SEED_BASE = 100_000


def eval_seeds(n: int = N_EVAL_EPISODES, base: int = EVAL_SEED_BASE) -> list[int]:
    """Return the fixed evaluation seed list (identical for both obs modes)."""
    return [base + i for i in range(n)]


def make_single_env(obs_mode: str) -> SemCostNavEnv:
    """Construct one (unwrapped) environment with the shared task config."""
    return SemCostNavEnv(
        obs_mode=obs_mode,
        max_steps=MAX_STEPS,
        map_config=MAP_CONFIG,
        reward_config=REWARD_CONFIG,
        render_mode="rgb_array",
    )


def make_training_env(obs_mode: str, n_envs: int = N_ENVS, seed: int = TRAIN_SEED):
    """Build a Monitor-wrapped vectorized training environment."""
    return make_vec_env(
        lambda: Monitor(make_single_env(obs_mode)),
        n_envs=n_envs,
        seed=seed,
    )


def build_ppo(env, seed: int = TRAIN_SEED, device: str = "cpu") -> PPO:
    """Construct a PPO model with the shared hyperparameters and policy."""
    return PPO(
        "CnnPolicy",
        env,
        seed=seed,
        device=device,
        verbose=0,
        policy_kwargs=POLICY_KWARGS,
        **PPO_KWARGS,
    )


def config_snapshot() -> dict:
    """A JSON-serializable snapshot of the fairness-critical configuration."""
    return {
        "max_steps": MAX_STEPS,
        "reward_config": dict(REWARD_CONFIG),
        "map_config": asdict(MAP_CONFIG),
        "train_timesteps_default": DEFAULT_TIMESTEPS,
        "n_envs": N_ENVS,
        "train_seed": TRAIN_SEED,
        "ppo_kwargs": dict(PPO_KWARGS),
        "n_eval_episodes": N_EVAL_EPISODES,
        "eval_seed_base": EVAL_SEED_BASE,
    }


# Keys from a train metadata file that MUST match between the two obs modes for
# the comparison to be fair. (obs_mode and observation channel count are the
# only intended differences and are deliberately excluded.)
FAIRNESS_KEYS = ("timesteps", "seed")
FAIRNESS_CONFIG_KEYS = (
    "max_steps",
    "reward_config",
    "map_config",
    "train_seed",
    "ppo_kwargs",
    "n_eval_episodes",
    "eval_seed_base",
)


def fairness_signature(train_meta: dict) -> dict:
    """Extract the fairness-critical fields from a train_<obs>.json dict."""
    sig = {k: train_meta.get(k) for k in FAIRNESS_KEYS}
    cfg = train_meta.get("config", {})
    sig["config"] = {k: cfg.get(k) for k in FAIRNESS_CONFIG_KEYS}
    return sig


def assert_fair(meta_a: dict, meta_b: dict) -> None:
    """Raise AssertionError if two runs differ on any fairness-critical field."""
    sa, sb = fairness_signature(meta_a), fairness_signature(meta_b)
    if sa != sb:
        diffs = [k for k in sa if sa[k] != sb.get(k)]
        raise AssertionError(
            "unfair comparison: runs differ on fairness-critical fields "
            f"{diffs}.\n  A={sa}\n  B={sb}"
        )


def dependency_versions() -> dict:
    """Record key library versions for reproducibility provenance."""
    import platform

    import gymnasium
    import numpy
    import stable_baselines3
    import torch

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "gymnasium": gymnasium.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "torch": torch.__version__,
    }
