"""Custom CNN feature extractor for the small egocentric observation.

Stable-Baselines3's default ``NatureCNN`` assumes large (>=36px) images and
fails on our compact egocentric view. ``SmallNavCNN`` is a light convolutional
stack that works for small multi-channel inputs and is shared identically by
the depth-only and depth+semantic policies (only the input channel count
differs).
"""

from __future__ import annotations

import gymnasium as gym
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class SmallNavCNN(BaseFeaturesExtractor):
    """A compact CNN for (C, S, S) observations with small S."""

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
