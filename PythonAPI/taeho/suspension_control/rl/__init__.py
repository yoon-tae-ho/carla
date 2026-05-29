"""RL helpers for residual suspension control.

The modules in this package intentionally keep optional ML dependencies behind
small adapters so the CARLA route suite can import without torch, gymnasium, or
stable-baselines3 installed.
"""

from .carla_env import SuspensionCarlaEnv
from .normalizer import FixedScaleNormalizer
from .observations import ObservationBuilder, ObservationSpec
from .policy import PolicyAdapter
from .reward import RewardTransition, SuspensionReward, SuspensionRewardConfig
from .rollout_logger import RolloutLogger

__all__ = [
    "FixedScaleNormalizer",
    "ObservationBuilder",
    "ObservationSpec",
    "PolicyAdapter",
    "RewardTransition",
    "RolloutLogger",
    "SuspensionCarlaEnv",
    "SuspensionReward",
    "SuspensionRewardConfig",
]
