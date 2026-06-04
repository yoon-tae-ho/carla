"""Phase 4 route-env compatibility names."""

from __future__ import annotations

from .carla_online_env import CarlaSuspensionEnv


CarlaRouteSuspensionEnv = CarlaSuspensionEnv


__all__ = [
    "CarlaRouteSuspensionEnv",
    "CarlaSuspensionEnv",
]

