"""Phase 4 route-backend compatibility names.

The implementation lives in :mod:`suspension_control.rl.carla_backend`.
This module keeps the Phase 4 route-training API names importable without
duplicating backend behavior.
"""

from __future__ import annotations

from .carla_backend import (
    BackendResetResult,
    BackendStepResult,
    BaseCarlaRouteBackend,
    CarlaSuspensionBackend,
    FakeCarlaRouteBackend,
    FakeCarlaSuspensionBackend,
    LIVE_BACKEND_UNAVAILABLE_MESSAGE,
    LiveCarlaSuspensionBackend,
)


LiveRouteProcessBackend = LiveCarlaSuspensionBackend


__all__ = [
    "BackendResetResult",
    "BackendStepResult",
    "BaseCarlaRouteBackend",
    "CarlaSuspensionBackend",
    "FakeCarlaRouteBackend",
    "FakeCarlaSuspensionBackend",
    "LIVE_BACKEND_UNAVAILABLE_MESSAGE",
    "LiveCarlaSuspensionBackend",
    "LiveRouteProcessBackend",
]

