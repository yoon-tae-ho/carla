"""Contracts shared by Phase-A residual MPC follow-up work."""

from __future__ import annotations

import math
from typing import Sequence, Tuple


WHEEL_ORDER: Tuple[str, str, str, str] = (
    "front_left",
    "front_right",
    "rear_left",
    "rear_right",
)

MODAL_BASIS_NAMES: Tuple[str, str, str, str] = (
    "mean",
    "roll_front",
    "roll_rear",
    "pitch",
)

MODAL_TO_WHEEL_MATRIX: Tuple[Tuple[float, float, float, float], ...] = (
    (1.0, -1.0, 0.0, 1.0),
    (1.0, 1.0, 0.0, 1.0),
    (1.0, 0.0, -1.0, -1.0),
    (1.0, 0.0, 1.0, -1.0),
)


def modal_to_wheel_residual(modal: Sequence[float]) -> Tuple[float, ...]:
    """Map [mean, roll_front, roll_rear, pitch] to FL/FR/RL/RR residuals."""

    values = _finite_tuple(modal, len(MODAL_BASIS_NAMES), "modal")
    return tuple(
        sum(row[index] * values[index] for index in range(len(values)))
        for row in MODAL_TO_WHEEL_MATRIX)


def wheel_residual_to_modal(wheel_residual: Sequence[float]) -> Tuple[float, ...]:
    """Return the least-squares modal coordinates for FL/FR/RL/RR residuals."""

    fl, fr, rl, rr = _finite_tuple(
        wheel_residual,
        len(WHEEL_ORDER),
        "wheel_residual")
    front_mean = 0.5 * (fl + fr)
    rear_mean = 0.5 * (rl + rr)
    return (
        0.5 * (front_mean + rear_mean),
        0.5 * (fr - fl),
        0.5 * (rr - rl),
        0.5 * (front_mean - rear_mean),
    )


def _finite_tuple(
    values: Sequence[float],
    expected_len: int,
    name: str,
) -> Tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != expected_len:
        raise ValueError(
            "%s must have length %d, got %d" %
            (name, expected_len, len(result)))
    if not all(math.isfinite(value) for value in result):
        raise ValueError("%s contains nonfinite value" % name)
    return result
