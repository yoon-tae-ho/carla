"""Stability metrics for suspension-control experiments."""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence


def _get(row, name, default=0.0):
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


def _values(profile: Iterable, name: str) -> List[float]:
    return [float(_get(row, name, 0.0)) for row in profile]


def rms(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(value * value for value in values) / float(len(values)))


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def steady_slice(profile: Sequence, steady_fraction: float = 0.30):
    if not profile:
        return []
    start = int(len(profile) * (1.0 - steady_fraction))
    return profile[max(0, start):]


def peak_abs(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return max(abs(value) for value in values)


def stability_metrics(profile: Sequence, steady_fraction: float = 0.30):
    steady = steady_slice(profile, steady_fraction)
    roll = _values(profile, "roll")
    pitch = _values(profile, "pitch")
    yaw_rate = _values(profile, "yaw_rate")
    lateral_acc = _values(profile, "local_ay")

    return {
        "peak_abs_roll": peak_abs(roll),
        "peak_abs_pitch": peak_abs(pitch),
        "peak_abs_yaw_rate": peak_abs(yaw_rate),
        "peak_abs_lateral_acc": peak_abs(lateral_acc),
        "rms_roll": rms(roll),
        "rms_pitch": rms(pitch),
        "rms_yaw_rate": rms(yaw_rate),
        "rms_lateral_acc": rms(lateral_acc),
        "mean_abs_roll_tail": mean([abs(value) for value in _values(steady, "roll")]),
        "mean_abs_pitch_tail": mean([abs(value) for value in _values(steady, "pitch")]),
        "mean_abs_yaw_rate_tail": mean([
            abs(value) for value in _values(steady, "yaw_rate")
        ]),
        "mean_abs_lateral_acc_tail": mean([
            abs(value) for value in _values(steady, "local_ay")
        ]),
    }


def distance_xyz(left, right) -> float:
    dx = float(_get(left, "x")) - float(_get(right, "x"))
    dy = float(_get(left, "y")) - float(_get(right, "y"))
    dz = float(_get(left, "z")) - float(_get(right, "z"))
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def compare_profiles(baseline: Sequence, candidate: Sequence):
    if len(baseline) != len(candidate):
        raise ValueError(
            "profile length mismatch: %d vs %d" %
            (len(baseline), len(candidate)))

    position_diffs = []
    speed_diffs = []
    roll_diffs = []
    pitch_diffs = []
    yaw_rate_diffs = []
    lateral_acc_diffs = []

    for base, row in zip(baseline, candidate):
        position_diffs.append(distance_xyz(base, row))
        speed_diffs.append(abs(float(_get(base, "speed")) - float(_get(row, "speed"))))
        roll_diffs.append(abs(float(_get(base, "roll")) - float(_get(row, "roll"))))
        pitch_diffs.append(abs(float(_get(base, "pitch")) - float(_get(row, "pitch"))))
        yaw_rate_diffs.append(abs(
            float(_get(base, "yaw_rate")) - float(_get(row, "yaw_rate"))))
        lateral_acc_diffs.append(abs(
            float(_get(base, "local_ay")) - float(_get(row, "local_ay"))))

    return {
        "max_position_diff": max(position_diffs) if position_diffs else 0.0,
        "rms_position_diff": rms(position_diffs),
        "max_speed_diff": max(speed_diffs) if speed_diffs else 0.0,
        "rms_speed_diff": rms(speed_diffs),
        "max_roll_diff": max(roll_diffs) if roll_diffs else 0.0,
        "rms_roll_diff": rms(roll_diffs),
        "max_pitch_diff": max(pitch_diffs) if pitch_diffs else 0.0,
        "rms_pitch_diff": rms(pitch_diffs),
        "max_yaw_rate_diff": max(yaw_rate_diffs) if yaw_rate_diffs else 0.0,
        "rms_yaw_rate_diff": rms(yaw_rate_diffs),
        "max_lateral_acc_diff": (
            max(lateral_acc_diffs) if lateral_acc_diffs else 0.0),
        "rms_lateral_acc_diff": rms(lateral_acc_diffs),
    }
