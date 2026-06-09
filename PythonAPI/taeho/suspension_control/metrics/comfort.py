"""Comfort metrics for suspension-control experiments."""

from __future__ import annotations

import math
from typing import Optional, Sequence

from .stability import _get, _values, mean, peak_abs, rms


def _safe_float(value) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _time_delta(left, right, default_dt: float) -> Optional[float]:
    row_dt = _safe_float(_get(right, "dt", None))
    if row_dt is not None:
        return row_dt if row_dt > 0.0 else None

    elapsed_left = _safe_float(_get(left, "elapsed_seconds", None))
    elapsed_right = _safe_float(_get(right, "elapsed_seconds", None))
    if elapsed_left is not None and elapsed_right is not None:
        dt = elapsed_right - elapsed_left
        return dt if dt > 0.0 else None

    return default_dt if default_dt > 0.0 else None


def jerk_series(profile: Sequence, axis: str = "az", default_dt: float = 0.05):
    if len(profile) < 2:
        return []
    jerks = []
    for left, right in zip(profile[:-1], profile[1:]):
        dt = _time_delta(left, right, default_dt)
        left_value = _safe_float(_get(left, axis))
        right_value = _safe_float(_get(right, axis))
        if dt is None or left_value is None or right_value is None:
            continue
        jerks.append((right_value - left_value) / dt)
    return jerks


def comfort_metrics(profile: Sequence, default_dt: float = 0.05):
    ax = _values(profile, "local_ax")
    ay = _values(profile, "local_ay")
    az = _values(profile, "az")
    roll_rate = _values(profile, "roll_rate")
    pitch_rate = _values(profile, "pitch_rate")

    vertical_jerk = jerk_series(profile, axis="az", default_dt=default_dt)
    lateral_jerk = jerk_series(profile, axis="local_ay", default_dt=default_dt)
    longitudinal_jerk = jerk_series(profile, axis="local_ax", default_dt=default_dt)

    # A light-weight scalar for ranking experiments. Keep raw components in CSV
    # and tune this score only after collecting enough vehicle-specific data.
    comfort_score = (
        rms(az) +
        0.50 * rms(ay) +
        0.25 * rms(ax) +
        0.05 * rms(vertical_jerk) +
        0.03 * rms(lateral_jerk) +
        0.03 * rms(longitudinal_jerk) +
        0.02 * rms(roll_rate) +
        0.02 * rms(pitch_rate))

    return {
        "rms_vertical_acc": rms(az),
        "rms_lateral_acc": rms(ay),
        "rms_longitudinal_acc": rms(ax),
        "peak_abs_vertical_acc": peak_abs(az),
        "peak_abs_lateral_acc": peak_abs(ay),
        "peak_abs_longitudinal_acc": peak_abs(ax),
        "rms_vertical_jerk": rms(vertical_jerk),
        "rms_lateral_jerk": rms(lateral_jerk),
        "rms_longitudinal_jerk": rms(longitudinal_jerk),
        "peak_abs_vertical_jerk": peak_abs(vertical_jerk),
        "mean_abs_roll_rate": mean([abs(value) for value in roll_rate]),
        "mean_abs_pitch_rate": mean([abs(value) for value in pitch_rate]),
        "comfort_score": comfort_score,
    }
