"""Observation building for residual-RL suspension control."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .normalizer import FixedScaleNormalizer


WHEEL_LABELS = ("fl", "fr", "rl", "rr")

STATE_FEATURES = (
    "speed",
    "local_vx",
    "local_vy",
    "vz",
    "local_ax",
    "local_ay",
    "az",
    "roll",
    "pitch",
    "yaw_rate",
    "roll_rate",
    "pitch_rate",
    "jerk_x",
    "jerk_y",
    "jerk_z",
    "body_activity",
)

BASELINE_FEATURES = (
    "baseline_damper_fl",
    "baseline_damper_fr",
    "baseline_damper_rl",
    "baseline_damper_rr",
    "baseline_mean_damper",
    "baseline_damper_spread",
    "skyhook_corner_vz_fl",
    "skyhook_corner_vz_fr",
    "skyhook_corner_vz_rl",
    "skyhook_corner_vz_rr",
)

MEMORY_FEATURES = (
    "previous_action_fl",
    "previous_action_fr",
    "previous_action_rl",
    "previous_action_rr",
    "previous_damper_fl",
    "previous_damper_fr",
    "previous_damper_rl",
    "previous_damper_rr",
)

PLANNING_FEATURES = (
    "planning_available",
    "preview_curvature_now",
    "preview_curvature_mean",
    "preview_curvature_max_abs",
    "preview_curvature_signed_peak",
    "preview_target_speed_now",
    "preview_target_speed_mean",
    "preview_target_speed_min",
    "preview_target_speed_max",
    "preview_longitudinal_acc_mean",
    "preview_longitudinal_acc_max_abs",
    "preview_lateral_acc_mean",
    "preview_lateral_acc_max_abs",
    "preview_steer_now",
    "preview_steer_mean",
    "preview_steer_max_abs",
    "preview_throttle_mean",
    "preview_brake_mean",
    "preview_brake_max",
    "time_to_hard_brake",
    "time_to_sharp_turn",
    "time_to_high_lateral_acc",
)


@dataclass(frozen=True)
class ObservationSpec:
    feature_names: Tuple[str, ...]
    feature_scales: Mapping[str, float]


class ObservationBuilder:
    """Build deterministic normalized observations from controller context."""

    def __init__(
        self,
        normalizer: Optional[FixedScaleNormalizer] = None,
        wheel_count: int = 4,
        observation_clip: float = 5.0,
    ):
        self.wheel_count = int(wheel_count)
        self.normalizer = normalizer or FixedScaleNormalizer(
            clip=observation_clip)
        self.feature_names = self._feature_names()

    def spec(self) -> ObservationSpec:
        return ObservationSpec(
            feature_names=self.feature_names,
            feature_scales={
                name: self.normalizer.scale_for_name(name)
                for name in self.feature_names
            })

    def build(
        self,
        context: Any,
        baseline_output: Any,
        previous_action: Optional[Sequence[float]],
        previous_damper_scales: Optional[Sequence[float]],
    ) -> Tuple[List[float], Dict[str, Any]]:
        raw = self.raw_features(
            context,
            baseline_output,
            previous_action,
            previous_damper_scales)
        values: List[float] = []
        invalid_names = []
        clip_count = 0
        for name in self.feature_names:
            raw_value = raw.get(name, 0.0)
            try:
                normalized = self.normalizer.normalize(name, raw_value)
                if not math.isfinite(normalized):
                    invalid_names.append(name)
                    normalized = 0.0
            except (TypeError, ValueError):
                invalid_names.append(name)
                normalized = 0.0
            clipped = self.normalizer.clip_value(normalized)
            if clipped != normalized:
                clip_count += 1
            values.append(clipped)

        diagnostics: Dict[str, Any] = {
            "observation_valid": int(not invalid_names),
            "observation_size": len(values),
            "observation_clip_count": clip_count,
            "observation_invalid_features": ";".join(invalid_names),
            "planning_available": raw.get("planning_available", 0.0),
            "observation_feature_names": ",".join(self.feature_names),
        }
        for name in PLANNING_FEATURES:
            diagnostics[name] = raw.get(name, 0.0)
        diagnostics.update(self.normalizer.diagnostics())
        return values, diagnostics

    def raw_features(
        self,
        context: Any,
        baseline_output: Any,
        previous_action: Optional[Sequence[float]],
        previous_damper_scales: Optional[Sequence[float]],
    ) -> Dict[str, float]:
        state = context.state
        previous_state = getattr(context, "previous_state", None)
        dt = _positive_dt(context, state)
        raw: Dict[str, float] = {
            "speed": _float(getattr(state, "speed", 0.0)),
            "local_vx": _float(getattr(state, "local_vx", 0.0)),
            "local_vy": _float(getattr(state, "local_vy", 0.0)),
            "vz": _float(getattr(state, "vz", 0.0)),
            "local_ax": _float(getattr(state, "local_ax", 0.0)),
            "local_ay": _float(getattr(state, "local_ay", 0.0)),
            "az": _float(getattr(state, "az", 0.0)),
            "roll": _float(getattr(state, "roll", 0.0)),
            "pitch": _float(getattr(state, "pitch", 0.0)),
            "yaw_rate": _float(getattr(state, "yaw_rate", 0.0)),
            "roll_rate": _float(getattr(state, "roll_rate", 0.0)),
            "pitch_rate": _float(getattr(state, "pitch_rate", 0.0)),
            "body_activity": _body_activity(state),
        }
        raw.update(_jerk_features(state, previous_state, dt))
        raw.update(self._baseline_features(baseline_output))
        raw.update(self._memory_features(previous_action, previous_damper_scales))
        raw.update(self._planning_features(getattr(context, "planning", None)))
        return raw

    def _feature_names(self) -> Tuple[str, ...]:
        return tuple(
            list(STATE_FEATURES) +
            list(BASELINE_FEATURES) +
            list(MEMORY_FEATURES) +
            list(PLANNING_FEATURES))

    def _baseline_features(self, baseline_output: Any) -> Dict[str, float]:
        raw: Dict[str, float] = {}
        dampers = _command_dampers(baseline_output.command, self.wheel_count)
        for label, value in zip(WHEEL_LABELS, dampers):
            raw["baseline_damper_%s" % label] = value
        raw["baseline_mean_damper"] = _mean(dampers)
        raw["baseline_damper_spread"] = (
            max(dampers) - min(dampers) if dampers else 0.0)
        diagnostics = dict(getattr(baseline_output, "diagnostics", {}) or {})
        for label in WHEEL_LABELS:
            raw["skyhook_corner_vz_%s" % label] = _float(
                diagnostics.get("skyhook_corner_vz_%s" % label, 0.0))
        return raw

    def _memory_features(
        self,
        previous_action: Optional[Sequence[float]],
        previous_damper_scales: Optional[Sequence[float]],
    ) -> Dict[str, float]:
        raw: Dict[str, float] = {}
        actions = _match_length(previous_action, self.wheel_count, 0.0)
        dampers = _match_length(previous_damper_scales, self.wheel_count, 1.0)
        for label, value in zip(WHEEL_LABELS, actions):
            raw["previous_action_%s" % label] = value
        for label, value in zip(WHEEL_LABELS, dampers):
            raw["previous_damper_%s" % label] = value
        return raw

    def _planning_features(self, planning: Any) -> Dict[str, float]:
        values = {name: 0.0 for name in PLANNING_FEATURES}
        if planning is None:
            return values
        if hasattr(planning, "preview_summary"):
            try:
                summary = planning.preview_summary()
                for name in PLANNING_FEATURES:
                    if name in summary:
                        values[name] = _float(summary[name])
                return values
            except Exception:
                return values

        points = tuple(getattr(planning, "points", ()) or ())
        if not points:
            return values
        curvatures = [_float(getattr(point, "curvature", 0.0)) for point in points]
        speeds = [_float(getattr(point, "speed", 0.0)) for point in points]
        values["planning_available"] = 1.0
        values["preview_curvature_now"] = curvatures[0] if curvatures else 0.0
        values["preview_curvature_mean"] = _mean(curvatures)
        values["preview_curvature_max_abs"] = _max_abs(curvatures)
        values["preview_curvature_signed_peak"] = _signed_peak(curvatures)
        values["preview_target_speed_now"] = speeds[0] if speeds else 0.0
        values["preview_target_speed_mean"] = _mean(speeds)
        values["preview_target_speed_min"] = min(speeds) if speeds else 0.0
        values["preview_target_speed_max"] = max(speeds) if speeds else 0.0
        if len(speeds) >= 2:
            longitudinal = [
                speeds[index + 1] - speeds[index]
                for index in range(len(speeds) - 1)
            ]
            values["preview_longitudinal_acc_mean"] = _mean(longitudinal)
            values["preview_longitudinal_acc_max_abs"] = _max_abs(longitudinal)
        lateral_acc = [
            speed * speed * curvature
            for speed, curvature in zip(speeds, curvatures)
        ]
        values["preview_lateral_acc_mean"] = _mean(lateral_acc)
        values["preview_lateral_acc_max_abs"] = _max_abs(lateral_acc)
        return values


def _positive_dt(context: Any, state: Any) -> float:
    dt = _float(getattr(context, "dt", 0.0) or getattr(state, "dt", 0.0) or 0.0)
    return max(dt, 1.0e-9)


def _jerk_features(state: Any, previous_state: Any, dt: float) -> Dict[str, float]:
    if previous_state is None:
        return {"jerk_x": 0.0, "jerk_y": 0.0, "jerk_z": 0.0}
    return {
        "jerk_x": (
            _float(getattr(state, "local_ax", 0.0)) -
            _float(getattr(previous_state, "local_ax", 0.0))) / dt,
        "jerk_y": (
            _float(getattr(state, "local_ay", 0.0)) -
            _float(getattr(previous_state, "local_ay", 0.0))) / dt,
        "jerk_z": (
            _float(getattr(state, "az", 0.0)) -
            _float(getattr(previous_state, "az", 0.0))) / dt,
    }


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _match_length(
    values: Optional[Sequence[float]],
    count: int,
    fill: float,
) -> List[float]:
    result = list(values or [])
    result = [_float(value) for value in result[:count]]
    while len(result) < count:
        result.append(fill)
    return result


def _command_dampers(command: Any, count: int) -> List[float]:
    wheels = list(getattr(command, "wheels", ()) or [])
    values = [_float(getattr(wheel, "damper_scale", 1.0)) for wheel in wheels[:count]]
    while len(values) < count:
        values.append(1.0)
    return values


def _body_activity(state: Any) -> float:
    components = (
        abs(_float(getattr(state, "roll", 0.0))) / 10.0,
        abs(_float(getattr(state, "pitch", 0.0))) / 10.0,
        abs(_float(getattr(state, "roll_rate", 0.0))) / 60.0,
        abs(_float(getattr(state, "pitch_rate", 0.0))) / 60.0,
        abs(_float(getattr(state, "local_ay", 0.0))) / 10.0,
        abs(_float(getattr(state, "az", 0.0))) / 10.0,
    )
    return sum(components) / float(len(components))


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / float(len(values)) if values else 0.0


def _max_abs(values: Iterable[float]) -> float:
    values = list(values)
    return max(abs(value) for value in values) if values else 0.0


def _signed_peak(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return max(values, key=lambda value: abs(value))
