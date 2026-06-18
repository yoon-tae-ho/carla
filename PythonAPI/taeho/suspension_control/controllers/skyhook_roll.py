"""Skyhook-style damping with damping-only roll-control side distribution."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .base import (
    ControllerContext,
    ControllerOutput,
    SuspensionCommand,
    SuspensionController,
    VehicleState,
    WheelScale,
    clamp,
)


_EPS = 1.0e-9
_FROZEN_SPRING_SCALE = 1.0
_CORNER_LABELS = ("fl", "fr", "rl", "rr")


@dataclass(frozen=True)
class SkyhookRollConfig:
    """Configuration for the skyhook + roll damping-only baseline.

    CARLA does not expose deflection/relative damper velocity; this is a
    skyhook-style damping baseline with roll-aware per-side damper scaling,
    not a physical active force controller. Spring scale is intentionally
    frozen at identity scale so the experiment isolates damping.
    """

    default_dt: float = 0.05
    wheel_count: int = 4

    base_damper_scale: float = 1.0
    min_damper_scale: float = 0.80
    max_damper_scale: float = 1.40
    max_damper_delta_per_step: float = 0.04
    corner_velocity_deadband: float = 0.02
    corner_velocity_scale: float = 1.0
    skyhook_gain: float = 0.25

    # Compatibility fields only; spring output is frozen to identity scale.
    base_spring_scale: float = 1.0
    min_spring_scale: float = 0.95
    max_spring_scale: float = 1.15
    max_spring_delta_per_step: float = 0.02
    enable_roll_spring_control: bool = False

    half_track_m: float = 0.85
    half_wheelbase_m: float = 1.45

    g: float = 9.81
    roll_rate_deadband_rad: float = 0.015
    roll_rate_scale_radps: float = 0.35
    roll_angle_deadband_rad: float = 0.0035
    roll_angle_scale_rad: float = 0.060
    lateral_accel_start_g: float = 0.15
    lateral_accel_full_g: float = 0.60
    lateral_accel_side_deadband: float = 0.30
    roll_rate_weight: float = 0.60
    roll_angle_weight: float = 0.25
    lateral_accel_weight: float = 0.35
    roll_angle_to_spring_weight: float = 0.0
    max_roll_activity: float = 1.50

    roll_damper_gain: float = 0.12
    max_roll_damper_add: float = 0.25
    roll_spring_gain: float = 0.0
    max_roll_spring_add: float = 0.0

    nominal_front_roll_distribution: float = 0.60
    min_front_roll_distribution: float = 0.35
    max_front_roll_distribution: float = 0.75
    max_front_distribution_delta_per_step: float = 0.04
    outer_side_sign_from_ay: float = -1.0
    outer_side_sign_from_roll: float = 0.0
    outer_side_bias: float = 0.30
    inner_side_relief: float = 0.10

    enable_yaw_distribution: bool = False
    yaw_activation_start_g: float = 0.40
    yaw_activation_full_g: float = 0.60
    max_road_wheel_angle_rad: float = 0.60
    yaw_ref_min_speed: float = 5.0
    yaw_ref_limit_lat_g: float = 0.85
    yaw_front_distribution_kp: float = 0.08
    yaw_front_distribution_ki: float = 0.00
    yaw_error_integral_limit: float = 2.0
    yaw_integral_decay: float = 0.98
    small_yaw_ref: float = 0.03

    debug: bool = False

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SkyhookRollConfig":
        allowed = {item.name for item in fields(cls)}
        incoming = dict(values or {})
        if "half_track" in incoming and "half_track_m" not in incoming:
            incoming["half_track_m"] = incoming["half_track"]
        if "half_wheelbase" in incoming and "half_wheelbase_m" not in incoming:
            incoming["half_wheelbase_m"] = incoming["half_wheelbase"]
        kwargs = {
            key: value
            for key, value in incoming.items()
            if key in allowed
        }
        return cls(**kwargs)


class SkyhookRollController(SuspensionController):
    """Skyhook body damping plus roll-aware damper side distribution."""

    name = "skyhook_roll"

    def __init__(self, config: Optional[SkyhookRollConfig] = None):
        self.config = config or SkyhookRollConfig()
        self.previous_damper_scales = self._initial_damper_scales()
        self.previous_spring_scales = self._initial_spring_scales()
        self.previous_front_share = self._nominal_front_share()
        self.yaw_error_integral = 0.0

    def reset(self, native_suspension: Any = None) -> None:
        self.previous_damper_scales = self._initial_damper_scales()
        self.previous_spring_scales = self._initial_spring_scales()
        self.previous_front_share = self._nominal_front_share()
        self.yaw_error_integral = 0.0

    def compute(self, context: ControllerContext) -> ControllerOutput:
        cfg = self.config
        if int(cfg.wheel_count) != 4:
            return self._fallback_output("unsupported_wheel_count", context)

        state = context.state
        dt = self._dt(context)
        roll_rad = math.radians(_safe_float(state.roll))
        roll_rate_rad = math.radians(_safe_float(state.roll_rate))
        pitch_rate_rad = math.radians(_safe_float(state.pitch_rate))
        yaw_rate_rad = math.radians(_safe_float(state.yaw_rate))
        local_ay = self._local_ay(state)
        speed = self._speed(state)
        steer = self._steer(state)

        corners = self._corner_coordinates()
        corner_velocities = tuple(
            _safe_float(state.vz) + roll_rate_rad * y - pitch_rate_rad * x
            for x, y in corners)
        skyhook_activities = tuple(
            self._corner_activity(corner_vz)
            for corner_vz in corner_velocities)
        skyhook_dampers = tuple(
            cfg.base_damper_scale + cfg.skyhook_gain * activity
            for activity in skyhook_activities)

        lat_g = abs(local_ay) / max(abs(_safe_float(cfg.g, 9.81)), _EPS)
        roll_rate_activity = self._activity(
            abs(roll_rate_rad),
            cfg.roll_rate_deadband_rad,
            cfg.roll_rate_scale_radps)
        roll_angle_activity = self._activity(
            abs(roll_rad),
            cfg.roll_angle_deadband_rad,
            cfg.roll_angle_scale_rad)
        lat_activity = _ramp(
            lat_g,
            _safe_float(cfg.lateral_accel_start_g),
            _safe_float(cfg.lateral_accel_full_g))
        roll_damping_activity = clamp(
            _safe_float(cfg.roll_rate_weight) * roll_rate_activity +
            _safe_float(cfg.roll_angle_weight) * roll_angle_activity +
            _safe_float(cfg.lateral_accel_weight) * lat_activity,
            0.0,
            max(0.0, _safe_float(cfg.max_roll_activity)))
        roll_stiffness_activity = clamp(
            max(
                lat_activity,
                _safe_float(cfg.roll_angle_to_spring_weight) *
                roll_angle_activity),
            0.0,
            max(0.0, _safe_float(cfg.max_roll_activity)))

        outer_side_sign = self._outer_side_sign(local_ay, roll_rad)
        side_weights = tuple(
            self._side_weight(y, outer_side_sign)
            for _, y in corners)
        front_share, yaw_ref, under_yaw_error_norm = self._front_share(
            local_ay=local_ay,
            speed=speed,
            steer=steer,
            yaw_rate_rad=yaw_rate_rad,
            dt=dt)
        front_axle_weight = 2.0 * front_share
        rear_axle_weight = 2.0 * (1.0 - front_share)

        desired_springs = []
        desired_dampers = []
        roll_damper_adds = []
        roll_spring_adds = []
        for index, ((x, _), side_weight, damper_skyhook) in enumerate(zip(
                corners,
                side_weights,
                skyhook_dampers)):
            axle_weight = front_axle_weight if x > 0.0 else rear_axle_weight
            roll_damper_add = min(
                max(
                    0.0,
                    _safe_float(cfg.roll_damper_gain) *
                    roll_damping_activity *
                    axle_weight *
                    side_weight),
                max(0.0, _safe_float(cfg.max_roll_damper_add)))
            desired_damper = clamp(
                damper_skyhook + roll_damper_add,
                cfg.min_damper_scale,
                cfg.max_damper_scale)
            desired_damper = self._rate_limit(
                desired_damper,
                self.previous_damper_scales[index],
                cfg.max_damper_delta_per_step)

            roll_spring_add = 0.0
            desired_spring = _FROZEN_SPRING_SCALE

            desired_springs.append(desired_spring)
            desired_dampers.append(desired_damper)
            roll_damper_adds.append(roll_damper_add)
            roll_spring_adds.append(roll_spring_add)

        self.previous_spring_scales = list(desired_springs)
        self.previous_damper_scales = list(desired_dampers)
        self.previous_front_share = front_share

        command = SuspensionCommand(tuple(
            WheelScale(spring_scale=spring, damper_scale=damper)
            for spring, damper in zip(desired_springs, desired_dampers)
        )).validate(expected_wheels=4)

        diagnostics = self._diagnostics(
            roll_rad=roll_rad,
            roll_rate_rad=roll_rate_rad,
            pitch_rate_rad=pitch_rate_rad,
            yaw_rate_rad=yaw_rate_rad,
            local_ay=local_ay,
            lat_activity=lat_activity,
            roll_damping_activity=roll_damping_activity,
            roll_stiffness_activity=roll_stiffness_activity,
            front_share=front_share,
            yaw_ref=yaw_ref,
            under_yaw_error_norm=under_yaw_error_norm,
            outer_side_sign=outer_side_sign,
            side_weights=side_weights,
            corner_velocities=corner_velocities,
            skyhook_activities=skyhook_activities,
            skyhook_dampers=skyhook_dampers,
            roll_damper_adds=tuple(roll_damper_adds),
            roll_spring_adds=tuple(roll_spring_adds),
            spring_scales=tuple(desired_springs),
            damper_scales=tuple(desired_dampers),
            fallback_reason="")
        return ControllerOutput(command=command, diagnostics=diagnostics)

    def _initial_damper_scales(self) -> list:
        return [
            _safe_float(self.config.base_damper_scale, 1.0)
            for _ in range(max(1, int(self.config.wheel_count)))
        ]

    def _initial_spring_scales(self) -> list:
        return [
            _FROZEN_SPRING_SCALE
            for _ in range(max(1, int(self.config.wheel_count)))
        ]

    def _fallback_output(
        self,
        reason: str,
        context: Optional[ControllerContext] = None,
    ) -> ControllerOutput:
        cfg = self.config
        wheel_count = self._fallback_wheel_count(context)
        spring = _FROZEN_SPRING_SCALE
        damper = clamp(
            _safe_float(cfg.base_damper_scale, 1.0),
            cfg.min_damper_scale,
            cfg.max_damper_scale)
        command = SuspensionCommand.uniform(
            spring_scale=spring,
            damper_scale=damper,
            wheel_count=wheel_count).validate(expected_wheels=wheel_count)
        diagnostics = self._diagnostics(
            roll_rad=0.0,
            roll_rate_rad=0.0,
            pitch_rate_rad=0.0,
            yaw_rate_rad=0.0,
            local_ay=0.0,
            lat_activity=0.0,
            roll_damping_activity=0.0,
            roll_stiffness_activity=0.0,
            front_share=self._nominal_front_share(),
            yaw_ref=0.0,
            under_yaw_error_norm=0.0,
            outer_side_sign=0.0,
            side_weights=(1.0,) * wheel_count,
            corner_velocities=(0.0,) * wheel_count,
            skyhook_activities=(0.0,) * wheel_count,
            skyhook_dampers=(damper,) * wheel_count,
            roll_damper_adds=(0.0,) * wheel_count,
            roll_spring_adds=(0.0,) * wheel_count,
            spring_scales=(spring,) * wheel_count,
            damper_scales=(damper,) * wheel_count,
            fallback_reason=reason)
        return ControllerOutput(command=command, diagnostics=diagnostics)

    def _fallback_wheel_count(
        self,
        context: Optional[ControllerContext],
    ) -> int:
        if context is not None:
            for source_name in ("native_suspension", "current_suspension"):
                source = getattr(context, source_name, None)
                wheels = getattr(source, "wheels", None)
                if wheels is not None:
                    try:
                        wheel_count = len(wheels)
                    except TypeError:
                        continue
                    if wheel_count > 0:
                        return wheel_count
        return 4

    def _diagnostics(
        self,
        roll_rad: float,
        roll_rate_rad: float,
        pitch_rate_rad: float,
        yaw_rate_rad: float,
        local_ay: float,
        lat_activity: float,
        roll_damping_activity: float,
        roll_stiffness_activity: float,
        front_share: float,
        yaw_ref: float,
        under_yaw_error_norm: float,
        outer_side_sign: float,
        side_weights: Sequence[float],
        corner_velocities: Sequence[float],
        skyhook_activities: Sequence[float],
        skyhook_dampers: Sequence[float],
        roll_damper_adds: Sequence[float],
        roll_spring_adds: Sequence[float],
        spring_scales: Sequence[float],
        damper_scales: Sequence[float],
        fallback_reason: str,
    ) -> Dict[str, Any]:
        controller_name = (
            "skyhook_roll_yaw"
            if self.config.enable_yaw_distribution
            else self.name)
        diagnostics: Dict[str, Any] = {
            "controller": controller_name,
            "spring_scale": _mean(spring_scales),
            "damper_scale": _mean(damper_scales),
            "skyhook_roll_mode": (
                "yaw" if self.config.enable_yaw_distribution else "fixed"),
            "skyhook_roll_roll_rad": roll_rad,
            "skyhook_roll_roll_rate_rad": roll_rate_rad,
            "skyhook_roll_pitch_rate_rad": pitch_rate_rad,
            "skyhook_roll_yaw_rate_rad": yaw_rate_rad,
            "skyhook_roll_local_ay": local_ay,
            "skyhook_roll_lat_activity": lat_activity,
            "skyhook_roll_damping_activity": roll_damping_activity,
            "skyhook_roll_stiffness_activity": roll_stiffness_activity,
            "skyhook_roll_front_share": front_share,
            "skyhook_roll_yaw_ref": yaw_ref,
            "skyhook_roll_under_yaw_error_norm": under_yaw_error_norm,
            "skyhook_roll_outer_side_sign": outer_side_sign,
            "skyhook_roll_fallback_reason": fallback_reason,
            "skyhook_mean_abs_corner_vz": _mean_abs(corner_velocities),
            "skyhook_max_activity": (
                max(skyhook_activities) if skyhook_activities else 0.0),
            "skyhook_roll_rate_rad": roll_rate_rad,
            "skyhook_pitch_rate_rad": pitch_rate_rad,
        }
        for label, corner_vz, activity, skyhook_damper in zip(
                _CORNER_LABELS,
                corner_velocities,
                skyhook_activities,
                skyhook_dampers):
            diagnostics["skyhook_corner_vz_%s" % label] = corner_vz
            diagnostics["skyhook_activity_%s" % label] = activity
            diagnostics["skyhook_damper_scale_%s" % label] = skyhook_damper
        for label, value in zip(_CORNER_LABELS, side_weights):
            diagnostics["skyhook_roll_side_weight_%s" % label] = value
        for label, value in zip(_CORNER_LABELS, spring_scales):
            diagnostics["skyhook_roll_spring_scale_%s" % label] = value
        for label, value in zip(_CORNER_LABELS, damper_scales):
            diagnostics["skyhook_roll_damper_scale_%s" % label] = value
        for label, value in zip(_CORNER_LABELS, roll_damper_adds):
            diagnostics["skyhook_roll_damper_add_%s" % label] = value
        for label, value in zip(_CORNER_LABELS, roll_spring_adds):
            diagnostics["skyhook_roll_spring_add_%s" % label] = value
        return diagnostics

    def _dt(self, context: ControllerContext) -> float:
        dt = _safe_float(getattr(context, "dt", 0.0), 0.0)
        if dt <= 0.0:
            dt = _safe_float(getattr(context.state, "dt", 0.0), 0.0)
        if dt <= 0.0:
            dt = _safe_float(self.config.default_dt, 0.05)
        return max(dt, _EPS)

    def _local_ay(self, state: VehicleState) -> float:
        local_ay = _safe_float(getattr(state, "local_ay", 0.0), None)
        if local_ay is not None:
            return local_ay
        return _safe_float(getattr(state, "ay", 0.0), 0.0)

    def _speed(self, state: VehicleState) -> float:
        speed = _safe_float(getattr(state, "speed", 0.0), None)
        if speed is not None and speed > 0.0:
            return speed
        vx = _safe_float(getattr(state, "vx", 0.0))
        vy = _safe_float(getattr(state, "vy", 0.0))
        vz = _safe_float(getattr(state, "vz", 0.0))
        velocity_norm = math.sqrt(vx * vx + vy * vy + vz * vz)
        if velocity_norm > 0.0:
            return velocity_norm
        if speed is not None and speed >= 0.0:
            return speed
        return 0.0

    def _steer(self, state: VehicleState) -> Optional[float]:
        return _safe_float(getattr(state, "steer", 0.0), None)

    def _corner_activity(self, corner_vz: float) -> float:
        cfg = self.config
        magnitude = max(
            0.0,
            abs(corner_vz) - max(0.0, _safe_float(cfg.corner_velocity_deadband)))
        return magnitude / max(abs(_safe_float(cfg.corner_velocity_scale, 1.0)), _EPS)

    def _activity(self, magnitude: float, deadband: float, scale: float) -> float:
        return max(0.0, magnitude - max(0.0, _safe_float(deadband))) / max(
            abs(_safe_float(scale, 1.0)),
            _EPS)

    def _corner_coordinates(self) -> Tuple[Tuple[float, float], ...]:
        cfg = self.config
        x_front = _safe_float(cfg.half_wheelbase_m, 1.45)
        x_rear = -x_front
        y_left = -_safe_float(cfg.half_track_m, 0.85)
        y_right = -y_left
        return (
            (x_front, y_left),
            (x_front, y_right),
            (x_rear, y_left),
            (x_rear, y_right),
        )

    def _outer_side_sign(self, local_ay: float, roll_rad: float) -> float:
        cfg = self.config
        if abs(local_ay) >= max(0.0, _safe_float(cfg.lateral_accel_side_deadband)):
            return _sign(_safe_float(cfg.outer_side_sign_from_ay) * _sign(local_ay))
        if (
                _safe_float(cfg.outer_side_sign_from_roll) != 0.0 and
                abs(roll_rad) >= max(0.0, _safe_float(cfg.roll_angle_deadband_rad))):
            return _sign(
                _safe_float(cfg.outer_side_sign_from_roll) *
                _sign(roll_rad))
        return 0.0

    def _side_weight(self, y: float, outer_side_sign: float) -> float:
        cfg = self.config
        if outer_side_sign == 0.0:
            return 1.0
        if _sign(y) == _sign(outer_side_sign):
            return max(0.0, 1.0 + _safe_float(cfg.outer_side_bias))
        return max(0.0, 1.0 - _safe_float(cfg.inner_side_relief))

    def _nominal_front_share(self) -> float:
        cfg = self.config
        return clamp(
            _safe_float(cfg.nominal_front_roll_distribution, 0.60),
            cfg.min_front_roll_distribution,
            cfg.max_front_roll_distribution)

    def _front_share(
        self,
        local_ay: float,
        speed: float,
        steer: Optional[float],
        yaw_rate_rad: float,
        dt: float,
    ) -> Tuple[float, float, float]:
        cfg = self.config
        nominal = self._nominal_front_share()
        yaw_ref = 0.0
        under_yaw_error_norm = 0.0
        desired = nominal

        if cfg.enable_yaw_distribution:
            yaw_ref = self._yaw_reference(speed, steer)
            activation = _ramp(
                abs(local_ay) / max(abs(_safe_float(cfg.g, 9.81)), _EPS),
                cfg.yaw_activation_start_g,
                cfg.yaw_activation_full_g)
            small_ref = max(abs(_safe_float(cfg.small_yaw_ref, 0.03)), _EPS)
            if activation > 0.0 and abs(yaw_ref) > small_ref:
                direction = _sign(yaw_ref)
                under_yaw_error = (yaw_ref - yaw_rate_rad) * direction
                under_yaw_error_norm = under_yaw_error / max(abs(yaw_ref), small_ref)
                integral_candidate = clamp(
                    self.yaw_error_integral + under_yaw_error_norm * dt,
                    -abs(_safe_float(cfg.yaw_error_integral_limit, 2.0)),
                    abs(_safe_float(cfg.yaw_error_integral_limit, 2.0)))
                delta_candidate = activation * (
                    -_safe_float(cfg.yaw_front_distribution_kp) *
                    under_yaw_error_norm -
                    _safe_float(cfg.yaw_front_distribution_ki) *
                    integral_candidate)
                unsaturated = nominal + delta_candidate
                low = _safe_float(cfg.min_front_roll_distribution, 0.35)
                high = _safe_float(cfg.max_front_roll_distribution, 0.75)
                if not (
                        (unsaturated < low and delta_candidate < 0.0) or
                        (unsaturated > high and delta_candidate > 0.0)):
                    self.yaw_error_integral = integral_candidate
                else:
                    self.yaw_error_integral *= _safe_float(
                        cfg.yaw_integral_decay,
                        0.98)
                front_delta = activation * (
                    -_safe_float(cfg.yaw_front_distribution_kp) *
                    under_yaw_error_norm -
                    _safe_float(cfg.yaw_front_distribution_ki) *
                    self.yaw_error_integral)
                desired = clamp(
                    nominal + front_delta,
                    cfg.min_front_roll_distribution,
                    cfg.max_front_roll_distribution)
            else:
                self.yaw_error_integral *= _safe_float(cfg.yaw_integral_decay, 0.98)

        return (
            self._rate_limit(
                desired,
                self.previous_front_share,
                cfg.max_front_distribution_delta_per_step),
            yaw_ref,
            under_yaw_error_norm)

    def _yaw_reference(self, speed: float, steer: Optional[float]) -> float:
        cfg = self.config
        min_speed = max(_safe_float(cfg.yaw_ref_min_speed, 5.0), _EPS)
        if steer is None or speed < min_speed:
            return 0.0
        wheel_angle = clamp(
            steer,
            -1.0,
            1.0) * abs(_safe_float(cfg.max_road_wheel_angle_rad, 0.60))
        wheelbase = max(2.0 * abs(_safe_float(cfg.half_wheelbase_m, 1.45)), _EPS)
        yaw_ref = speed * math.tan(wheel_angle) / wheelbase
        yaw_ref_limit = (
            abs(_safe_float(cfg.yaw_ref_limit_lat_g, 0.85)) *
            max(abs(_safe_float(cfg.g, 9.81)), _EPS) /
            max(speed, min_speed))
        return clamp(yaw_ref, -yaw_ref_limit, yaw_ref_limit)

    def _rate_limit(self, desired: float, previous: float, max_delta: float) -> float:
        max_step = max(0.0, _safe_float(max_delta, 0.0))
        if max_step == 0.0:
            return desired
        return clamp(desired, previous - max_step, previous + max_step)


def _safe_float(value: Any, default: Any = 0.0) -> Any:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _sign(value: float) -> float:
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return -1.0
    return 0.0


def _ramp(value: float, start: float, full: float) -> float:
    value = _safe_float(value)
    start = _safe_float(start)
    full = _safe_float(full)
    if full <= start:
        return 1.0 if value > start else 0.0
    if value <= start:
        return 0.0
    if value >= full:
        return 1.0
    return (value - start) / (full - start)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(_safe_float(value) for value in values) / float(len(values))


def _mean_abs(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(abs(_safe_float(value)) for value in values) / float(len(values))
