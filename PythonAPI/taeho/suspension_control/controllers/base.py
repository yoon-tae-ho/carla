"""Common controller interface and CARLA-independent data objects."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


DEFAULT_WHEEL_ORDER = ("front_left", "front_right", "rear_left", "rear_right")


def clamp(value: float, low: float, high: float) -> float:
    if low > high:
        raise ValueError("low must be <= high")
    return max(low, min(high, value))


def finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("%s must be finite, got %r" % (name, value))
    return result


@dataclass(frozen=True)
class WheelScale:
    """Per-wheel spring and damper scale relative to native CARLA values."""

    spring_scale: float = 1.0
    damper_scale: float = 1.0

    def validate(self) -> "WheelScale":
        spring = finite_float(self.spring_scale, "spring_scale")
        damper = finite_float(self.damper_scale, "damper_scale")
        if spring <= 0.0:
            raise ValueError("spring_scale must be positive, got %r" % spring)
        if damper <= 0.0:
            raise ValueError("damper_scale must be positive, got %r" % damper)
        return self

    def clipped(
        self,
        min_spring_scale: float,
        max_spring_scale: float,
        min_damper_scale: float,
        max_damper_scale: float,
    ) -> "WheelScale":
        return WheelScale(
            spring_scale=clamp(
                self.spring_scale,
                min_spring_scale,
                max_spring_scale),
            damper_scale=clamp(
                self.damper_scale,
                min_damper_scale,
                max_damper_scale),
        )


@dataclass(frozen=True)
class SuspensionCommand:
    """Command returned by controllers.

    The command is scale-based on purpose. CARLA's runtime API expects absolute
    spring/damper values, but scale factors make controllers portable across
    vehicles and across native wheel values.
    """

    wheels: Tuple[WheelScale, ...] = field(
        default_factory=lambda: tuple(WheelScale() for _ in DEFAULT_WHEEL_ORDER))

    @classmethod
    def identity(cls, wheel_count: int = 4) -> "SuspensionCommand":
        return cls(tuple(WheelScale() for _ in range(wheel_count)))

    @classmethod
    def uniform(
        cls,
        spring_scale: float = 1.0,
        damper_scale: float = 1.0,
        wheel_count: int = 4,
    ) -> "SuspensionCommand":
        return cls(tuple(
            WheelScale(spring_scale, damper_scale)
            for _ in range(wheel_count)))

    def validate(self, expected_wheels: Optional[int] = 4) -> "SuspensionCommand":
        if expected_wheels is not None and len(self.wheels) != expected_wheels:
            raise ValueError(
                "expected %d wheels, got %d" %
                (expected_wheels, len(self.wheels)))
        for wheel in self.wheels:
            wheel.validate()
        return self

    def clipped(
        self,
        min_spring_scale: float,
        max_spring_scale: float,
        min_damper_scale: float,
        max_damper_scale: float,
    ) -> "SuspensionCommand":
        return SuspensionCommand(tuple(
            wheel.clipped(
                min_spring_scale,
                max_spring_scale,
                min_damper_scale,
                max_damper_scale)
            for wheel in self.wheels))

    def is_close(
        self,
        other: Optional["SuspensionCommand"],
        rel_tol: float = 1.0e-6,
        abs_tol: float = 1.0e-6,
    ) -> bool:
        if other is None or len(self.wheels) != len(other.wheels):
            return False
        for left, right in zip(self.wheels, other.wheels):
            if not math.isclose(
                    left.spring_scale,
                    right.spring_scale,
                    rel_tol=rel_tol,
                    abs_tol=abs_tol):
                return False
            if not math.isclose(
                    left.damper_scale,
                    right.damper_scale,
                    rel_tol=rel_tol,
                    abs_tol=abs_tol):
                return False
        return True

    def as_scale_lists(self) -> Dict[str, Tuple[float, ...]]:
        return {
            "spring_scales": tuple(wheel.spring_scale for wheel in self.wheels),
            "damper_scales": tuple(wheel.damper_scale for wheel in self.wheels),
        }


@dataclass(frozen=True)
class PlanningPoint:
    """One future point from an autonomous-driving planner."""

    time_seconds: float = 0.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    speed: float = 0.0
    yaw: float = 0.0
    curvature: float = 0.0


@dataclass(frozen=True)
class PlanningInfo:
    """Optional planning preview for proactive controllers.

    The original public contract was ``points``, ``source``, and ``metadata``.
    Those fields stay intact while richer preview arrays live alongside them.
    """

    points: Tuple[PlanningPoint, ...] = ()
    available: bool = False
    source: str = "empty"
    frame: int = -1
    horizon_dt: float = 0.1

    trajectory_xy: Tuple[Tuple[float, float], ...] = ()
    trajectory_yaw: Tuple[float, ...] = ()
    target_speed: Tuple[float, ...] = ()
    curvature: Tuple[float, ...] = ()

    steer: Tuple[float, ...] = ()
    throttle: Tuple[float, ...] = ()
    brake: Tuple[float, ...] = ()

    predicted_ax: Tuple[float, ...] = ()
    predicted_ay: Tuple[float, ...] = ()

    route_deviation: Optional[float] = None
    lane_invasion_count: int = 0
    collision_count: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> "PlanningInfo":
        return cls()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "points": tuple(point.__dict__ for point in self.points),
            "available": bool(self.available),
            "source": self.source,
            "frame": self.frame,
            "horizon_dt": self.horizon_dt,
            "trajectory_xy": self.trajectory_xy,
            "trajectory_yaw": self.trajectory_yaw,
            "target_speed": self.target_speed,
            "curvature": self.curvature,
            "steer": self.steer,
            "throttle": self.throttle,
            "brake": self.brake,
            "predicted_ax": self.predicted_ax,
            "predicted_ay": self.predicted_ay,
            "route_deviation": self.route_deviation,
            "lane_invasion_count": self.lane_invasion_count,
            "collision_count": self.collision_count,
            "metadata": dict(self.metadata or {}),
            "extra": dict(self.extra or {}),
        }

    def preview_summary(self) -> Dict[str, float]:
        curvatures = _planning_values(
            self.curvature,
            (point.curvature for point in self.points))
        speeds = _planning_values(
            self.target_speed,
            (point.speed for point in self.points))
        horizon_dt = max(_safe_float(self.horizon_dt, 0.1), 1.0e-6)
        predicted_ax = _planning_values(
            self.predicted_ax,
            _speed_derivatives(speeds, horizon_dt))
        predicted_ay = _planning_values(
            self.predicted_ay,
            (speed * speed * curvature
             for speed, curvature in zip(speeds, curvatures)))
        steer = _planning_values(self.steer, ())
        throttle = _planning_values(self.throttle, ())
        brake = _planning_values(self.brake, ())

        available = bool(
            self.available or
            self.points or
            self.trajectory_xy or
            speeds or
            curvatures or
            steer or
            throttle or
            brake)

        return {
            "planning_available": 1.0 if available else 0.0,
            "preview_curvature_now": curvatures[0] if curvatures else 0.0,
            "preview_curvature_mean": mean_abs_signed(curvatures),
            "preview_curvature_max_abs": _max_abs(curvatures),
            "preview_curvature_signed_peak": _signed_peak(curvatures),
            "preview_target_speed_now": speeds[0] if speeds else 0.0,
            "preview_target_speed_mean": mean_abs_signed(speeds),
            "preview_target_speed_min": min(speeds) if speeds else 0.0,
            "preview_target_speed_max": max(speeds) if speeds else 0.0,
            "preview_longitudinal_acc_mean": mean_abs_signed(predicted_ax),
            "preview_longitudinal_acc_max_abs": _max_abs(predicted_ax),
            "preview_lateral_acc_mean": mean_abs_signed(predicted_ay),
            "preview_lateral_acc_max_abs": _max_abs(predicted_ay),
            "preview_steer_now": steer[0] if steer else 0.0,
            "preview_steer_mean": mean_abs_signed(steer),
            "preview_steer_max_abs": _max_abs(steer),
            "preview_throttle_mean": mean_abs_signed(throttle),
            "preview_brake_mean": mean_abs_signed(brake),
            "preview_brake_max": max(brake) if brake else 0.0,
            "time_to_hard_brake": _time_to_threshold(
                brake,
                horizon_dt,
                lambda value: value >= 0.25),
            "time_to_sharp_turn": _time_to_threshold(
                curvatures,
                horizon_dt,
                lambda value: abs(value) >= 0.04),
            "time_to_high_lateral_acc": _time_to_threshold(
                predicted_ay,
                horizon_dt,
                lambda value: abs(value) >= 2.5),
        }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _planning_values(
    primary: Sequence[float],
    fallback: Sequence[float],
) -> Tuple[float, ...]:
    values = tuple(
        _safe_float(value)
        for value in (() if primary is None else primary))
    if values:
        return values
    return tuple(_safe_float(value) for value in fallback)


def _speed_derivatives(
    speeds: Sequence[float],
    horizon_dt: float,
) -> Tuple[float, ...]:
    if len(speeds) < 2:
        return ()
    return tuple(
        (speeds[index + 1] - speeds[index]) / horizon_dt
        for index in range(len(speeds) - 1))


def mean_abs_signed(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(float(value) for value in values) / float(len(values))


def _max_abs(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return max(abs(float(value)) for value in values)


def _signed_peak(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return max((float(value) for value in values), key=lambda value: abs(value))


def _time_to_threshold(
    values: Sequence[float],
    horizon_dt: float,
    predicate: Any,
) -> float:
    for index, value in enumerate(values):
        if predicate(float(value)):
            return float(index) * horizon_dt
    return 0.0


@dataclass(frozen=True)
class VehicleState:
    """Vehicle feedback snapshot in the same units used by CARLA scripts.

    Angles are degrees. Angular rates follow CARLA's `get_angular_velocity`
    output, which existing project scripts treat as deg/s.
    """

    step: int = 0
    frame: int = 0
    elapsed_seconds: float = 0.0
    dt: float = 0.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    speed: float = 0.0
    local_vx: float = 0.0
    local_vy: float = 0.0
    ax: float = 0.0
    ay: float = 0.0
    az: float = 0.0
    local_ax: float = 0.0
    local_ay: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    roll_rate: float = 0.0
    pitch_rate: float = 0.0
    yaw_rate: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    steer: float = 0.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "VehicleState":
        kwargs = {}
        for name in cls.__dataclass_fields__:
            if name in values:
                kwargs[name] = values[name]
        return cls(**kwargs)

    def as_dict(self) -> Dict[str, float]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class ControllerContext:
    """Inputs passed to a suspension controller at one control step."""

    state: VehicleState
    previous_state: Optional[VehicleState] = None
    planning: PlanningInfo = field(default_factory=PlanningInfo.empty)
    native_suspension: Any = None
    current_suspension: Any = None
    suspension_state: Any = None
    suspension_state_valid: bool = False
    suspension_state_invalid_reason: str = ""
    native_spring_strength_by_wheel: Tuple[float, ...] = ()
    native_damper_rate_by_wheel: Tuple[float, ...] = ()
    step: int = 0
    dt: float = 0.05


@dataclass(frozen=True)
class ControllerOutput:
    """Controller result for one step."""

    command: SuspensionCommand
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


class SuspensionController:
    """Base class for all suspension controllers."""

    name = "base"

    def reset(self, native_suspension: Any = None) -> None:
        """Reset controller state before a new episode."""

    def compute(self, context: ControllerContext) -> ControllerOutput:
        raise NotImplementedError


def command_from_uniform_scales(
    spring_scale: float,
    damper_scale: float,
    wheel_count: int = 4,
) -> SuspensionCommand:
    return SuspensionCommand.uniform(
        spring_scale=spring_scale,
        damper_scale=damper_scale,
        wheel_count=wheel_count)


def ensure_positive_finite_scales(command: SuspensionCommand) -> SuspensionCommand:
    return command.validate(expected_wheels=len(command.wheels))


def mean_abs(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(abs(value) for value in values) / float(len(values))
