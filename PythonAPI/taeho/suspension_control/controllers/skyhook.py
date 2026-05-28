"""Skyhook-style semi-active damping controller."""

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


@dataclass(frozen=True)
class SkyhookConfig:
    """Configuration for a simple skyhook-style damping baseline.

    CARLA's current experiment state does not expose suspension deflection or
    relative wheel velocity. This controller therefore estimates each corner's
    sprung-mass vertical velocity from body heave, roll rate, and pitch rate,
    then increases damping at corners with larger body vertical motion.
    Springs are intentionally frozen at identity scale so the experiment
    isolates damping.
    """

    default_dt: float = 0.05
    wheel_count: int = 4

    base_damper_scale: float = 1.0
    min_damper_scale: float = 0.80
    max_damper_scale: float = 1.30
    max_damper_delta_per_step: float = 0.04

    half_track_m: float = 0.85
    half_wheelbase_m: float = 1.45
    corner_velocity_scale: float = 1.0
    corner_velocity_deadband: float = 0.02
    skyhook_gain: float = 0.25

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SkyhookConfig":
        allowed = {item.name for item in fields(cls)}
        kwargs = {key: value for key, value in values.items() if key in allowed}
        return cls(**kwargs)


class SkyhookController(SuspensionController):
    """Per-corner skyhook-style damper controller.

    The controller is intentionally small and feedback-only. It does not try to
    track a target body activity like the PID controller; it maps estimated
    sprung-mass corner velocity directly to damper scale.
    """

    name = "skyhook"

    def __init__(self, config: Optional[SkyhookConfig] = None):
        self.config = config or SkyhookConfig()
        self.previous_damper_scales = [
            self.config.base_damper_scale for _ in range(self.config.wheel_count)
        ]

    def reset(self, native_suspension: Any = None) -> None:
        self.previous_damper_scales = [
            self.config.base_damper_scale for _ in range(self.config.wheel_count)
        ]

    def compute(self, context: ControllerContext) -> ControllerOutput:
        cfg = self.config
        state = context.state
        spring_scale = 1.0

        corner_velocities = self._corner_vertical_velocities(state)
        damper_scales = []
        activities = []
        for index, corner_vz in enumerate(corner_velocities):
            activity = self._corner_activity(corner_vz)
            desired = clamp(
                cfg.base_damper_scale + cfg.skyhook_gain * activity,
                cfg.min_damper_scale,
                cfg.max_damper_scale)
            damper_scales.append(self._rate_limit_damper(index, desired))
            activities.append(activity)

        command = SuspensionCommand(tuple(
            WheelScale(spring_scale=spring_scale, damper_scale=damper_scale)
            for damper_scale in damper_scales
        )).validate(expected_wheels=cfg.wheel_count)

        mean_damper_scale = sum(damper_scales) / float(len(damper_scales))
        mean_abs_corner_vz = (
            sum(abs(value) for value in corner_velocities) /
            float(len(corner_velocities)))
        diagnostics: Dict[str, Any] = {
            "controller": self.name,
            "spring_scale": spring_scale,
            "damper_scale": mean_damper_scale,
            "skyhook_mean_abs_corner_vz": mean_abs_corner_vz,
            "skyhook_max_activity": max(activities) if activities else 0.0,
            "skyhook_roll_rate_rad": math.radians(state.roll_rate),
            "skyhook_pitch_rate_rad": math.radians(state.pitch_rate),
        }
        for label, corner_vz, activity, damper_scale in zip(
                ("fl", "fr", "rl", "rr"),
                corner_velocities,
                activities,
                damper_scales):
            diagnostics["skyhook_corner_vz_%s" % label] = corner_vz
            diagnostics["skyhook_activity_%s" % label] = activity
            diagnostics["skyhook_damper_scale_%s" % label] = damper_scale
        return ControllerOutput(command=command, diagnostics=diagnostics)

    def _corner_activity(self, corner_vz: float) -> float:
        cfg = self.config
        magnitude = max(0.0, abs(corner_vz) - cfg.corner_velocity_deadband)
        return magnitude / max(abs(cfg.corner_velocity_scale), 1.0e-9)

    def _rate_limit_damper(self, index: int, desired_damper_scale: float) -> float:
        cfg = self.config
        max_delta = max(0.0, cfg.max_damper_delta_per_step)
        previous = self.previous_damper_scales[index]
        if max_delta == 0.0:
            limited = desired_damper_scale
        else:
            limited = clamp(
                desired_damper_scale,
                previous - max_delta,
                previous + max_delta)
        self.previous_damper_scales[index] = limited
        return limited

    def _corner_vertical_velocities(self, state: VehicleState) -> Tuple[float, ...]:
        cfg = self.config
        roll_rate = math.radians(state.roll_rate)
        pitch_rate = math.radians(state.pitch_rate)
        x_front = cfg.half_wheelbase_m
        x_rear = -cfg.half_wheelbase_m
        y_left = -cfg.half_track_m
        y_right = cfg.half_track_m
        corners = (
            (x_front, y_left),
            (x_front, y_right),
            (x_rear, y_left),
            (x_rear, y_right),
        )
        velocities = tuple(
            state.vz + roll_rate * y - pitch_rate * x
            for x, y in corners)
        return self._match_wheel_count(velocities)

    def _match_wheel_count(self, values: Sequence[float]) -> Tuple[float, ...]:
        wheel_count = int(self.config.wheel_count)
        if wheel_count == len(values):
            return tuple(values)
        if wheel_count < len(values):
            return tuple(values[:wheel_count])
        if not values:
            return tuple(0.0 for _ in range(wheel_count))
        return tuple(values) + tuple(
            values[-1] for _ in range(wheel_count - len(values)))
