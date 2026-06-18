"""Target-speed-only damping schedule for Step 07 LEAD experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Dict, Mapping, Optional

from .base import (
    ControllerContext,
    ControllerOutput,
    SuspensionCommand,
    SuspensionController,
    clamp,
    finite_float,
)


@dataclass(frozen=True)
class TargetSpeedScheduleConfig:
    """Conservative target-speed-only uniform damping schedule.

    The controller intentionally consumes only ``PlanningInfo.target_speed``.
    It does not read curvature, trajectory, waypoints, steering, throttle,
    brake, sensors, or LEAD internals.
    """

    damper_min: float = 1.0
    damper_max: float = 1.05
    low_speed_gain: float = 0.03
    drop_gain: float = 0.02
    low_speed_reference_mps: float = 8.0
    drop_rate_reference_mps2: float = 2.0
    smoothing_time_constant_sec: float = 0.5
    max_scale_rate_per_sec: float = 0.10
    fallback_damper_scale: float = 1.0
    spring_scale: float = 1.0
    default_dt: float = 0.05
    wheel_count: int = 4
    shadow_mode: bool = False
    shadow_damper_scale: float = 1.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "TargetSpeedScheduleConfig":
        allowed = {item.name for item in fields(cls)}
        kwargs = {key: value for key, value in values.items() if key in allowed}
        return cls(**kwargs)


class TargetSpeedScheduleController(SuspensionController):
    """Uniform damper schedule driven only by planner target speed."""

    name = "target_speed_schedule"

    def __init__(self, config: Optional[TargetSpeedScheduleConfig] = None):
        self.config = config or TargetSpeedScheduleConfig()
        self._validate_config()
        self.target_speed_ema: Optional[float] = None
        self.previous_damper_scale = self.config.fallback_damper_scale

    def reset(self, native_suspension: Any = None) -> None:
        self.target_speed_ema = None
        self.previous_damper_scale = self.config.fallback_damper_scale

    def compute(self, context: ControllerContext) -> ControllerOutput:
        cfg = self.config
        dt = self._dt(context)
        target_speed_raw, fallback_reason = self._target_speed(context)
        valid = fallback_reason == ""

        if not valid:
            self.target_speed_ema = None
            self.previous_damper_scale = cfg.fallback_damper_scale
            return self._output(
                damper_scale=cfg.fallback_damper_scale,
                target_speed_raw=target_speed_raw,
                target_speed_valid=0,
                target_speed_ema="",
                low_speed_component=0.0,
                target_speed_drop_rate=0.0,
                drop_component=0.0,
                damper_scale_raw=cfg.fallback_damper_scale,
                damper_scale_bounded=cfg.fallback_damper_scale,
                rate_limited=0,
                fallback_reason=fallback_reason)

        assert target_speed_raw is not None
        previous_ema = self.target_speed_ema
        target_speed_ema = self._smooth_target_speed(target_speed_raw, dt)
        if previous_ema is None:
            target_speed_drop_rate = 0.0
        else:
            target_speed_drop_rate = max(
                0.0,
                (previous_ema - target_speed_ema) / max(dt, 1.0e-9))

        low_speed_component = cfg.low_speed_gain * clamp(
            (cfg.low_speed_reference_mps - target_speed_ema) /
            max(abs(cfg.low_speed_reference_mps), 1.0e-9),
            0.0,
            1.0)
        drop_component = cfg.drop_gain * clamp(
            target_speed_drop_rate / max(abs(cfg.drop_rate_reference_mps2), 1.0e-9),
            0.0,
            1.0)
        damper_scale_raw = (
            cfg.fallback_damper_scale +
            low_speed_component +
            drop_component)
        damper_scale_bounded = clamp(
            damper_scale_raw,
            cfg.damper_min,
            cfg.damper_max)
        damper_scale_applied, rate_limited = self._rate_limit(
            damper_scale_bounded,
            dt)

        return self._output(
            damper_scale=damper_scale_applied,
            target_speed_raw=target_speed_raw,
            target_speed_valid=1,
            target_speed_ema=target_speed_ema,
            low_speed_component=low_speed_component,
            target_speed_drop_rate=target_speed_drop_rate,
            drop_component=drop_component,
            damper_scale_raw=damper_scale_raw,
            damper_scale_bounded=damper_scale_bounded,
            rate_limited=rate_limited,
            fallback_reason="")

    def _validate_config(self) -> None:
        cfg = self.config
        finite_float(cfg.damper_min, "damper_min")
        finite_float(cfg.damper_max, "damper_max")
        finite_float(cfg.low_speed_gain, "low_speed_gain")
        finite_float(cfg.drop_gain, "drop_gain")
        finite_float(cfg.low_speed_reference_mps, "low_speed_reference_mps")
        finite_float(cfg.drop_rate_reference_mps2, "drop_rate_reference_mps2")
        finite_float(
            cfg.smoothing_time_constant_sec,
            "smoothing_time_constant_sec")
        finite_float(cfg.max_scale_rate_per_sec, "max_scale_rate_per_sec")
        finite_float(cfg.fallback_damper_scale, "fallback_damper_scale")
        finite_float(cfg.spring_scale, "spring_scale")
        finite_float(cfg.default_dt, "default_dt")
        finite_float(cfg.shadow_damper_scale, "shadow_damper_scale")
        if cfg.damper_min <= 0.0:
            raise ValueError("damper_min must be positive")
        if cfg.damper_max < cfg.damper_min:
            raise ValueError("damper_max must be >= damper_min")
        if cfg.spring_scale <= 0.0:
            raise ValueError("spring_scale must be positive")
        if int(cfg.wheel_count) <= 0:
            raise ValueError("wheel_count must be positive")
        if not (cfg.damper_min <= cfg.fallback_damper_scale <= cfg.damper_max):
            raise ValueError("fallback_damper_scale must be within damper bounds")
        if cfg.shadow_damper_scale <= 0.0:
            raise ValueError("shadow_damper_scale must be positive")

    def _dt(self, context: ControllerContext) -> float:
        dt = context.dt if context.dt > 0.0 else context.state.dt
        if dt <= 0.0 or not math.isfinite(dt):
            dt = self.config.default_dt
        return max(dt, 1.0e-9)

    def _target_speed(self, context: ControllerContext):
        planning = context.planning
        if planning is None:
            return None, "missing_planning"
        if not planning.available:
            return None, "planning_unavailable"
        if not planning.target_speed:
            return None, "empty_target_speed"
        try:
            target_speed = float(planning.target_speed[0])
        except (TypeError, ValueError):
            return None, "nonfinite_target_speed"
        if not math.isfinite(target_speed):
            return target_speed, "nonfinite_target_speed"
        if target_speed < 0.0:
            return target_speed, "negative_target_speed"
        return target_speed, ""

    def _smooth_target_speed(self, target_speed: float, dt: float) -> float:
        tau = max(0.0, self.config.smoothing_time_constant_sec)
        if self.target_speed_ema is None or tau == 0.0:
            self.target_speed_ema = target_speed
        else:
            alpha = clamp(dt / (tau + dt), 0.0, 1.0)
            self.target_speed_ema += alpha * (target_speed - self.target_speed_ema)
        return self.target_speed_ema

    def _rate_limit(self, desired_damper_scale: float, dt: float):
        max_delta = max(0.0, self.config.max_scale_rate_per_sec) * dt
        if max_delta == 0.0:
            limited = desired_damper_scale
        else:
            limited = clamp(
                desired_damper_scale,
                self.previous_damper_scale - max_delta,
                self.previous_damper_scale + max_delta)
        rate_limited = int(not math.isclose(
            limited,
            desired_damper_scale,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12))
        self.previous_damper_scale = limited
        return limited, rate_limited

    def _output(
        self,
        damper_scale: float,
        target_speed_raw: Optional[float],
        target_speed_valid: int,
        target_speed_ema: Any,
        low_speed_component: float,
        target_speed_drop_rate: float,
        drop_component: float,
        damper_scale_raw: float,
        damper_scale_bounded: float,
        rate_limited: int,
        fallback_reason: str,
    ) -> ControllerOutput:
        cfg = self.config
        command_damper_scale = (
            cfg.shadow_damper_scale if cfg.shadow_mode else damper_scale)
        command = SuspensionCommand.uniform(
            spring_scale=cfg.spring_scale,
            damper_scale=command_damper_scale,
            wheel_count=int(cfg.wheel_count)).validate(
                expected_wheels=int(cfg.wheel_count))
        diagnostics: Dict[str, Any] = {
            "controller": self.name,
            "spring_scale": cfg.spring_scale,
            "damper_scale": command_damper_scale,
            "tss_target_speed_raw": (
                "" if target_speed_raw is None else target_speed_raw),
            "tss_target_speed_valid": target_speed_valid,
            "tss_target_speed_ema": target_speed_ema,
            "tss_low_speed_component": low_speed_component,
            "tss_target_speed_drop_rate": target_speed_drop_rate,
            "tss_drop_component": drop_component,
            "tss_damper_scale_raw": damper_scale_raw,
            "tss_damper_scale_bounded": damper_scale_bounded,
            "tss_damper_scale_computed": damper_scale,
            "tss_damper_scale_applied": command_damper_scale,
            "tss_rate_limited": rate_limited,
            "tss_fallback_reason": fallback_reason,
            "tss_shadow_mode": int(cfg.shadow_mode),
            "tss_used_curvature": 0,
            "tss_used_trajectory": 0,
            "tss_used_control": 0,
        }
        return ControllerOutput(command=command, diagnostics=diagnostics)
