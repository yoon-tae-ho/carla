"""Feedback PID suspension controller baseline."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Dict, Mapping, Optional

from .base import (
    ControllerContext,
    ControllerOutput,
    SuspensionCommand,
    SuspensionController,
    VehicleState,
    clamp,
)


def _scaled_abs(value: float, scale: float, deadband: float = 0.0) -> float:
    magnitude = max(0.0, abs(value) - deadband)
    return magnitude / max(abs(scale), 1.0e-9)


@dataclass(frozen=True)
class FeedbackPIDConfig:
    """Configuration for the feedback-only PID baseline.

    The controller compresses body motion into one positive "activity" signal
    and runs PID on that signal. The result changes a uniform damper scale.
    Spring scale is intentionally frozen so PID experiments isolate damping.
    """

    default_dt: float = 0.05
    wheel_count: int = 4

    base_spring_scale: float = 1.0
    base_damper_scale: float = 1.0
    min_spring_scale: float = 0.90
    max_spring_scale: float = 1.10
    min_damper_scale: float = 0.80
    max_damper_scale: float = 1.30
    max_damper_delta_per_step: float = 0.04
    # Kept for config compatibility; ignored while spring is frozen.
    spring_gain_from_damper_delta: float = 0.0

    kp: float = 0.22
    ki: float = 0.015
    kd: float = 0.025
    integral_limit: float = 2.0
    integral_decay_when_quiet: float = 0.98

    activity_target: float = 0.04
    activity_deadband: float = 0.01

    roll_angle_scale: float = 5.0
    pitch_angle_scale: float = 4.0
    roll_rate_scale: float = 35.0
    pitch_rate_scale: float = 35.0
    lateral_acc_scale: float = 5.0
    vertical_acc_scale: float = 4.0
    lateral_acc_deadband: float = 0.15
    vertical_acc_deadband: float = 0.20

    roll_angle_weight: float = 0.35
    pitch_angle_weight: float = 0.25
    roll_rate_weight: float = 0.20
    pitch_rate_weight: float = 0.15
    lateral_acc_weight: float = 0.25
    vertical_acc_weight: float = 0.20

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "FeedbackPIDConfig":
        allowed = {item.name for item in fields(cls)}
        kwargs = {key: value for key, value in values.items() if key in allowed}
        return cls(**kwargs)


class FeedbackPIDController(SuspensionController):
    """Simple feedback PID controller for comparison experiments.

    This is intentionally not a planning-preview controller. It uses only the
    current measured vehicle state, which makes it a clean feedback baseline for
    future proactive controllers.
    """

    name = "feedback_pid"

    def __init__(self, config: Optional[FeedbackPIDConfig] = None):
        self.config = config or FeedbackPIDConfig()
        self.integral_error = 0.0
        self.previous_error = None
        self.previous_damper_scale = self.config.base_damper_scale

    def reset(self, native_suspension: Any = None) -> None:
        self.integral_error = 0.0
        self.previous_error = None
        self.previous_damper_scale = self.config.base_damper_scale

    def compute(self, context: ControllerContext) -> ControllerOutput:
        cfg = self.config
        dt = context.dt if context.dt > 0.0 else cfg.default_dt
        state = context.state

        activity, components = self._body_activity(state)
        error = activity - cfg.activity_target
        if abs(error) < cfg.activity_deadband:
            error = 0.0

        if error <= 0.0:
            self.integral_error *= cfg.integral_decay_when_quiet
        else:
            self.integral_error = clamp(
                self.integral_error + error * dt,
                -cfg.integral_limit,
                cfg.integral_limit)

        if self.previous_error is None:
            derivative = 0.0
        else:
            derivative = (error - self.previous_error) / max(dt, 1.0e-9)
        self.previous_error = error

        p_term = cfg.kp * error
        i_term = cfg.ki * self.integral_error
        d_term = cfg.kd * derivative
        damper_delta = p_term + i_term + d_term

        desired_damper_scale = clamp(
            cfg.base_damper_scale + damper_delta,
            cfg.min_damper_scale,
            cfg.max_damper_scale)
        desired_damper_scale = self._rate_limit_damper(desired_damper_scale)

        spring_scale = clamp(
            cfg.base_spring_scale,
            cfg.min_spring_scale,
            cfg.max_spring_scale)

        command = SuspensionCommand.uniform(
            spring_scale=spring_scale,
            damper_scale=desired_damper_scale,
            wheel_count=cfg.wheel_count).validate(expected_wheels=cfg.wheel_count)

        diagnostics: Dict[str, Any] = {
            "controller": self.name,
            "activity": activity,
            "activity_error": error,
            "integral_error": self.integral_error,
            "derivative_error": derivative,
            "p_term": p_term,
            "i_term": i_term,
            "d_term": d_term,
            "spring_scale": spring_scale,
            "damper_scale": desired_damper_scale,
        }
        diagnostics.update(components)
        return ControllerOutput(command=command, diagnostics=diagnostics)

    def _rate_limit_damper(self, desired_damper_scale: float) -> float:
        cfg = self.config
        max_delta = max(0.0, cfg.max_damper_delta_per_step)
        if max_delta == 0.0:
            limited = desired_damper_scale
        else:
            limited = clamp(
                desired_damper_scale,
                self.previous_damper_scale - max_delta,
                self.previous_damper_scale + max_delta)
        self.previous_damper_scale = limited
        return limited

    def _body_activity(self, state: VehicleState):
        cfg = self.config
        components = {
            "activity_roll_angle": cfg.roll_angle_weight * _scaled_abs(
                state.roll,
                cfg.roll_angle_scale),
            "activity_pitch_angle": cfg.pitch_angle_weight * _scaled_abs(
                state.pitch,
                cfg.pitch_angle_scale),
            "activity_roll_rate": cfg.roll_rate_weight * _scaled_abs(
                state.roll_rate,
                cfg.roll_rate_scale),
            "activity_pitch_rate": cfg.pitch_rate_weight * _scaled_abs(
                state.pitch_rate,
                cfg.pitch_rate_scale),
            "activity_lateral_acc": cfg.lateral_acc_weight * _scaled_abs(
                state.local_ay,
                cfg.lateral_acc_scale,
                cfg.lateral_acc_deadband),
            "activity_vertical_acc": cfg.vertical_acc_weight * _scaled_abs(
                state.az,
                cfg.vertical_acc_scale,
                cfg.vertical_acc_deadband),
        }
        activity = sum(components.values())
        if not math.isfinite(activity):
            raise ValueError("PID activity became non-finite: %r" % activity)
        return activity, components
