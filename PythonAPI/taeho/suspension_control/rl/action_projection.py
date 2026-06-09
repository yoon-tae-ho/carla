"""Shared residual action projection for training and deployment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ..controllers.base import (
    SuspensionCommand,
    WheelScale,
    clamp,
)


WHEEL_LABELS = ("fl", "fr", "rl", "rr")


@dataclass(frozen=True)
class ResidualActionProjectorConfig:
    wheel_count: int = 4
    spring_frozen: bool = True
    base_spring_scale: float = 1.0
    min_spring_scale: float = 0.95
    max_spring_scale: float = 1.05
    min_damper_scale: float = 0.80
    max_damper_scale: float = 1.30
    max_damper_residual_scale: float = 0.08
    max_damper_delta_per_step: float = 0.04
    min_speed_for_policy: float = 0.5
    max_abs_roll_for_full_policy: float = 6.0
    max_abs_pitch_for_full_policy: float = 6.0
    max_abs_lateral_acc_for_full_policy: float = 7.0
    max_abs_yaw_rate_for_full_policy: float = 60.0


@dataclass(frozen=True)
class ProjectionResult:
    command: SuspensionCommand
    sanitized_action: Tuple[float, ...]
    raw_residual_damper_per_wheel: Tuple[float, ...]
    scaled_residual_damper_per_wheel: Tuple[float, ...]
    scale_clipped_residual_damper_per_wheel: Tuple[float, ...]
    safety_scaled_residual_damper_per_wheel: Tuple[float, ...]
    rate_limited_residual_damper_per_wheel: Tuple[float, ...]
    final_residual_damper_per_wheel: Tuple[float, ...]
    baseline_damper_per_wheel: Tuple[float, ...]
    final_damper_per_wheel: Tuple[float, ...]
    safety_gain: float
    safety_gate_active: int
    safety_gate_reason: str
    fallback_reason: str
    action_scale: float
    residual_mode: str
    scripted_residual_kind: str
    diagnostics: Mapping[str, Any]

    @property
    def action(self) -> Tuple[float, ...]:
        return self.sanitized_action

    @property
    def raw_residuals(self) -> Tuple[float, ...]:
        return self.raw_residual_damper_per_wheel

    @property
    def safety_scaled_residuals(self) -> Tuple[float, ...]:
        return self.safety_scaled_residual_damper_per_wheel

    @property
    def scaled_residuals(self) -> Tuple[float, ...]:
        return self.scaled_residual_damper_per_wheel

    @property
    def scale_clipped_residuals(self) -> Tuple[float, ...]:
        return self.scale_clipped_residual_damper_per_wheel

    @property
    def rate_limited_residuals(self) -> Tuple[float, ...]:
        return self.rate_limited_residual_damper_per_wheel

    @property
    def final_residuals(self) -> Tuple[float, ...]:
        return self.final_residual_damper_per_wheel

    @property
    def baseline_dampers(self) -> Tuple[float, ...]:
        return self.baseline_damper_per_wheel

    @property
    def final_dampers(self) -> Tuple[float, ...]:
        return self.final_damper_per_wheel

    @property
    def safety_diagnostics(self) -> Dict[str, Any]:
        return {
            key: value
            for key, value in dict(self.diagnostics).items()
            if key == "rl_safety_gain" or key.startswith("rl_safety_gate_")
        }


class ResidualActionProjector:
    """Project normalized residual actions onto safe suspension commands."""

    def __init__(
        self,
        config: Optional[ResidualActionProjectorConfig] = None,
    ):
        self.config = config or ResidualActionProjectorConfig()
        self.previous_damper_scales: Optional[Tuple[float, ...]] = None

    def reset(self) -> None:
        self.previous_damper_scales = None

    def project(
        self,
        baseline_command: SuspensionCommand,
        action: Sequence[float],
        state: Any,
        observation_valid: bool = True,
        action_invalid: bool = False,
        obs_diag: Optional[Mapping[str, Any]] = None,
        previous_final_dampers: Optional[Sequence[float]] = None,
        action_scale: float = 1.0,
        residual_mode: str = "learned_policy",
        scripted_residual_kind: str = "",
        scripted_residual_value: Any = "",
        raw_residual_override: Optional[Sequence[float]] = None,
    ) -> ProjectionResult:
        cfg = self.config
        baseline = baseline_command.validate(expected_wheels=cfg.wheel_count)
        baseline_dampers = _damper_scales(baseline)
        action_scale = _finite_nonnegative(action_scale, 1.0)
        if obs_diag is not None:
            observation_valid = bool(
                dict(obs_diag).get("observation_valid", observation_valid))
        action_values, action_error = self.sanitize_action(action)
        action_invalid = bool(action_invalid or action_values is None)
        if action_values is None:
            action_values = tuple(0.0 for _ in range(cfg.wheel_count))

        safety_gain, safety_diag = self.safety_gate(
            state,
            observation_valid=observation_valid,
            action_invalid=action_invalid)

        if raw_residual_override is None:
            raw_residuals = tuple(
                value * max(0.0, cfg.max_damper_residual_scale)
                for value in action_values)
        else:
            raw_residuals = _sanitize_residuals(
                raw_residual_override,
                cfg.wheel_count)
        scaled_residuals = tuple(
            float(residual) * action_scale
            for residual in raw_residuals)
        scale_clip_limit = max(0.0, cfg.max_damper_residual_scale)
        scale_clipped_residuals = tuple(
            clamp(residual, -scale_clip_limit, scale_clip_limit)
            if scale_clip_limit > 0.0 else 0.0
            for residual in scaled_residuals)
        residual_scale_clip_active = int(any(
            abs(float(raw) - float(clipped)) > 1.0e-12
            for raw, clipped in zip(scaled_residuals, scale_clipped_residuals)))
        safety_scaled_residuals = tuple(
            residual * safety_gain
            for residual in scale_clipped_residuals)

        zero_residual = all(
            abs(value) <= 1.0e-12 for value in scale_clipped_residuals)
        if zero_residual:
            result = self._exact_baseline_result(
                baseline,
                action_values,
                raw_residuals,
                scaled_residuals,
                scale_clipped_residuals,
                safety_scaled_residuals,
                safety_gain,
                safety_diag,
                fallback_reason="",
                action_scale=action_scale,
                residual_mode=residual_mode,
                scripted_residual_kind=scripted_residual_kind,
                scripted_residual_value=scripted_residual_value,
                residual_scale_clip_active=residual_scale_clip_active)
            self.previous_damper_scales = result.final_damper_per_wheel
            return result

        if action_error:
            fallback_reason = action_error
        elif action_invalid:
            fallback_reason = "invalid_policy_action"
        elif safety_gain <= 0.0:
            fallback_reason = "safety_gate_zero"
        else:
            fallback_reason = ""

        if fallback_reason:
            result = self._exact_baseline_result(
                baseline,
                action_values,
                raw_residuals,
                scaled_residuals,
                scale_clipped_residuals,
                safety_scaled_residuals,
                safety_gain,
                safety_diag,
                fallback_reason=fallback_reason,
                action_scale=action_scale,
                residual_mode=residual_mode,
                scripted_residual_kind=scripted_residual_kind,
                scripted_residual_value=scripted_residual_value,
                residual_scale_clip_active=residual_scale_clip_active)
            self.previous_damper_scales = result.final_damper_per_wheel
            return result

        previous = (
            _match_length(previous_final_dampers, cfg.wheel_count, 1.0)
            if previous_final_dampers is not None
            else _match_length(
                self.previous_damper_scales,
                cfg.wheel_count,
                1.0)
            if self.previous_damper_scales is not None
            else baseline_dampers)
        wheels = []
        rate_limited_residuals = []
        final_residuals = []
        final_clamp_active = 0
        for index, (wheel, residual) in enumerate(
                zip(baseline.wheels, safety_scaled_residuals)):
            desired = float(wheel.damper_scale) + float(residual)
            max_delta = max(0.0, cfg.max_damper_delta_per_step)
            if max_delta > 0.0:
                desired = clamp(
                    desired,
                    previous[index] - max_delta,
                    previous[index] + max_delta)
            rate_limited_residuals.append(desired - wheel.damper_scale)
            desired_after_rate_limit = desired
            damper = clamp(
                desired,
                cfg.min_damper_scale,
                cfg.max_damper_scale)
            if abs(damper - desired_after_rate_limit) > 1.0e-12:
                final_clamp_active = 1
            if cfg.spring_frozen:
                spring = cfg.base_spring_scale
            else:
                spring = wheel.spring_scale
            spring = clamp(spring, cfg.min_spring_scale, cfg.max_spring_scale)
            wheels.append(WheelScale(spring_scale=spring, damper_scale=damper))
            final_residuals.append(damper - wheel.damper_scale)

        command = SuspensionCommand(tuple(wheels)).validate(
            expected_wheels=cfg.wheel_count)
        final_dampers = _damper_scales(command)
        self.previous_damper_scales = final_dampers
        return self._result(
            command,
            action_values,
            raw_residuals,
            scaled_residuals,
            scale_clipped_residuals,
            safety_scaled_residuals,
            tuple(rate_limited_residuals),
            tuple(final_residuals),
            baseline_dampers,
            final_dampers,
            safety_gain,
            safety_diag,
            fallback_reason="",
            action_scale=action_scale,
            residual_mode=residual_mode,
            scripted_residual_kind=scripted_residual_kind,
            scripted_residual_value=scripted_residual_value,
            residual_scale_clip_active=residual_scale_clip_active,
            final_clamp_active=final_clamp_active)

    def sanitize_action(
        self,
        action: Sequence[float],
    ) -> Tuple[Optional[Tuple[float, ...]], str]:
        try:
            values = list(action)
        except TypeError:
            return None, "policy_action_not_sequence"
        if len(values) != self.config.wheel_count:
            return None, "policy_action_shape_mismatch"
        result = []
        for value in values:
            try:
                number = float(value)
            except (TypeError, ValueError):
                return None, "policy_action_non_numeric"
            if not math.isfinite(number):
                return None, "policy_action_non_finite"
            result.append(clamp(number, -1.0, 1.0))
        return tuple(result), ""

    def safety_gate(
        self,
        state: Any,
        observation_valid: bool = True,
        nonfinite_obs: bool = False,
        action_invalid: bool = False,
    ) -> Tuple[float, Dict[str, Any]]:
        cfg = self.config
        speed, speed_finite = _finite_value(getattr(state, "speed", 0.0))
        roll, roll_finite = _finite_value(getattr(state, "roll", 0.0))
        pitch, pitch_finite = _finite_value(getattr(state, "pitch", 0.0))
        lateral_acc, lateral_finite = _finite_value(
            getattr(state, "local_ay", 0.0))
        yaw_rate, yaw_finite = _finite_value(getattr(state, "yaw_rate", 0.0))

        nonfinite_obs = bool(
            nonfinite_obs or
            not observation_valid or
            not all((
                speed_finite,
                roll_finite,
                pitch_finite,
                lateral_finite,
                yaw_finite,
            )))

        speed_limit = (
            not nonfinite_obs and
            speed < max(0.0, cfg.min_speed_for_policy))
        roll_limit = _limit_active(roll, cfg.max_abs_roll_for_full_policy)
        pitch_limit = _limit_active(pitch, cfg.max_abs_pitch_for_full_policy)
        lateral_limit = _limit_active(
            lateral_acc,
            cfg.max_abs_lateral_acc_for_full_policy)
        yaw_limit = _limit_active(
            yaw_rate,
            cfg.max_abs_yaw_rate_for_full_policy)

        reasons = []
        if nonfinite_obs:
            reasons.append("nonfinite_obs")
        if action_invalid:
            reasons.append("action_invalid")
        if speed_limit:
            reasons.append("speed_below_min")
        if roll_limit:
            reasons.append("roll_limit")
        if pitch_limit:
            reasons.append("pitch_limit")
        if lateral_limit:
            reasons.append("lateral_acc_limit")
        if yaw_limit:
            reasons.append("yaw_rate_limit")

        if nonfinite_obs or action_invalid or speed_limit:
            gain = 0.0
        else:
            gains = (
                _limit_gain(abs(roll), cfg.max_abs_roll_for_full_policy),
                _limit_gain(abs(pitch), cfg.max_abs_pitch_for_full_policy),
                _limit_gain(abs(lateral_acc), cfg.max_abs_lateral_acc_for_full_policy),
                _limit_gain(abs(yaw_rate), cfg.max_abs_yaw_rate_for_full_policy),
            )
            gain = clamp(min(gains), 0.0, 1.0)
        active = bool(reasons or gain < 1.0)
        return gain, {
            "rl_safety_gate_active": int(active),
            "rl_safety_gate_reason": ";".join(reasons),
            "rl_safety_gate_speed_limit": int(speed_limit),
            "rl_safety_gate_roll_limit": int(roll_limit),
            "rl_safety_gate_pitch_limit": int(pitch_limit),
            "rl_safety_gate_lateral_acc_limit": int(lateral_limit),
            "rl_safety_gate_yaw_rate_limit": int(yaw_limit),
            "rl_safety_gate_nonfinite_obs": int(nonfinite_obs),
            "rl_safety_gate_action_invalid": int(bool(action_invalid)),
        }

    def _exact_baseline_result(
        self,
        baseline: SuspensionCommand,
        action_values: Tuple[float, ...],
        raw_residuals: Tuple[float, ...],
        scaled_residuals: Tuple[float, ...],
        scale_clipped_residuals: Tuple[float, ...],
        safety_scaled_residuals: Tuple[float, ...],
        safety_gain: float,
        safety_diag: Mapping[str, Any],
        fallback_reason: str,
        action_scale: float,
        residual_mode: str,
        scripted_residual_kind: str,
        scripted_residual_value: Any,
        residual_scale_clip_active: int = 0,
    ) -> ProjectionResult:
        baseline_dampers = _damper_scales(baseline)
        zero_residuals = tuple(0.0 for _ in range(self.config.wheel_count))
        return self._result(
            baseline,
            action_values,
            raw_residuals,
            scaled_residuals,
            scale_clipped_residuals,
            safety_scaled_residuals,
            zero_residuals,
            zero_residuals,
            baseline_dampers,
            baseline_dampers,
            safety_gain,
            safety_diag,
            fallback_reason=fallback_reason,
            action_scale=action_scale,
            residual_mode=residual_mode,
            scripted_residual_kind=scripted_residual_kind,
            scripted_residual_value=scripted_residual_value,
            residual_scale_clip_active=residual_scale_clip_active,
            final_clamp_active=0)

    def _result(
        self,
        command: SuspensionCommand,
        action_values: Tuple[float, ...],
        raw_residuals: Tuple[float, ...],
        scaled_residuals: Tuple[float, ...],
        scale_clipped_residuals: Tuple[float, ...],
        safety_scaled_residuals: Tuple[float, ...],
        rate_limited_residuals: Tuple[float, ...],
        final_residuals: Tuple[float, ...],
        baseline_dampers: Tuple[float, ...],
        final_dampers: Tuple[float, ...],
        safety_gain: float,
        safety_diag: Mapping[str, Any],
        fallback_reason: str,
        action_scale: float,
        residual_mode: str,
        scripted_residual_kind: str,
        scripted_residual_value: Any,
        residual_scale_clip_active: int = 0,
        final_clamp_active: int = 0,
    ) -> ProjectionResult:
        diagnostics = dict(_empty_safety_gate_diagnostics())
        diagnostics.update(dict(safety_diag or {}))
        residual_saturation_active = int(
            bool(residual_scale_clip_active) or bool(final_clamp_active))
        diagnostics.update({
            "rl_residual_mode": residual_mode,
            "rl_scripted_residual_kind": scripted_residual_kind,
            "rl_scripted_residual_value": scripted_residual_value,
            "rl_action_scale": action_scale,
            "rl_residual_gain": action_scale,
            "rl_safety_gain": safety_gain,
            "rl_fallback_reason": fallback_reason,
            "rl_mean_action": _mean(action_values),
            "rl_mean_abs_action": _mean_abs(action_values),
            "rl_mean_raw_residual_damper": _mean(raw_residuals),
            "rl_mean_abs_raw_residual_damper": _mean_abs(raw_residuals),
            "rl_mean_scaled_residual_damper": _mean(scaled_residuals),
            "rl_mean_abs_scaled_residual_damper": _mean_abs(scaled_residuals),
            "rl_mean_scale_clipped_residual_damper": _mean(scale_clipped_residuals),
            "rl_mean_abs_scale_clipped_residual_damper": _mean_abs(scale_clipped_residuals),
            "rl_mean_residual_damper": _mean(final_residuals),
            "rl_mean_abs_residual_damper": _mean_abs(final_residuals),
            "rl_mean_final_residual_damper": _mean(final_residuals),
            "rl_mean_abs_final_residual_damper": _mean_abs(final_residuals),
            "rl_residual_scale_clip": int(bool(residual_scale_clip_active)),
            "rl_damper_final_clamp": int(bool(final_clamp_active)),
            "rl_residual_saturation": residual_saturation_active,
        })
        for label, value in zip(WHEEL_LABELS, action_values):
            diagnostics["rl_action_%s" % label] = value
        for label, value in zip(WHEEL_LABELS, raw_residuals):
            diagnostics["rl_raw_residual_damper_%s" % label] = value
        for label, value in zip(WHEEL_LABELS, scaled_residuals):
            diagnostics["rl_scaled_residual_damper_%s" % label] = value
        for label, value in zip(WHEEL_LABELS, scale_clipped_residuals):
            diagnostics["rl_scale_clipped_residual_damper_%s" % label] = value
        for label, value in zip(WHEEL_LABELS, safety_scaled_residuals):
            diagnostics["rl_safety_scaled_residual_damper_%s" % label] = value
        for label, value in zip(WHEEL_LABELS, rate_limited_residuals):
            diagnostics["rl_rate_limited_residual_damper_%s" % label] = value
        for label, value in zip(WHEEL_LABELS, final_residuals):
            diagnostics["rl_residual_damper_%s" % label] = value
            diagnostics["rl_final_residual_damper_%s" % label] = value
        for label, value in zip(WHEEL_LABELS, baseline_dampers):
            diagnostics["rl_baseline_damper_%s" % label] = value
        for label, value in zip(WHEEL_LABELS, final_dampers):
            diagnostics["rl_final_damper_%s" % label] = value
        return ProjectionResult(
            command=command,
            sanitized_action=action_values,
            raw_residual_damper_per_wheel=raw_residuals,
            scaled_residual_damper_per_wheel=scaled_residuals,
            scale_clipped_residual_damper_per_wheel=scale_clipped_residuals,
            safety_scaled_residual_damper_per_wheel=safety_scaled_residuals,
            rate_limited_residual_damper_per_wheel=rate_limited_residuals,
            final_residual_damper_per_wheel=final_residuals,
            baseline_damper_per_wheel=baseline_dampers,
            final_damper_per_wheel=final_dampers,
            safety_gain=safety_gain,
            safety_gate_active=int(diagnostics["rl_safety_gate_active"]),
            safety_gate_reason=str(diagnostics["rl_safety_gate_reason"]),
            fallback_reason=fallback_reason,
            action_scale=action_scale,
            residual_mode=residual_mode,
            scripted_residual_kind=scripted_residual_kind,
            diagnostics=diagnostics)


ResidualProjectionResult = ProjectionResult


def _damper_scales(command: SuspensionCommand) -> Tuple[float, ...]:
    return tuple(float(wheel.damper_scale) for wheel in command.wheels)


def _match_length(
    values: Optional[Sequence[float]],
    count: int,
    fill: float,
) -> Tuple[float, ...]:
    result = list(values or [])
    result = [float(value) for value in result[:count]]
    while len(result) < count:
        result.append(fill)
    return tuple(result)


def _sanitize_residuals(
    values: Sequence[float],
    count: int,
) -> Tuple[float, ...]:
    result = _match_length(values, count, 0.0)
    sanitized = []
    for value in result:
        number = _finite_nonnegative(abs(value), 0.0)
        sanitized.append(math.copysign(number, value))
    return tuple(sanitized)


def _finite_nonnegative(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number) or number < 0.0:
        return default
    return number


def _finite_value(value: Any) -> Tuple[float, bool]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0, False
    if not math.isfinite(number):
        return 0.0, False
    return number, True


def _limit_active(value: float, full_value: float) -> bool:
    full_value = abs(float(full_value))
    return full_value > 0.0 and abs(float(value)) > full_value


def _limit_gain(value: float, full_value: float) -> float:
    full_value = abs(float(full_value))
    if full_value <= 0.0 or value <= full_value:
        return 1.0
    return clamp(full_value / max(value, 1.0e-9), 0.0, 1.0)


def _empty_safety_gate_diagnostics() -> Dict[str, Any]:
    return {
        "rl_safety_gate_active": 0,
        "rl_safety_gate_reason": "",
        "rl_safety_gate_speed_limit": 0,
        "rl_safety_gate_roll_limit": 0,
        "rl_safety_gate_pitch_limit": 0,
        "rl_safety_gate_lateral_acc_limit": 0,
        "rl_safety_gate_yaw_rate_limit": 0,
        "rl_safety_gate_nonfinite_obs": 0,
        "rl_safety_gate_action_invalid": 0,
    }


def _mean_abs(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(abs(float(value)) for value in values) / float(len(values))


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(float(value) for value in values) / float(len(values))
