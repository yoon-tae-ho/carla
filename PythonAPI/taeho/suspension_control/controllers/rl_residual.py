"""Residual-RL suspension controller over PID or skyhook baselines."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .base import (
    ControllerContext,
    ControllerOutput,
    SuspensionCommand,
    SuspensionController,
    WheelScale,
    clamp,
)
from .pid import FeedbackPIDConfig, FeedbackPIDController
from .skyhook import SkyhookConfig, SkyhookController
from ..rl.normalizer import FixedScaleNormalizer
from ..rl.observations import ObservationBuilder, PLANNING_FEATURES
from ..rl.policy import PolicyAdapter


WHEEL_LABELS = ("fl", "fr", "rl", "rr")


@dataclass(frozen=True)
class ResidualRLConfig:
    default_dt: float = 0.05
    wheel_count: int = 4

    baseline: str = "skyhook"
    policy_path: str = ""
    normalizer_path: str = ""
    allow_untrained_policy: bool = False
    deterministic_policy: bool = True

    spring_frozen: bool = True
    base_spring_scale: float = 1.0
    min_spring_scale: float = 0.95
    max_spring_scale: float = 1.05
    max_spring_residual_scale: float = 0.0

    min_damper_scale: float = 0.80
    max_damper_scale: float = 1.30
    max_damper_residual_scale: float = 0.08
    max_damper_delta_per_step: float = 0.04

    min_speed_for_policy: float = 0.5
    max_abs_roll_for_full_policy: float = 6.0
    max_abs_pitch_for_full_policy: float = 6.0
    max_abs_lateral_acc_for_full_policy: float = 7.0
    invalid_observation_fallback: bool = True
    invalid_policy_fallback: bool = True

    observation_clip: float = 5.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ResidualRLConfig":
        allowed = {item.name for item in fields(cls)}
        kwargs = {key: value for key, value in values.items() if key in allowed}
        return cls(**kwargs)


class ResidualRLController(SuspensionController):
    """Bounded damping residual over an existing stabilizing baseline."""

    name = "rl_residual"

    def __init__(
        self,
        config: Optional[ResidualRLConfig] = None,
        baseline_controller: Optional[SuspensionController] = None,
        policy: Optional[PolicyAdapter] = None,
        observation_builder: Optional[ObservationBuilder] = None,
    ):
        self.config = config or ResidualRLConfig()
        self.baseline_controller = (
            baseline_controller or self._make_baseline_controller())
        self.policy = policy or PolicyAdapter(
            policy_path=self.config.policy_path,
            action_dim=self.config.wheel_count,
            allow_dummy_zero=self.config.allow_untrained_policy)
        normalizer = FixedScaleNormalizer(
            normalizer_path=self.config.normalizer_path,
            clip=self.config.observation_clip)
        self.observation_builder = observation_builder or ObservationBuilder(
            normalizer=normalizer,
            wheel_count=self.config.wheel_count,
            observation_clip=self.config.observation_clip)
        self.previous_action = [0.0 for _ in range(self.config.wheel_count)]
        self.previous_damper_scales: Optional[List[float]] = None

    def reset(self, native_suspension: Any = None) -> None:
        self.baseline_controller.reset(native_suspension)
        self.previous_action = [0.0 for _ in range(self.config.wheel_count)]
        self.previous_damper_scales = None

    def compute(self, context: ControllerContext) -> ControllerOutput:
        cfg = self.config
        baseline_output = self.baseline_controller.compute(context)
        baseline_command = baseline_output.command.validate(
            expected_wheels=cfg.wheel_count)
        observation, obs_diag = self.observation_builder.build(
            context,
            baseline_output,
            self.previous_action,
            self.previous_damper_scales)

        if not self.policy.is_available and not cfg.allow_untrained_policy:
            return self._baseline_fallback(
                baseline_output,
                obs_diag,
                [0.0 for _ in range(cfg.wheel_count)],
                0.0,
                "policy_unavailable")

        if not obs_diag.get("observation_valid", 0):
            if cfg.invalid_observation_fallback:
                return self._baseline_fallback(
                    baseline_output,
                    obs_diag,
                    [0.0 for _ in range(cfg.wheel_count)],
                    0.0,
                    "invalid_observation")

        try:
            action = self.policy.predict(
                observation,
                deterministic=cfg.deterministic_policy)
        except Exception:
            if cfg.invalid_policy_fallback:
                return self._baseline_fallback(
                    baseline_output,
                    obs_diag,
                    [0.0 for _ in range(cfg.wheel_count)],
                    0.0,
                    "policy_predict_failed")
            raise

        action_values, action_error = self._sanitize_action(action)
        if action_values is None:
            if cfg.invalid_policy_fallback:
                return self._baseline_fallback(
                    baseline_output,
                    obs_diag,
                    [0.0 for _ in range(cfg.wheel_count)],
                    0.0,
                    action_error or "invalid_policy_action")
            raise ValueError(action_error or "invalid policy action")

        safety_gain = self._safety_gain(context.state, obs_diag)
        if safety_gain <= 0.0:
            return self._baseline_fallback(
                baseline_output,
                obs_diag,
                action_values,
                safety_gain,
                "safety_gate_zero")

        if all(abs(value) <= 1.0e-12 for value in action_values):
            return self._baseline_fallback(
                baseline_output,
                obs_diag,
                action_values,
                safety_gain,
                "zero_residual")

        command, residuals = self._project_residual(
            baseline_command,
            action_values,
            safety_gain)
        diagnostics = self._diagnostics(
            baseline_output,
            obs_diag,
            action_values,
            residuals,
            _damper_scales(baseline_command),
            _damper_scales(command),
            safety_gain,
            fallback_reason="")
        self.previous_action = list(action_values)
        self.previous_damper_scales = _damper_scales(command)
        return ControllerOutput(command=command, diagnostics=diagnostics)

    def _make_baseline_controller(self) -> SuspensionController:
        baseline = str(self.config.baseline).strip().lower()
        if baseline == "skyhook":
            return SkyhookController(SkyhookConfig(
                default_dt=self.config.default_dt,
                wheel_count=self.config.wheel_count,
                min_damper_scale=self.config.min_damper_scale,
                max_damper_scale=self.config.max_damper_scale,
                max_damper_delta_per_step=self.config.max_damper_delta_per_step))
        if baseline == "pid":
            return FeedbackPIDController(FeedbackPIDConfig(
                default_dt=self.config.default_dt,
                wheel_count=self.config.wheel_count,
                min_spring_scale=self.config.min_spring_scale,
                max_spring_scale=self.config.max_spring_scale,
                min_damper_scale=self.config.min_damper_scale,
                max_damper_scale=self.config.max_damper_scale,
                max_damper_delta_per_step=self.config.max_damper_delta_per_step))
        raise ValueError("unknown residual RL baseline %s" % self.config.baseline)

    def _baseline_fallback(
        self,
        baseline_output: ControllerOutput,
        obs_diag: Mapping[str, Any],
        action: Sequence[float],
        safety_gain: float,
        fallback_reason: str,
    ) -> ControllerOutput:
        command = baseline_output.command.validate(
            expected_wheels=self.config.wheel_count)
        baseline_dampers = _damper_scales(command)
        zero_residuals = [0.0 for _ in range(self.config.wheel_count)]
        action_values = _match_length(action, self.config.wheel_count, 0.0)
        diagnostics = self._diagnostics(
            baseline_output,
            obs_diag,
            action_values,
            zero_residuals,
            baseline_dampers,
            baseline_dampers,
            safety_gain,
            fallback_reason=fallback_reason)
        self.previous_action = [0.0 for _ in range(self.config.wheel_count)]
        self.previous_damper_scales = list(baseline_dampers)
        return ControllerOutput(command=command, diagnostics=diagnostics)

    def _project_residual(
        self,
        baseline_command: SuspensionCommand,
        action: Sequence[float],
        safety_gain: float,
    ) -> Tuple[SuspensionCommand, List[float]]:
        cfg = self.config
        baseline_dampers = _damper_scales(baseline_command)
        previous = (
            _match_length(self.previous_damper_scales, cfg.wheel_count, 1.0)
            if self.previous_damper_scales is not None
            else list(baseline_dampers))
        wheels = []
        residuals = []
        for index, (wheel, action_value) in enumerate(
                zip(baseline_command.wheels, action)):
            raw_residual = (
                float(action_value) *
                max(0.0, cfg.max_damper_residual_scale) *
                safety_gain)
            desired = wheel.damper_scale + raw_residual
            max_delta = max(0.0, cfg.max_damper_delta_per_step)
            if max_delta > 0.0:
                desired = clamp(
                    desired,
                    previous[index] - max_delta,
                    previous[index] + max_delta)
            damper = clamp(
                desired,
                cfg.min_damper_scale,
                cfg.max_damper_scale)
            if cfg.spring_frozen:
                spring = cfg.base_spring_scale
            else:
                spring = wheel.spring_scale
            spring = clamp(spring, cfg.min_spring_scale, cfg.max_spring_scale)
            wheels.append(WheelScale(spring_scale=spring, damper_scale=damper))
            residuals.append(damper - wheel.damper_scale)
        command = SuspensionCommand(tuple(wheels)).validate(
            expected_wheels=cfg.wheel_count)
        return command, residuals

    def _sanitize_action(
        self,
        action: Sequence[float],
    ) -> Tuple[Optional[List[float]], str]:
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
        return result, ""

    def _safety_gain(self, state: Any, obs_diag: Mapping[str, Any]) -> float:
        del obs_diag
        cfg = self.config
        speed = _finite_or_zero(getattr(state, "speed", 0.0))
        if speed < max(0.0, cfg.min_speed_for_policy):
            return 0.0
        roll = abs(_finite_or_zero(getattr(state, "roll", 0.0)))
        pitch = abs(_finite_or_zero(getattr(state, "pitch", 0.0)))
        lateral_acc = abs(_finite_or_zero(getattr(state, "local_ay", 0.0)))
        gains = (
            _limit_gain(roll, cfg.max_abs_roll_for_full_policy),
            _limit_gain(pitch, cfg.max_abs_pitch_for_full_policy),
            _limit_gain(lateral_acc, cfg.max_abs_lateral_acc_for_full_policy),
        )
        return clamp(min(gains), 0.0, 1.0)

    def _diagnostics(
        self,
        baseline_output: ControllerOutput,
        obs_diag: Mapping[str, Any],
        action: Sequence[float],
        residuals: Sequence[float],
        baseline_dampers: Sequence[float],
        final_dampers: Sequence[float],
        safety_gain: float,
        fallback_reason: str,
    ) -> Dict[str, Any]:
        diagnostics: Dict[str, Any] = dict(baseline_output.diagnostics)
        diagnostics.update({
            "controller": self.name,
            "rl_baseline": self.config.baseline,
            "rl_policy_available": int(self.policy.is_available),
            "rl_policy_status": self.policy.status,
            "rl_observation_valid": int(obs_diag.get("observation_valid", 0)),
            "rl_safety_gain": safety_gain,
            "rl_fallback_reason": fallback_reason,
            "rl_planning_available": obs_diag.get("planning_available", 0.0),
            "rl_observation_size": obs_diag.get("observation_size", 0),
            "rl_observation_clip_count": obs_diag.get(
                "observation_clip_count",
                0),
            "rl_mean_abs_action": _mean_abs(action),
            "rl_mean_abs_residual_damper": _mean_abs(residuals),
        })
        for name in PLANNING_FEATURES:
            diagnostics[name] = obs_diag.get(name, 0.0)
        for label, value in zip(WHEEL_LABELS, action):
            diagnostics["rl_action_%s" % label] = value
        for label, value in zip(WHEEL_LABELS, residuals):
            diagnostics["rl_residual_damper_%s" % label] = value
        for label, value in zip(WHEEL_LABELS, baseline_dampers):
            diagnostics["rl_baseline_damper_%s" % label] = value
        for label, value in zip(WHEEL_LABELS, final_dampers):
            diagnostics["rl_final_damper_%s" % label] = value
        return diagnostics


def _damper_scales(command: SuspensionCommand) -> List[float]:
    return [float(wheel.damper_scale) for wheel in command.wheels]


def _match_length(
    values: Optional[Sequence[float]],
    count: int,
    fill: float,
) -> List[float]:
    result = list(values or [])
    result = [float(value) for value in result[:count]]
    while len(result) < count:
        result.append(fill)
    return result


def _finite_or_zero(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _limit_gain(value: float, full_value: float) -> float:
    full_value = abs(float(full_value))
    if full_value <= 0.0 or value <= full_value:
        return 1.0
    return clamp(full_value / max(value, 1.0e-9), 0.0, 1.0)


def _mean_abs(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(abs(float(value)) for value in values) / float(len(values))
