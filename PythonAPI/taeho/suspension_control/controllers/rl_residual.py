"""Residual-RL suspension controller over PID or skyhook baselines."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .base import (
    ControllerContext,
    ControllerOutput,
    SuspensionCommand,
    SuspensionController,
)
from .pid import FeedbackPIDConfig, FeedbackPIDController
from .skyhook import SkyhookConfig, SkyhookController
from ..rl.normalizer import FixedScaleNormalizer
from ..rl.observations import ObservationBuilder, PLANNING_FEATURES
from ..rl.policy import PolicyAdapter
from ..rl.action_projection import (
    ResidualActionProjector,
    ResidualActionProjectorConfig,
)


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
    max_abs_yaw_rate_for_full_policy: float = 60.0
    invalid_observation_fallback: bool = True
    invalid_policy_fallback: bool = True

    observation_clip: float = 5.0
    rl_residual_mode: str = "learned_policy"
    rl_action_scale: float = 1.0
    rl_residual_gain: float = 1.0
    scripted_residual_kind: str = ""
    scripted_residual_value: float = 0.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ResidualRLConfig":
        allowed = {item.name for item in fields(cls)}
        kwargs = {key: value for key, value in values.items() if key in allowed}
        return cls(**kwargs)


def _projector_config_from_residual_config(
    config: ResidualRLConfig,
) -> ResidualActionProjectorConfig:
    return ResidualActionProjectorConfig(
        wheel_count=config.wheel_count,
        spring_frozen=config.spring_frozen,
        base_spring_scale=config.base_spring_scale,
        min_spring_scale=config.min_spring_scale,
        max_spring_scale=config.max_spring_scale,
        min_damper_scale=config.min_damper_scale,
        max_damper_scale=config.max_damper_scale,
        max_damper_residual_scale=config.max_damper_residual_scale,
        max_damper_delta_per_step=config.max_damper_delta_per_step,
        min_speed_for_policy=config.min_speed_for_policy,
        max_abs_roll_for_full_policy=config.max_abs_roll_for_full_policy,
        max_abs_pitch_for_full_policy=config.max_abs_pitch_for_full_policy,
        max_abs_lateral_acc_for_full_policy=(
            config.max_abs_lateral_acc_for_full_policy),
        max_abs_yaw_rate_for_full_policy=config.max_abs_yaw_rate_for_full_policy)


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
        self.projector = ResidualActionProjector(
            _projector_config_from_residual_config(self.config))
        self.previous_action = [0.0 for _ in range(self.config.wheel_count)]
        self.previous_damper_scales: Optional[List[float]] = None

    def reset(self, native_suspension: Any = None) -> None:
        self.baseline_controller.reset(native_suspension)
        self.projector.reset()
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
        observation_valid = bool(obs_diag.get("observation_valid", 0))
        safety_gain, safety_diag = self.projector.safety_gate(
            context.state,
            observation_valid=observation_valid)

        if self._uses_scripted_residual():
            projection = self.projector.project(
                baseline_command,
                [0.0 for _ in range(cfg.wheel_count)],
                context.state,
                observation_valid=observation_valid,
                action_scale=1.0,
                residual_mode="scripted",
                scripted_residual_kind=self._scripted_residual_kind(),
                scripted_residual_value=self._scripted_residual_value(),
                raw_residual_override=self._scripted_residuals())
            diagnostics = self._diagnostics(
                baseline_output,
                obs_diag,
                projection.sanitized_action,
                projection.final_residual_damper_per_wheel,
                projection.baseline_damper_per_wheel,
                projection.final_damper_per_wheel,
                projection.safety_gain,
                fallback_reason=projection.fallback_reason,
                safety_diag=projection.diagnostics)
            self.previous_action = list(projection.sanitized_action)
            self.previous_damper_scales = list(projection.final_damper_per_wheel)
            return ControllerOutput(command=projection.command, diagnostics=diagnostics)

        if not self.policy.is_available and not cfg.allow_untrained_policy:
            return self._baseline_fallback(
                baseline_output,
                obs_diag,
                [0.0 for _ in range(cfg.wheel_count)],
                safety_gain,
                "policy_unavailable",
                safety_diag)

        if not observation_valid:
            if cfg.invalid_observation_fallback:
                safety_gain, safety_diag = self.projector.safety_gate(
                    context.state,
                    observation_valid=False)
                return self._baseline_fallback(
                    baseline_output,
                    obs_diag,
                    [0.0 for _ in range(cfg.wheel_count)],
                    safety_gain,
                    "invalid_observation",
                    safety_diag)

        try:
            action = self.policy.predict(
                observation,
                deterministic=cfg.deterministic_policy)
        except Exception:
            if cfg.invalid_policy_fallback:
                safety_gain, safety_diag = self.projector.safety_gate(
                    context.state,
                    observation_valid=observation_valid,
                    action_invalid=True)
                return self._baseline_fallback(
                    baseline_output,
                    obs_diag,
                    [0.0 for _ in range(cfg.wheel_count)],
                    safety_gain,
                    "policy_predict_failed",
                    safety_diag)
            raise

        action_values, action_error = self.projector.sanitize_action(action)
        if action_values is None:
            if cfg.invalid_policy_fallback:
                safety_gain, safety_diag = self.projector.safety_gate(
                    context.state,
                    observation_valid=observation_valid,
                    action_invalid=True)
                return self._baseline_fallback(
                    baseline_output,
                    obs_diag,
                    [0.0 for _ in range(cfg.wheel_count)],
                    safety_gain,
                    action_error or "invalid_policy_action",
                    safety_diag)
            raise ValueError(action_error or "invalid policy action")

        projection = self.projector.project(
            baseline_command,
            action_values,
            context.state,
            observation_valid=observation_valid,
            action_scale=self._action_scale(),
            residual_mode="learned_policy")
        diagnostics = self._diagnostics(
            baseline_output,
            obs_diag,
            projection.sanitized_action,
            projection.final_residual_damper_per_wheel,
            projection.baseline_damper_per_wheel,
            projection.final_damper_per_wheel,
            projection.safety_gain,
            fallback_reason=projection.fallback_reason,
            safety_diag=projection.diagnostics)
        if projection.fallback_reason:
            self.previous_action = [0.0 for _ in range(cfg.wheel_count)]
        else:
            self.previous_action = list(projection.sanitized_action)
        self.previous_damper_scales = list(projection.final_damper_per_wheel)
        return ControllerOutput(command=projection.command, diagnostics=diagnostics)

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
        safety_diag: Optional[Mapping[str, Any]] = None,
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
            fallback_reason=fallback_reason,
            safety_diag=safety_diag)
        self.previous_action = [0.0 for _ in range(self.config.wheel_count)]
        self.previous_damper_scales = list(baseline_dampers)
        self.projector.previous_damper_scales = tuple(baseline_dampers)
        return ControllerOutput(command=command, diagnostics=diagnostics)

    def _zero_residual_output(
        self,
        baseline_output: ControllerOutput,
        obs_diag: Mapping[str, Any],
        action: Sequence[float],
        safety_gain: float,
        safety_diag: Optional[Mapping[str, Any]] = None,
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
            fallback_reason="",
            safety_diag=safety_diag)
        self.previous_action = list(action_values)
        self.previous_damper_scales = list(baseline_dampers)
        self.projector.previous_damper_scales = tuple(baseline_dampers)
        return ControllerOutput(command=command, diagnostics=diagnostics)

    def _safety_gain(self, state: Any, obs_diag: Mapping[str, Any]) -> float:
        gain, _ = self.projector.safety_gate(
            state,
            observation_valid=bool(obs_diag.get("observation_valid", 0)))
        return gain

    def _uses_scripted_residual(self) -> bool:
        return str(self.config.rl_residual_mode).strip().lower() == "scripted"

    def _action_scale(self) -> float:
        try:
            explicit = float(self.config.rl_action_scale)
        except (TypeError, ValueError):
            explicit = 1.0
        try:
            gain = float(self.config.rl_residual_gain)
        except (TypeError, ValueError):
            gain = 1.0
        value = explicit if abs(explicit - 1.0) > 1.0e-12 else gain
        return max(0.0, value)

    def _scripted_residual_kind(self) -> str:
        kind = str(self.config.scripted_residual_kind).strip()
        if kind:
            return kind
        value = self._scripted_residual_value()
        if abs(value) <= 1.0e-12:
            return "zero"
        sign = "p" if value > 0.0 else "m"
        magnitude = ("%.2f" % abs(value)).replace(".", "p")
        return "const_%s%s" % (sign, magnitude)

    def _scripted_residual_value(self) -> float:
        try:
            return float(self.config.scripted_residual_value)
        except (TypeError, ValueError):
            return 0.0

    def _scripted_residuals(self) -> List[float]:
        return [self._scripted_residual_value() for _ in range(self.config.wheel_count)]

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
        safety_diag: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        diagnostics: Dict[str, Any] = dict(baseline_output.diagnostics)
        diagnostics.update(_empty_safety_gate_diagnostics())
        diagnostics.update(dict(safety_diag or {}))
        diagnostics.update({
            "controller": self.name,
            "rl_baseline": self.config.baseline,
            "rl_residual_mode": (
                "scripted" if self._uses_scripted_residual() else "learned_policy"),
            "rl_action_scale": self._action_scale(),
            "rl_residual_gain": self._action_scale(),
            "rl_scripted_residual_kind": (
                self._scripted_residual_kind()
                if self._uses_scripted_residual() else ""),
            "rl_scripted_residual_value": (
                self._scripted_residual_value()
                if self._uses_scripted_residual() else ""),
            "rl_policy_available": int(self.policy.is_available),
            "rl_policy_status": self.policy.status,
            "rl_policy_builtin_id": getattr(self.policy, "builtin_id", ""),
            "rl_policy_alias_deprecated": int(
                getattr(self.policy, "alias_deprecated", False)),
            "rl_observation_valid": int(obs_diag.get("observation_valid", 0)),
            "rl_safety_gain": safety_gain,
            "rl_fallback_reason": fallback_reason,
            "rl_planning_available": obs_diag.get("planning_available", 0.0),
            "rl_observation_size": obs_diag.get("observation_size", 0),
            "rl_observation_clip_count": obs_diag.get(
                "observation_clip_count",
                0),
            "rl_mean_action": _mean(action),
            "rl_mean_abs_action": _mean_abs(action),
            "rl_mean_residual_damper": _mean(residuals),
            "rl_mean_abs_residual_damper": _mean_abs(residuals),
            "rl_mean_final_residual_damper": _mean(residuals),
            "rl_mean_abs_final_residual_damper": _mean_abs(residuals),
        })
        for name in PLANNING_FEATURES:
            diagnostics[name] = obs_diag.get(name, 0.0)
        for label, value in zip(WHEEL_LABELS, action):
            diagnostics["rl_action_%s" % label] = value
        for label, value in zip(WHEEL_LABELS, action):
            raw_residual = float(value) * max(
                0.0,
                float(self.config.max_damper_residual_scale))
            diagnostics.setdefault(
                "rl_raw_residual_damper_%s" % label,
                raw_residual)
            diagnostics.setdefault(
                "rl_safety_scaled_residual_damper_%s" % label,
                raw_residual * float(safety_gain))
        for label, value in zip(WHEEL_LABELS, residuals):
            diagnostics["rl_residual_damper_%s" % label] = value
            diagnostics.setdefault(
                "rl_rate_limited_residual_damper_%s" % label,
                value)
            diagnostics.setdefault(
                "rl_final_residual_damper_%s" % label,
                value)
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
