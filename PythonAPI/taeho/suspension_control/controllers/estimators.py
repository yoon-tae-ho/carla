"""Suspension-state estimator dry-run controllers."""

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
)


_EPS = 1.0e-9
_LABELS = ("fl", "fr", "rl", "rr")
_WHEEL_NAMES = ("FL", "FR", "RL", "RR")


@dataclass(frozen=True)
class SkyhookEstimatorDryRunConfig:
    """Dry-run skyhook estimator config.

    The controller computes candidate state-based skyhook signals for analysis
    only. It never applies those proposed scales; the returned command is
    identity on every tick.
    """

    controller_version: str = "skyhook_estimator_dryrun_v1"
    default_dt: float = 0.05
    wheel_count: int = 4

    half_track_m: float = 0.85
    half_wheelbase_m: float = 1.45

    neutral_damper_scale: float = 1.0
    soft_damper_scale: float = 0.80
    hard_damper_scale: float = 1.30
    sprung_velocity_deadband_mps: float = 0.02
    relative_velocity_deadband_mps: float = 0.005
    skyhook_product_deadband: float = 1.0e-4

    require_velocity: bool = True
    require_contact: bool = True
    identity_if_any_wheel_invalid: bool = True

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> "SkyhookEstimatorDryRunConfig":
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


class SkyhookEstimatorDryRunController(SuspensionController):
    """Compute state-based skyhook candidates while returning identity."""

    name = "skyhook_estimator_dryrun"
    requires_suspension_state = True

    def __init__(
        self,
        config: Optional[SkyhookEstimatorDryRunConfig] = None,
    ):
        self.config = config or SkyhookEstimatorDryRunConfig()

    def compute(self, context: ControllerContext) -> ControllerOutput:
        wheel_count = self._wheel_count(context)
        command = SuspensionCommand.identity(wheel_count).validate(
            expected_wheels=wheel_count)

        state = context.suspension_state
        wheels = tuple(getattr(state, "wheels", ()) or ())
        valid_reason = self._invalid_reason(context, wheels)
        state_valid = valid_reason == ""
        corner_velocities, roll_components, pitch_components = (
            self._corner_vertical_velocity_terms(context.state))
        native_springs = tuple(context.native_spring_strength_by_wheel or ())
        native_dampers = tuple(context.native_damper_rate_by_wheel or ())

        diagnostics: Dict[str, Any] = {
            "controller": self.name,
            "controller_version": self.config.controller_version,
            "spring_scale": 1.0,
            "damper_scale": 1.0,
            "skyhook_dryrun_command_identity": 1,
            "suspension_state_valid": int(state_valid),
            "contact_valid_all": int(self._contact_valid_all(wheels)),
            "fallback_mode": "identity" if not state_valid else "dryrun_identity",
            "identity_fallback_this_tick": int(not state_valid),
            "identity_fallback_that_would_have_occurred": int(not state_valid),
            "identity_fallback_reason": valid_reason,
            "compression_convention_validated": int(bool(
                getattr(state, "compression_convention_validated", False))),
            "suspension_state_source": getattr(state, "state_source", ""),
            "suspension_failure_reason": getattr(state, "failure_reason", ""),
            "suspension_wheel_count": len(wheels),
            "skyhook_mean_abs_corner_vz": _mean_abs(corner_velocities),
        }

        for label, index in zip(_LABELS, range(len(_LABELS))):
            wheel = wheels[index] if index < len(wheels) else None
            diagnostics.update(self._wheel_diagnostics(
                label=label,
                index=index,
                wheel=wheel,
                v_sprung=corner_velocities[index],
                v_roll=roll_components[index],
                v_pitch=pitch_components[index],
                native_spring=(
                    native_springs[index]
                    if index < len(native_springs) else ""),
                native_damper=(
                    native_dampers[index]
                    if index < len(native_dampers) else "")))

        if state_valid:
            diagnostics.update(self._candidate_diagnostics(
                wheels=wheels,
                corner_velocities=corner_velocities))
        else:
            diagnostics.update(self._empty_candidate_diagnostics())

        return ControllerOutput(command=command, diagnostics=diagnostics)

    def _wheel_count(self, context: ControllerContext) -> int:
        for source_name in ("native_suspension", "current_suspension"):
            source = getattr(context, source_name, None)
            wheels = getattr(source, "wheels", None)
            if wheels is None:
                continue
            try:
                count = len(wheels)
            except TypeError:
                continue
            if count > 0:
                return count
        return max(1, int(self.config.wheel_count))

    def _invalid_reason(
        self,
        context: ControllerContext,
        wheels: Sequence[Any],
    ) -> str:
        cfg = self.config
        if not bool(context.suspension_state_valid):
            reason = str(getattr(
                context,
                "suspension_state_invalid_reason",
                "") or "")
            if reason:
                return reason
            if context.suspension_state is None:
                return "suspension_state_unavailable"
            failure = str(getattr(
                context.suspension_state,
                "failure_reason",
                "") or "")
            return failure or "suspension_state_invalid"
        expected = int(cfg.wheel_count)
        if len(wheels) != expected:
            return "wheel_count_%d_expected_%d" % (len(wheels), expected)
        if not bool(getattr(context.suspension_state, "state_valid", False)):
            return str(getattr(
                context.suspension_state,
                "failure_reason",
                "") or "state_valid_false")
        if cfg.require_velocity and not bool(getattr(
                context.suspension_state,
                "velocity_valid",
                False)):
            return "state_velocity_valid_false"
        if not cfg.identity_if_any_wheel_invalid:
            return ""
        for index, wheel in enumerate(wheels):
            if not bool(getattr(wheel, "field_valid", False)):
                return "wheel_%d_field_valid_false" % index
            if cfg.require_velocity and not bool(
                    getattr(wheel, "velocity_valid", False)):
                return "wheel_%d_velocity_valid_false" % index
            if cfg.require_contact and not bool(
                    getattr(wheel, "contact_valid", False)):
                return "wheel_%d_contact_valid_false" % index
            if cfg.require_contact and bool(getattr(wheel, "wheel_in_air", False)):
                return "wheel_%d_in_air" % index
            for field_name in (
                    "raw_suspension_offset_m",
                    "suspension_compression_m",
                    "suspension_travel_m",
                    "normalized_travel"):
                if not _is_finite(getattr(wheel, field_name, None)):
                    return "wheel_%d_%s_nonfinite" % (index, field_name)
            if cfg.require_velocity and not _is_finite(
                    getattr(wheel, "suspension_velocity_mps", None)):
                return "wheel_%d_suspension_velocity_mps_nonfinite" % index
        return ""

    def _contact_valid_all(self, wheels: Sequence[Any]) -> bool:
        if not wheels:
            return False
        return all(
            bool(getattr(wheel, "contact_valid", False)) and
            not bool(getattr(wheel, "wheel_in_air", False))
            for wheel in wheels)

    def _wheel_diagnostics(
        self,
        label: str,
        index: int,
        wheel: Any,
        v_sprung: float,
        v_roll: float,
        v_pitch: float,
        native_spring: Any,
        native_damper: Any,
    ) -> Dict[str, Any]:
        diagnostics = {
            "wheel_index_raw_%s" % label: "",
            "wheel_name_canonical_%s" % label: (
                _WHEEL_NAMES[index] if index < len(_WHEEL_NAMES) else ""),
            "raw_suspension_offset_m_%s" % label: "",
            "compression_m_%s" % label: "",
            "suspension_travel_m_%s" % label: "",
            "suspension_velocity_mps_%s" % label: "",
            "normalized_travel_%s" % label: "",
            "contact_valid_%s" % label: "",
            "wheel_in_air_%s" % label: "",
            "field_valid_%s" % label: "",
            "velocity_valid_%s" % label: "",
            "native_spring_strength_%s" % label: native_spring,
            "native_damper_rate_%s" % label: native_damper,
            "v_sprung_%s" % label: v_sprung,
            "v_roll_%s" % label: v_roll,
            "v_pitch_%s" % label: v_pitch,
            "final_spring_scale_%s" % label: 1.0,
            "final_damper_scale_%s" % label: 1.0,
        }
        if wheel is None:
            return diagnostics
        diagnostics.update({
            "wheel_index_raw_%s" % label: getattr(wheel, "wheel_index_raw", index),
            "wheel_name_canonical_%s" % label: getattr(
                wheel,
                "wheel_name_canonical",
                diagnostics["wheel_name_canonical_%s" % label]),
            "raw_suspension_offset_m_%s" % label: _finite_or_empty(
                getattr(wheel, "raw_suspension_offset_m", "")),
            "compression_m_%s" % label: _finite_or_empty(
                getattr(wheel, "suspension_compression_m", "")),
            "suspension_travel_m_%s" % label: _finite_or_empty(
                getattr(wheel, "suspension_travel_m", "")),
            "suspension_velocity_mps_%s" % label: _finite_or_empty(
                getattr(wheel, "suspension_velocity_mps", "")),
            "normalized_travel_%s" % label: _finite_or_empty(
                getattr(wheel, "normalized_travel", "")),
            "contact_valid_%s" % label: int(bool(
                getattr(wheel, "contact_valid", False))),
            "wheel_in_air_%s" % label: int(bool(
                getattr(wheel, "wheel_in_air", False))),
            "field_valid_%s" % label: int(bool(
                getattr(wheel, "field_valid", False))),
            "velocity_valid_%s" % label: int(bool(
                getattr(wheel, "velocity_valid", False))),
        })
        return diagnostics

    def _candidate_diagnostics(
        self,
        wheels: Sequence[Any],
        corner_velocities: Sequence[float],
    ) -> Dict[str, Any]:
        diagnostics: Dict[str, Any] = {}
        for candidate_name, sign in (("A", -1.0), ("B", 1.0)):
            v_rel_values = []
            products = []
            proposed_scales = []
            soft_flags = []
            hard_flags = []
            neutral_flags = []
            for label, wheel, v_sprung in zip(_LABELS, wheels, corner_velocities):
                v_rel = sign * _safe_float(
                    getattr(wheel, "suspension_velocity_mps", 0.0))
                product = _safe_float(v_sprung) * v_rel
                mode = self._candidate_mode(v_sprung, v_rel, product)
                proposed = self._proposed_scale(mode)
                soft = 1 if mode == "soft" else 0
                hard = 1 if mode == "hard" else 0
                neutral = 1 if mode == "neutral" else 0

                diagnostics[
                    "v_rel_extension_candidate_%s_%s" %
                    (candidate_name, label)] = v_rel
                diagnostics[
                    "skyhook_product_candidate_%s_%s" %
                    (candidate_name, label)] = product
                diagnostics[
                    "proposed_damper_scale_candidate_%s_%s" %
                    (candidate_name, label)] = proposed
                diagnostics[
                    "soft_mode_candidate_%s_%s" %
                    (candidate_name, label)] = soft
                diagnostics[
                    "hard_mode_candidate_%s_%s" %
                    (candidate_name, label)] = hard
                diagnostics[
                    "neutral_mode_candidate_%s_%s" %
                    (candidate_name, label)] = neutral

                v_rel_values.append(v_rel)
                products.append(product)
                proposed_scales.append(proposed)
                soft_flags.append(soft)
                hard_flags.append(hard)
                neutral_flags.append(neutral)

            diagnostics[
                "v_rel_extension_candidate_%s" % candidate_name] = _mean(
                    v_rel_values)
            diagnostics[
                "skyhook_product_candidate_%s" % candidate_name] = _mean(
                    products)
            diagnostics[
                "proposed_damper_scale_candidate_%s" % candidate_name] = _mean(
                    proposed_scales)
            diagnostics[
                "soft_mode_ratio_candidate_%s" % candidate_name] = _mean(
                    soft_flags)
            diagnostics[
                "hard_mode_ratio_candidate_%s" % candidate_name] = _mean(
                    hard_flags)
            diagnostics[
                "neutral_mode_ratio_candidate_%s" % candidate_name] = _mean(
                    neutral_flags)
        return diagnostics

    def _empty_candidate_diagnostics(self) -> Dict[str, Any]:
        diagnostics: Dict[str, Any] = {}
        for candidate_name in ("A", "B"):
            diagnostics["v_rel_extension_candidate_%s" % candidate_name] = ""
            diagnostics["skyhook_product_candidate_%s" % candidate_name] = ""
            diagnostics["proposed_damper_scale_candidate_%s" % candidate_name] = ""
            diagnostics["soft_mode_ratio_candidate_%s" % candidate_name] = ""
            diagnostics["hard_mode_ratio_candidate_%s" % candidate_name] = ""
            diagnostics["neutral_mode_ratio_candidate_%s" % candidate_name] = ""
            for label in _LABELS:
                diagnostics[
                    "v_rel_extension_candidate_%s_%s" %
                    (candidate_name, label)] = ""
                diagnostics[
                    "skyhook_product_candidate_%s_%s" %
                    (candidate_name, label)] = ""
                diagnostics[
                    "proposed_damper_scale_candidate_%s_%s" %
                    (candidate_name, label)] = ""
                diagnostics[
                    "soft_mode_candidate_%s_%s" %
                    (candidate_name, label)] = ""
                diagnostics[
                    "hard_mode_candidate_%s_%s" %
                    (candidate_name, label)] = ""
                diagnostics[
                    "neutral_mode_candidate_%s_%s" %
                    (candidate_name, label)] = ""
        return diagnostics

    def _candidate_mode(
        self,
        v_sprung: float,
        v_rel: float,
        product: float,
    ) -> str:
        cfg = self.config
        if (
                abs(_safe_float(v_sprung)) <=
                max(0.0, _safe_float(cfg.sprung_velocity_deadband_mps))):
            return "neutral"
        if (
                abs(_safe_float(v_rel)) <=
                max(0.0, _safe_float(cfg.relative_velocity_deadband_mps))):
            return "neutral"
        if (
                abs(_safe_float(product)) <=
                max(0.0, _safe_float(cfg.skyhook_product_deadband))):
            return "neutral"
        return "hard" if product > 0.0 else "soft"

    def _proposed_scale(self, mode: str) -> float:
        cfg = self.config
        if mode == "hard":
            return _safe_float(cfg.hard_damper_scale, 1.30)
        if mode == "soft":
            return _safe_float(cfg.soft_damper_scale, 0.80)
        return _safe_float(cfg.neutral_damper_scale, 1.0)

    def _corner_vertical_velocity_terms(
        self,
        state: VehicleState,
    ) -> Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[float, ...]]:
        roll_rate = math.radians(_safe_float(getattr(state, "roll_rate", 0.0)))
        pitch_rate = math.radians(_safe_float(getattr(state, "pitch_rate", 0.0)))
        vz = _safe_float(getattr(state, "vz", 0.0))
        corners = self._corner_coordinates()
        roll_components = tuple(roll_rate * y for _, y in corners)
        pitch_components = tuple(-pitch_rate * x for x, _ in corners)
        velocities = tuple(
            vz + v_roll + v_pitch
            for v_roll, v_pitch in zip(roll_components, pitch_components))
        return velocities, roll_components, pitch_components

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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _finite_or_empty(value: Any) -> Any:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return ""
    return result if math.isfinite(result) else ""


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _mean(values: Sequence[float]) -> Any:
    return sum(float(value) for value in values) / float(len(values)) if values else ""


def _mean_abs(values: Sequence[float]) -> float:
    return (
        sum(abs(float(value)) for value in values) / float(len(values))
        if values else 0.0)
