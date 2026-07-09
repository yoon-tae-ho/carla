"""Canonical skyhook damping plus pure modal roll-rate damping."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .base import (
    ControllerContext,
    ControllerOutput,
    SuspensionCommand,
    WheelScale,
)
from .skyhook import (
    SkyhookConfig,
    SkyhookController,
    canonical_skyhook_v3_projection,
)


_EPS = 1.0e-9
_FROZEN_SPRING_SCALE = 1.0
_CORNER_LABELS = ("fl", "fr", "rl", "rr")

SKYHOOK_ROLL_V3_CANONICAL_MODAL_VERSION = (
    "skyhook_roll_v3_canonical_modal_roll_rate")


@dataclass(frozen=True)
class SkyhookRollV3CanonicalModalConfig(SkyhookConfig):
    """Config for canonical skyhook plus modal roll-rate damping only."""

    controller_version: str = SKYHOOK_ROLL_V3_CANONICAL_MODAL_VERSION

    roll_modal_enabled: bool = True
    roll_modal_c_scale: float = 0.20
    roll_modal_distribution: str = "geometry_min_norm"

    use_roll_angle_feedback: bool = False
    use_lateral_accel_gate: bool = False
    use_outer_side_bias: bool = False
    use_front_rear_roll_distribution: bool = False
    enable_yaw_distribution: bool = False
    use_planning_preview: bool = False

    roll_spring_enabled: bool = False
    enable_roll_spring_control: bool = False
    base_spring_scale: float = 1.0
    min_spring_scale: float = 1.0
    max_spring_scale: float = 1.0
    max_spring_delta_per_step: float = 0.0

    # Legacy skyhook_roll knobs are accepted for config compatibility only.
    # They are intentionally not read by the v3 command law.
    roll_damper_enabled: bool = True
    roll_c_scale: float = 0.20
    max_roll_extra_scale: float = 0.08
    roll_softening_override_limit: float = 0.03
    roll_ay_on: float = 1.5
    roll_ay_full: float = 5.0
    roll_rate_on_deg_s: float = 1.5
    roll_rate_full_deg_s: float = 12.0
    roll_angle_on_deg: float = 1.0
    roll_angle_full_deg: float = 5.0
    outer_side_sign_from_ay: float = -1.0
    outer_side_bias: float = 0.10
    inner_side_relief: float = 0.05
    nominal_front_roll_distribution: float = 0.55
    yaw_distribution_mode: str = "log_only"
    yaw_activation_start_g: float = 0.40
    yaw_activation_full_g: float = 0.60
    g: float = 9.81
    max_road_wheel_angle_rad: float = 0.60
    yaw_ref_min_speed: float = 5.0
    yaw_ref_limit_lat_g: float = 0.85
    yaw_front_distribution_kp: float = 0.08
    yaw_front_distribution_ki: float = 0.00
    yaw_error_integral_limit: float = 2.0
    yaw_integral_decay: float = 0.98
    small_yaw_ref: float = 0.03
    min_front_roll_distribution: float = 0.35
    max_front_roll_distribution: float = 0.75
    max_front_distribution_delta_per_step: float = 0.04

    debug: bool = False

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> "SkyhookRollV3CanonicalModalConfig":
        allowed = {item.name for item in fields(cls)}
        incoming = dict(values or {})
        if "half_track" in incoming and "half_track_m" not in incoming:
            incoming["half_track_m"] = incoming["half_track"]
        if "half_wheelbase" in incoming and "half_wheelbase_m" not in incoming:
            incoming["half_wheelbase_m"] = incoming["half_wheelbase"]
        if (
                "relative_velocity_deadband" in incoming and
                "rel_velocity_deadband" not in incoming):
            incoming["rel_velocity_deadband"] = incoming[
                "relative_velocity_deadband"]
        if (
                "corner_velocity_deadband" in incoming and
                "sprung_velocity_deadband" not in incoming):
            incoming["sprung_velocity_deadband"] = incoming[
                "corner_velocity_deadband"]
        if (
                "base_damper_scale" in incoming and
                "neutral_damper_scale" not in incoming):
            incoming["neutral_damper_scale"] = incoming["base_damper_scale"]
        kwargs = {
            key: value
            for key, value in incoming.items()
            if key in allowed
        }
        return cls(**kwargs)


class SkyhookRollV3CanonicalModalController(SkyhookController):
    """Feedback-only damper controller with geometry-only roll modal residual."""

    name = "skyhook_roll_v3"
    requires_suspension_state = True

    def __init__(
        self,
        config: Optional[SkyhookRollV3CanonicalModalConfig] = None,
    ):
        super().__init__(config or SkyhookRollV3CanonicalModalConfig())

    def compute(self, context: ControllerContext) -> ControllerOutput:
        cfg = self.config
        wheel_count = self._wheel_count(context)
        self._ensure_previous_count(wheel_count)

        (
            corner_velocities,
            roll_components,
            pitch_components,
            angular_diagnostics,
        ) = self._corner_vertical_velocity_terms(context.state)
        corner_velocities = _match_length(corner_velocities, wheel_count, 0.0)
        roll_components = _match_length(roll_components, wheel_count, 0.0)
        pitch_components = _match_length(pitch_components, wheel_count, 0.0)
        corners = _match_length(
            self._corner_coordinates(),
            wheel_count,
            (0.0, 0.0))

        state = context.suspension_state
        wheels = tuple(getattr(state, "wheels", ()) or ())
        native_dampers = self._native_damper_rates(context, wheel_count)
        c_sky_values = tuple(
            self._skyhook_damping(native_damper)
            for native_damper in _match_length(native_dampers, wheel_count, 0.0))
        invalid_reason = self._invalid_reason(
            context,
            wheels,
            native_dampers,
            wheel_count)
        state_valid = invalid_reason == ""

        roll_modal = self._roll_modal_info(
            roll_rate_rad_s=_safe_float(
                angular_diagnostics["roll_rate_rad_s"]),
            corners=corners,
            c_sky_values=c_sky_values)
        diagnostics = self._base_diagnostics(
            context=context,
            wheel_count=wheel_count,
            state_valid=state_valid,
            invalid_reason=invalid_reason,
            corner_velocities=corner_velocities,
            angular_diagnostics=angular_diagnostics)
        self._add_v3_scalar_diagnostics(diagnostics, roll_modal)

        if not state_valid:
            self.previous_damper_scales = [1.0 for _ in range(wheel_count)]
            command = SuspensionCommand.identity(wheel_count).validate(
                expected_wheels=wheel_count)
            diagnostics.update({
                "spring_scale": 1.0,
                "damper_scale": 1.0,
                "fallback_mode": "identity",
                "identity_fallback_this_tick": 1,
                "identity_fallback_that_would_have_occurred": 1,
                "identity_fallback_reason": invalid_reason,
            })
            for index in range(wheel_count):
                wheel = wheels[index] if index < len(wheels) else None
                native_damper = (
                    native_dampers[index]
                    if index < len(native_dampers) else "")
                self._add_wheel_diagnostics(
                    diagnostics,
                    index=index,
                    wheel=wheel,
                    v_sprung=corner_velocities[index],
                    v_roll=roll_components[index],
                    v_pitch=pitch_components[index],
                    native_damper=native_damper,
                    final_target=1.0,
                    final_damper=1.0,
                    law_result=None)
                self._add_v3_wheel_fallback_diagnostics(
                    diagnostics,
                    index=index,
                    corner=corners[index],
                    native_damper=native_damper,
                    c_sky=c_sky_values[index],
                    v_s_base=corner_velocities[index],
                    f_roll_modal=0.0)
            return ControllerOutput(command=command, diagnostics=diagnostics)

        damper_scales = []
        law_results = []
        for index in range(wheel_count):
            wheel = wheels[index]
            x_i, y_i = corners[index]
            v_s_base = corner_velocities[index]
            v_rel = (
                _safe_float(cfg.relative_extension_sign, -1.0) *
                _safe_float(getattr(wheel, "suspension_velocity_mps", 0.0)))
            native = max(_safe_float(native_dampers[index], 0.0), _EPS)
            c_sky = self._skyhook_damping(native)
            f_roll_modal = roll_modal["forces"][index]
            law_result = self._target_damper_scale_with_modal(
                x_i=x_i,
                y_i=y_i,
                v_s_base_i=v_s_base,
                v_rel_extension_i=v_rel,
                native_damper_i=native,
                c_sky_i=c_sky,
                f_roll_modal_i=f_roll_modal)
            final_damper = self._rate_limit_damper(
                index,
                law_result["final_target_damper"])
            law_result["final_damper_scale"] = final_damper
            law_result["rate_limited_damper"] = int(
                not math.isclose(
                    final_damper,
                    law_result["final_target_damper"],
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12))
            law_result["rate_limited_damper_scale"] = final_damper
            damper_scales.append(final_damper)
            law_results.append(law_result)
            self._add_wheel_diagnostics(
                diagnostics,
                index=index,
                wheel=wheel,
                v_sprung=v_s_base,
                v_roll=roll_components[index],
                v_pitch=pitch_components[index],
                native_damper=native,
                final_target=law_result["final_target_damper"],
                final_damper=final_damper,
                law_result=law_result)

        command = SuspensionCommand(tuple(
            WheelScale(
                spring_scale=_FROZEN_SPRING_SCALE,
                damper_scale=damper_scale)
            for damper_scale in damper_scales
        )).validate(expected_wheels=wheel_count)

        diagnostics.update({
            "spring_scale": 1.0,
            "damper_scale": _mean(damper_scales),
            "fallback_mode": "active",
            "identity_fallback_this_tick": 0,
            "identity_fallback_that_would_have_occurred": 0,
            "identity_fallback_reason": "",
            "soft_mode_ratio": _mean(_diag_wheel_values(
                diagnostics,
                "soft_mode",
                wheel_count)),
            "hard_mode_ratio": _mean(_diag_wheel_values(
                diagnostics,
                "hard_mode",
                wheel_count)),
            "neutral_mode_ratio": _mean(_diag_wheel_values(
                diagnostics,
                "neutral_mode",
                wheel_count)),
        })
        self._add_v3_scalar_diagnostics(diagnostics, roll_modal)
        return ControllerOutput(command=command, diagnostics=diagnostics)

    def _skyhook_damping(self, native_damper: float) -> float:
        native = max(_safe_float(native_damper, 0.0), _EPS)
        return max(0.0, _safe_float(self.config.skyhook_c_scale, 1.0)) * native

    def _roll_modal_info(
        self,
        roll_rate_rad_s: float,
        corners: Sequence[Tuple[float, float]],
        c_sky_values: Sequence[float],
    ) -> Dict[str, Any]:
        denom = sum(_safe_float(y_i) ** 2 for _, y_i in corners)
        c_phi = (
            max(0.0, _safe_float(self.config.roll_modal_c_scale, 0.20)) *
            sum(
                max(0.0, _safe_float(c_sky)) * (_safe_float(y_i) ** 2)
                for (_, y_i), c_sky in zip(corners, c_sky_values)))
        q_roll_des = -c_phi * _safe_float(roll_rate_rad_s)
        enabled = bool(getattr(self.config, "roll_modal_enabled", True))
        valid = int(enabled and denom > _EPS)
        if valid:
            forces = tuple(q_roll_des * _safe_float(y_i) / denom
                           for _, y_i in corners)
        else:
            forces = tuple(0.0 for _ in corners)
        return {
            "enabled": int(enabled),
            "distribution": str(getattr(
                self.config,
                "roll_modal_distribution",
                "geometry_min_norm")),
            "roll_rate_rad_s": _safe_float(roll_rate_rad_s),
            "roll_rate_deg_s": math.degrees(_safe_float(roll_rate_rad_s)),
            "C_phi": c_phi,
            "Q_roll_des": q_roll_des,
            "denom": denom,
            "valid": valid,
            "forces": forces,
            "heave_sum": sum(forces),
            "pitch_sum": sum(
                _safe_float(x_i) * force
                for (x_i, _), force in zip(corners, forces)),
            "roll_moment": sum(
                _safe_float(y_i) * force
                for (_, y_i), force in zip(corners, forces)),
        }

    def _target_damper_scale_with_modal(
        self,
        x_i: float,
        y_i: float,
        v_s_base_i: float,
        v_rel_extension_i: float,
        native_damper_i: float,
        c_sky_i: float,
        f_roll_modal_i: float,
    ) -> Dict[str, Any]:
        cfg = self.config
        native = max(_safe_float(native_damper_i, 0.0), _EPS)
        c_sky = max(0.0, _safe_float(c_sky_i, 0.0))
        f_sky_ideal = -c_sky * _safe_float(v_s_base_i)
        f_total_ideal = f_sky_ideal + _safe_float(f_roll_modal_i)
        c_ref = max(c_sky, max(_safe_float(cfg.projection_eps), _EPS))
        v_eff_total = -f_total_ideal / c_ref

        base_result = canonical_skyhook_v3_projection(
            v_eff_i=v_s_base_i,
            v_rel_extension_i=v_rel_extension_i,
            C_native_i=native,
            C_sky_i=c_sky,
            config=cfg)
        total_result = canonical_skyhook_v3_projection(
            v_eff_i=v_eff_total,
            v_rel_extension_i=v_rel_extension_i,
            C_native_i=native,
            C_sky_i=c_sky,
            config=cfg)

        final_target = total_result["final_target_damper"]
        return {
            "x": _safe_float(x_i),
            "y": _safe_float(y_i),
            "C_native": native,
            "C_sky": c_sky,
            "v_s_base": _safe_float(v_s_base_i),
            "v_rel_extension_mps": _safe_float(v_rel_extension_i),
            "F_sky_ideal": f_sky_ideal,
            "F_roll_ideal": _safe_float(f_roll_modal_i),
            "F_roll_modal": _safe_float(f_roll_modal_i),
            "F_total_ideal": f_total_ideal,
            "v_eff_total": v_eff_total,
            "base_target_damper": base_result["final_target_damper"],
            "skyhook_only_target_damper": base_result["final_target_damper"],
            "total_raw_target_damper": total_result["raw_target_damper"],
            "total_target_branch": total_result["target_branch"],
            "total_product": total_result["skyhook_product"],
            "total_required_scale_unclipped": total_result[
                "required_scale_unclipped"],
            "total_target_damper": final_target,
            "target_branch": total_result["target_branch"],
            "skyhook_product": total_result["skyhook_product"],
            "abs_v_sprung": total_result["abs_v_sprung"],
            "abs_v_rel": total_result["abs_v_rel"],
            "required_scale_unclipped": total_result[
                "required_scale_unclipped"],
            "raw_target_damper": total_result["raw_target_damper"],
            "target_after_minmax_clamp": total_result[
                "target_after_minmax_clamp"],
            "C_required": total_result["C_required"],
            "semi_active_feasible": total_result["semi_active_feasible"],
            "final_target_damper": final_target,
            "final_spring_scale": 1.0,
            "final_damper_scale": final_target,
            "rate_limited_damper": 0,
            "rate_limited_damper_scale": final_target,
            "clamped_damper": total_result["clamped_damper"],
            "soft_mode": total_result["soft_mode"],
            "hard_mode": total_result["hard_mode"],
            "neutral_mode": total_result["neutral_mode"],
        }

    def _add_v3_scalar_diagnostics(
        self,
        diagnostics: Dict[str, Any],
        roll_modal: Mapping[str, Any],
    ) -> None:
        diagnostics.update({
            "controller": self.name,
            "controller_name": self.name,
            "controller_version": self.config.controller_version,
            "skyhook_roll_v3_mode": (
                "canonical_modal_roll_rate"
                if int(roll_modal["enabled"]) else
                "canonical_skyhook_only"),
            "roll_modal_enabled": roll_modal["enabled"],
            "roll_modal_c_scale": _safe_float(
                self.config.roll_modal_c_scale,
                0.20),
            "roll_modal_distribution": roll_modal["distribution"],
            "roll_rate_rad_s": roll_modal["roll_rate_rad_s"],
            "roll_rate_deg_s": roll_modal["roll_rate_deg_s"],
            "C_phi": roll_modal["C_phi"],
            "Q_roll_des": roll_modal["Q_roll_des"],
            "roll_distribution_denom": roll_modal["denom"],
            "roll_modal_valid": roll_modal["valid"],
            "roll_residual_heave_sum": roll_modal["heave_sum"],
            "roll_residual_pitch_sum": roll_modal["pitch_sum"],
            "roll_residual_roll_moment": roll_modal["roll_moment"],
            "roll_angle_used_in_command": 0,
            "local_ay_used_in_command": 0,
            "yaw_used_in_command": 0,
            "planning_preview_used_in_command": 0,
            "spring_used_in_command": 0,
        })

    def _add_v3_wheel_fallback_diagnostics(
        self,
        diagnostics: Dict[str, Any],
        index: int,
        corner: Tuple[float, float],
        native_damper: Any,
        c_sky: float,
        v_s_base: float,
        f_roll_modal: float,
    ) -> None:
        label = _label(index)
        x_i, y_i = corner
        native = _safe_float(native_damper, 0.0)
        diagnostics.update({
            "x_%s" % label: _safe_float(x_i),
            "y_%s" % label: _safe_float(y_i),
            "C_native_%s" % label: native if native > 0.0 else "",
            "C_sky_%s" % label: _safe_float(c_sky),
            "v_s_base_%s" % label: _safe_float(v_s_base),
            "v_rel_extension_mps_%s" % label: "",
            "F_sky_ideal_%s" % label: "",
            "F_roll_modal_%s" % label: _safe_float(f_roll_modal),
            "F_total_ideal_%s" % label: "",
            "v_eff_total_%s" % label: "",
            "base_target_damper_%s" % label: 1.0,
            "total_raw_target_damper_%s" % label: 1.0,
            "total_target_branch_%s" % label: "identity_fallback",
            "total_product_%s" % label: "",
            "total_required_scale_unclipped_%s" % label: "",
            "total_target_damper_%s" % label: 1.0,
            "final_spring_scale_%s" % label: 1.0,
            "final_damper_scale_%s" % label: 1.0,
            "rate_limited_damper_%s" % label: 0,
            "rate_limited_damper_scale_%s" % label: 1.0,
        })


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _mean(values: Sequence[float]) -> Any:
    if not values:
        return ""
    return sum(_safe_float(value) for value in values) / float(len(values))


def _match_length(
    values: Sequence[Any],
    target_length: int,
    fill_value: Any,
) -> Tuple[Any, ...]:
    values = tuple(values or ())
    if len(values) == target_length:
        return values
    if len(values) > target_length:
        return values[:target_length]
    return values + tuple(fill_value for _ in range(target_length - len(values)))


def _label(index: int) -> str:
    if index < len(_CORNER_LABELS):
        return _CORNER_LABELS[index]
    return "w%d" % index


def _diag_wheel_values(
    diagnostics: Mapping[str, Any],
    prefix: str,
    wheel_count: int,
) -> Tuple[float, ...]:
    values = []
    for index in range(wheel_count):
        value = diagnostics.get("%s_%s" % (prefix, _label(index)))
        if _is_finite(value):
            values.append(float(value))
    return tuple(values)
