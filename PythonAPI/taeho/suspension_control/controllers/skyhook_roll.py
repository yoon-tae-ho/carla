"""State-strict skyhook damping with a conservative roll damping layer."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .base import (
    ControllerContext,
    ControllerOutput,
    SuspensionCommand,
    WheelScale,
    clamp,
)
from .skyhook import SkyhookConfig, SkyhookController


_EPS = 1.0e-9
_FROZEN_SPRING_SCALE = 1.0
_CORNER_LABELS = ("fl", "fr", "rl", "rr")


@dataclass(frozen=True)
class SkyhookRollConfig(SkyhookConfig):
    """Configuration for ``skyhook_roll_v2_state_damper_only``.

    The controller inherits the state-strict skyhook contract and adds only a
    desired roll damping force before semi-active projection. Springs stay at
    identity. Step 07 may log a yaw distribution proposal, but it is not
    applied to the suspension command.
    """

    controller_version: str = "skyhook_roll_v2_state_damper_only"

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

    # Compatibility knobs accepted from older configs. They are not applied.
    roll_spring_enabled: bool = False
    enable_roll_spring_control: bool = False
    base_spring_scale: float = 1.0
    min_spring_scale: float = 1.0
    max_spring_scale: float = 1.0
    max_spring_delta_per_step: float = 0.0

    enable_yaw_distribution: bool = False
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
    def from_mapping(cls, values: Mapping[str, Any]) -> "SkyhookRollConfig":
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


class SkyhookRollController(SkyhookController):
    """State-strict skyhook plus conservative semi-active roll damping."""

    name = "skyhook_roll"
    requires_suspension_state = True

    def __init__(self, config: Optional[SkyhookRollConfig] = None):
        super().__init__(config or SkyhookRollConfig())

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

        state = context.suspension_state
        wheels = tuple(getattr(state, "wheels", ()) or ())
        native_dampers = self._native_damper_rates(context, wheel_count)
        invalid_reason = self._invalid_reason(
            context,
            wheels,
            native_dampers,
            wheel_count)
        state_valid = invalid_reason == ""

        roll_info = self._roll_gate_and_weights(
            context=context,
            wheel_count=wheel_count,
            angular_diagnostics=angular_diagnostics)
        yaw_info = self._yaw_distribution_proposal(
            context=context,
            roll_info=roll_info)
        diagnostics = self._base_diagnostics(
            context=context,
            wheel_count=wheel_count,
            state_valid=state_valid,
            invalid_reason=invalid_reason,
            corner_velocities=corner_velocities,
            angular_diagnostics=angular_diagnostics)
        self._add_roll_diagnostics(
            diagnostics,
            roll_info,
            yaw_info,
            invalid_reason=invalid_reason,
            spring_scales=(1.0,) * wheel_count,
            damper_scales=(1.0,) * wheel_count)

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
                "skyhook_roll_fallback_reason": invalid_reason,
            })
            for index in range(wheel_count):
                wheel = wheels[index] if index < len(wheels) else None
                self._add_wheel_diagnostics(
                    diagnostics,
                    index=index,
                    wheel=wheel,
                    v_sprung=corner_velocities[index],
                    v_roll=roll_components[index],
                    v_pitch=pitch_components[index],
                    native_damper=(
                        native_dampers[index]
                        if index < len(native_dampers) else ""),
                    final_target=1.0,
                    final_damper=1.0,
                    law_result=None)
                self._add_roll_wheel_aliases(
                    diagnostics,
                    index=index,
                    side_weight=(
                        roll_info["side_weights"][index]
                        if index < len(roll_info["side_weights"]) else 1.0),
                    spring_scale=1.0,
                    damper_scale=1.0,
                    damper_add=0.0,
                    spring_add=0.0,
                    softening_guard_active=0)
            return ControllerOutput(command=command, diagnostics=diagnostics)

        damper_scales = []
        law_results = []
        for index in range(wheel_count):
            wheel = wheels[index]
            v_sprung = corner_velocities[index]
            v_rel = (
                _safe_float(cfg.relative_extension_sign, -1.0) *
                _safe_float(getattr(wheel, "suspension_velocity_mps", 0.0)))
            native_damper = native_dampers[index]
            f_roll = self._roll_force_ideal(
                native_damper=native_damper,
                v_roll=roll_components[index],
                roll_gate_global=roll_info["roll_gate_global"],
                side_weight=roll_info["side_weights"][index],
                axle_weight=roll_info["axle_weights"][index])
            law_result = self._target_damper_scale_with_roll(
                v_sprung=v_sprung,
                v_rel_extension=v_rel,
                native_damper=native_damper,
                f_roll_ideal=f_roll)
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
            damper_scales.append(final_damper)
            law_results.append(law_result)
            self._add_wheel_diagnostics(
                diagnostics,
                index=index,
                wheel=wheel,
                v_sprung=v_sprung,
                v_roll=roll_components[index],
                v_pitch=pitch_components[index],
                native_damper=native_damper,
                final_target=law_result["final_target_damper"],
                final_damper=final_damper,
                law_result=law_result)
            self._add_roll_wheel_aliases(
                diagnostics,
                index=index,
                side_weight=roll_info["side_weights"][index],
                spring_scale=1.0,
                damper_scale=final_damper,
                damper_add=max(
                    0.0,
                    _safe_float(law_result["final_target_damper"]) -
                    _safe_float(law_result["skyhook_only_target_damper"])),
                spring_add=0.0,
                softening_guard_active=law_result[
                    "roll_softening_guard_active"])

        command = SuspensionCommand(tuple(
            WheelScale(spring_scale=_FROZEN_SPRING_SCALE,
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
            "skyhook_roll_fallback_reason": "",
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
            "roll_softening_guard_ratio": _mean(
                tuple(result["roll_softening_guard_active"]
                      for result in law_results)),
        })
        self._add_roll_diagnostics(
            diagnostics,
            roll_info,
            yaw_info,
            invalid_reason="",
            spring_scales=(1.0,) * wheel_count,
            damper_scales=tuple(damper_scales))
        return ControllerOutput(command=command, diagnostics=diagnostics)

    def _target_damper_scale_with_roll(
        self,
        v_sprung: float,
        v_rel_extension: float,
        native_damper: float,
        f_roll_ideal: float,
    ) -> Dict[str, Any]:
        cfg = self.config
        native = max(_safe_float(native_damper, 0.0), _EPS)
        skyhook_c = max(0.0, _safe_float(cfg.skyhook_c_scale, 1.0)) * native
        f_sky_ideal = -skyhook_c * _safe_float(v_sprung)
        skyhook_only = self._project_ideal_force(
            v_sprung=v_sprung,
            v_rel_extension=v_rel_extension,
            native_damper=native,
            f_ideal=f_sky_ideal,
            high_extra_scale=0.0)
        total = self._project_ideal_force(
            v_sprung=v_sprung,
            v_rel_extension=v_rel_extension,
            native_damper=native,
            f_ideal=f_sky_ideal + f_roll_ideal,
            high_extra_scale=_safe_float(cfg.max_roll_extra_scale, 0.08))

        final_target = total["final_target_damper"]
        guard_active = 0
        neutral = _safe_float(cfg.neutral_damper_scale, 1.0)
        if skyhook_only["final_target_damper"] < neutral - 1.0e-12:
            cap = max(
                skyhook_only["final_target_damper"] +
                max(0.0, _safe_float(cfg.roll_softening_override_limit, 0.03)),
                neutral)
            if final_target > cap:
                final_target = cap
                guard_active = 1

        final_target = clamp(
            final_target,
            _safe_float(cfg.min_damper_scale, 0.75),
            _safe_float(cfg.max_damper_scale, 1.25))
        neutral = _safe_float(cfg.neutral_damper_scale, 1.0)
        soft = int(final_target < neutral - 1.0e-12)
        hard = int(final_target > neutral + 1.0e-12)
        neutral_mode = int(not soft and not hard)

        return {
            "v_rel_extension_mps": v_rel_extension,
            "F_sky_ideal": f_sky_ideal,
            "F_roll_ideal": f_roll_ideal,
            "F_total_ideal": f_sky_ideal + f_roll_ideal,
            "C_native": native,
            "C_required": total["C_required"],
            "semi_active_feasible": total["semi_active_feasible"],
            "skyhook_only_target_damper": skyhook_only[
                "final_target_damper"],
            "final_target_damper": final_target,
            "final_spring_scale": 1.0,
            "final_damper_scale": final_target,
            "rate_limited_damper": 0,
            "clamped_damper": int(total["clamped_damper"] or guard_active),
            "soft_mode": soft,
            "hard_mode": hard,
            "neutral_mode": neutral_mode,
            "roll_softening_guard_active": guard_active,
        }

    def _project_ideal_force(
        self,
        v_sprung: float,
        v_rel_extension: float,
        native_damper: float,
        f_ideal: float,
        high_extra_scale: float,
    ) -> Dict[str, Any]:
        cfg = self.config
        native = max(_safe_float(native_damper, 0.0), _EPS)
        v_sprung = _safe_float(v_sprung)
        v_rel = _safe_float(v_rel_extension)
        c_required: Any = (
            -_safe_float(f_ideal) / v_rel
            if abs(v_rel) > max(_safe_float(cfg.projection_eps), _EPS)
            else "")
        feasible = 0
        raw_target = _safe_float(cfg.neutral_damper_scale, 1.0)

        if abs(v_sprung) < max(0.0, _safe_float(cfg.sprung_velocity_deadband)):
            raw_target = _safe_float(cfg.neutral_damper_scale, 1.0)
        elif abs(v_rel) < max(0.0, _safe_float(cfg.rel_velocity_deadband)):
            raw_target = _safe_float(cfg.neutral_damper_scale, 1.0)
        elif _is_finite(c_required) and float(c_required) > 0.0:
            feasible = 1
            high_scale = (
                _safe_float(cfg.high_damper_scale, 1.22) +
                max(0.0, _safe_float(high_extra_scale, 0.0)))
            c_cmd = clamp(
                float(c_required),
                _safe_float(cfg.low_damper_scale, 0.82) * native,
                high_scale * native)
            raw_target = c_cmd / native
        else:
            raw_target = _safe_float(cfg.low_damper_scale, 0.82)

        final_target = clamp(
            raw_target,
            _safe_float(cfg.min_damper_scale, 0.75),
            _safe_float(cfg.max_damper_scale, 1.25))
        return {
            "C_required": c_required,
            "semi_active_feasible": feasible,
            "final_target_damper": final_target,
            "clamped_damper": int(not math.isclose(
                raw_target,
                final_target,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12)),
        }

    def _roll_force_ideal(
        self,
        native_damper: float,
        v_roll: float,
        roll_gate_global: float,
        side_weight: float,
        axle_weight: float,
    ) -> float:
        if not bool(getattr(self.config, "roll_damper_enabled", True)):
            return 0.0
        c_roll = (
            max(0.0, _safe_float(self.config.roll_c_scale, 0.20)) *
            max(_safe_float(native_damper, 0.0), 0.0))
        return (
            -c_roll *
            _safe_float(v_roll) *
            clamp(_safe_float(roll_gate_global), 0.0, 1.0) *
            max(0.0, _safe_float(side_weight, 1.0)) *
            max(0.0, _safe_float(axle_weight, 1.0)))

    def _roll_gate_and_weights(
        self,
        context: ControllerContext,
        wheel_count: int,
        angular_diagnostics: Mapping[str, Any],
    ) -> Dict[str, Any]:
        cfg = self.config
        state = context.state
        roll_rad = math.radians(_safe_float(getattr(state, "roll", 0.0)))
        roll_rate_rad = _safe_float(angular_diagnostics["roll_rate_rad_s"])
        pitch_rate_rad = _safe_float(angular_diagnostics["pitch_rate_rad_s"])
        yaw_rate_rad = _safe_float(angular_diagnostics["yaw_rate_rad_s"])
        local_ay = _safe_float(getattr(state, "local_ay", 0.0))

        ay_gate = _smoothstep(
            abs(local_ay),
            _safe_float(cfg.roll_ay_on, 1.5),
            _safe_float(cfg.roll_ay_full, 5.0))
        rate_gate = _smoothstep(
            abs(roll_rate_rad),
            math.radians(_safe_float(cfg.roll_rate_on_deg_s, 1.5)),
            math.radians(_safe_float(cfg.roll_rate_full_deg_s, 12.0)))
        angle_gate = _smoothstep(
            abs(roll_rad),
            math.radians(_safe_float(cfg.roll_angle_on_deg, 1.0)),
            math.radians(_safe_float(cfg.roll_angle_full_deg, 5.0)))
        roll_gate_global = ay_gate * max(rate_gate, 0.5 * angle_gate)

        corners = _match_length(self._corner_coordinates(), wheel_count, (0.0, 0.0))
        outer_side_sign = self._outer_side_sign(local_ay)
        side_weights = tuple(
            self._side_weight(y, outer_side_sign)
            for _, y in corners)
        front_share = clamp(
            _safe_float(cfg.nominal_front_roll_distribution, 0.55),
            0.0,
            1.0)
        front_axle_weight = 2.0 * front_share
        rear_axle_weight = 2.0 * (1.0 - front_share)
        axle_weights = tuple(
            front_axle_weight if x > 0.0 else rear_axle_weight
            for x, _ in corners)

        return {
            "roll_rad": roll_rad,
            "roll_rate_rad": roll_rate_rad,
            "pitch_rate_rad": pitch_rate_rad,
            "yaw_rate_rad": yaw_rate_rad,
            "local_ay": local_ay,
            "ay_gate": ay_gate,
            "rate_gate": rate_gate,
            "angle_gate": angle_gate,
            "roll_gate_global": roll_gate_global,
            "front_share": front_share,
            "outer_side_sign": outer_side_sign,
            "side_weights": side_weights,
            "axle_weights": axle_weights,
        }

    def _yaw_distribution_proposal(
        self,
        context: ControllerContext,
        roll_info: Mapping[str, Any],
    ) -> Dict[str, Any]:
        cfg = self.config
        state = context.state
        mode = self._yaw_distribution_mode()
        enabled = mode == "log_only"

        nominal_front = clamp(
            _safe_float(cfg.nominal_front_roll_distribution, 0.55),
            0.0,
            1.0)
        min_front = clamp(
            _safe_float(cfg.min_front_roll_distribution, 0.35),
            0.0,
            1.0)
        max_front = clamp(
            _safe_float(cfg.max_front_roll_distribution, 0.75),
            min_front,
            1.0)

        speed_signed, speed_abs = self._current_local_speed(state)
        road_wheel_angle = self._current_road_wheel_angle_rad(state)
        yaw_rate_actual = _safe_float(roll_info.get("yaw_rate_rad", 0.0))

        yaw_rate_ref = 0.0
        wheelbase = max(
            2.0 * abs(_safe_float(cfg.half_wheelbase_m, 1.45)),
            _EPS)
        if enabled and speed_abs >= max(0.0, _safe_float(cfg.yaw_ref_min_speed)):
            yaw_rate_ref = speed_signed * math.tan(road_wheel_angle) / wheelbase
            lat_limit = (
                max(0.0, _safe_float(cfg.yaw_ref_limit_lat_g, 0.85)) *
                max(_safe_float(cfg.g, 9.81), _EPS) /
                max(speed_abs, _EPS))
            yaw_rate_ref = clamp(yaw_rate_ref, -lat_limit, lat_limit)

        yaw_error = yaw_rate_ref - yaw_rate_actual if enabled else 0.0
        denom = max(
            abs(yaw_rate_ref),
            abs(_safe_float(cfg.small_yaw_ref, 0.03)),
            _EPS)
        yaw_error_norm = yaw_error / denom if enabled else 0.0

        activation = 0.0
        if enabled:
            g = max(_safe_float(cfg.g, 9.81), _EPS)
            activation = _smoothstep(
                abs(_safe_float(getattr(state, "local_ay", 0.0))) / g,
                max(0.0, _safe_float(cfg.yaw_activation_start_g, 0.40)),
                max(0.0, _safe_float(cfg.yaw_activation_full_g, 0.60)))

        front_delta = (
            -_safe_float(cfg.yaw_front_distribution_kp, 0.08) *
            activation *
            yaw_error_norm)
        front_proposed = clamp(nominal_front + front_delta, min_front, max_front)
        rear_proposed = 1.0 - front_proposed
        front_applied = _safe_float(roll_info.get("front_share", nominal_front))

        return {
            "yaw_distribution_mode": mode,
            "yaw_apply_enabled": 0,
            "yaw_rate_ref": yaw_rate_ref,
            "yaw_rate_actual": yaw_rate_actual if enabled else 0.0,
            "yaw_error": yaw_error,
            "yaw_error_norm": yaw_error_norm,
            "yaw_activation": activation,
            "front_distribution_nominal": nominal_front,
            "front_distribution_applied": front_applied,
            "rear_distribution_applied": 1.0 - front_applied,
            "front_distribution_proposed": front_proposed,
            "rear_distribution_proposed": rear_proposed,
            "road_wheel_angle_rad": road_wheel_angle if enabled else 0.0,
            "yaw_reference_speed_mps": speed_signed if enabled else 0.0,
        }

    def _yaw_distribution_mode(self) -> str:
        if not bool(getattr(self.config, "enable_yaw_distribution", False)):
            return "disabled"
        requested = str(
            getattr(self.config, "yaw_distribution_mode", "log_only") or
            "log_only").strip().lower()
        if requested in ("disabled", "off", "false", "0", "none"):
            return "disabled"
        return "log_only"

    def _current_local_speed(self, state: Any) -> Tuple[float, float]:
        local_vx = getattr(state, "local_vx", None)
        if _is_finite(local_vx) and abs(float(local_vx)) > _EPS:
            speed_signed = float(local_vx)
        else:
            speed_signed = _safe_float(getattr(state, "speed", 0.0), 0.0)
        return speed_signed, abs(speed_signed)

    def _current_road_wheel_angle_rad(self, state: Any) -> float:
        max_angle = max(
            abs(_safe_float(self.config.max_road_wheel_angle_rad, 0.60)),
            _EPS)
        for field_name in (
                "front_wheel_steer_angle_rad",
                "front_steer_angle_rad",
                "road_wheel_angle_rad",
                "steer_angle_rad",
                "wheel_steer_angle_rad"):
            value = getattr(state, field_name, None)
            if _is_finite(value):
                return clamp(float(value), -max_angle, max_angle)
        steer = clamp(_safe_float(getattr(state, "steer", 0.0)), -1.0, 1.0)
        return steer * max_angle

    def _outer_side_sign(self, local_ay: float) -> float:
        if abs(_safe_float(local_ay)) <= _EPS:
            return 0.0
        return _sign(
            _safe_float(self.config.outer_side_sign_from_ay, -1.0) *
            _sign(local_ay))

    def _side_weight(self, y: float, outer_side_sign: float) -> float:
        if outer_side_sign == 0.0:
            return 1.0
        if _sign(y) == _sign(outer_side_sign):
            return max(0.0, 1.0 + _safe_float(self.config.outer_side_bias, 0.10))
        return max(0.0, 1.0 - _safe_float(self.config.inner_side_relief, 0.05))

    def _add_roll_diagnostics(
        self,
        diagnostics: Dict[str, Any],
        roll_info: Mapping[str, Any],
        yaw_info: Mapping[str, Any],
        invalid_reason: str,
        spring_scales: Sequence[float],
        damper_scales: Sequence[float],
    ) -> None:
        yaw_mode = str(yaw_info.get("yaw_distribution_mode", "disabled"))
        yaw_enabled = yaw_mode == "log_only"
        controller_name = "skyhook_roll_yaw" if yaw_enabled else self.name
        roll_mode = (
            "yaw_log_only_no_apply"
            if yaw_enabled else
            "fixed_no_yaw_apply")
        diagnostics.update({
            "controller": controller_name,
            "roll_gate_global": roll_info["roll_gate_global"],
            "roll_ay_gate": roll_info["ay_gate"],
            "roll_rate_gate": roll_info["rate_gate"],
            "roll_angle_gate": roll_info["angle_gate"],
            "skyhook_roll_mode": roll_mode,
            "skyhook_roll_roll_rad": roll_info["roll_rad"],
            "skyhook_roll_roll_rate_rad": roll_info["roll_rate_rad"],
            "skyhook_roll_pitch_rate_rad": roll_info["pitch_rate_rad"],
            "skyhook_roll_yaw_rate_rad": roll_info["yaw_rate_rad"],
            "skyhook_roll_local_ay": roll_info["local_ay"],
            "skyhook_roll_lat_activity": roll_info["ay_gate"],
            "skyhook_roll_damping_activity": roll_info["roll_gate_global"],
            "skyhook_roll_stiffness_activity": 0.0,
            "skyhook_roll_front_share": roll_info["front_share"],
            "skyhook_roll_yaw_ref": yaw_info["yaw_rate_ref"],
            "skyhook_roll_under_yaw_error_norm": yaw_info["yaw_error_norm"],
            "skyhook_roll_outer_side_sign": roll_info["outer_side_sign"],
            "skyhook_roll_fallback_reason": invalid_reason,
            "yaw_distribution_mode": yaw_mode,
            "yaw_apply_enabled": yaw_info["yaw_apply_enabled"],
            "yaw_rate_ref": yaw_info["yaw_rate_ref"],
            "yaw_rate_actual": yaw_info["yaw_rate_actual"],
            "yaw_error": yaw_info["yaw_error"],
            "yaw_error_norm": yaw_info["yaw_error_norm"],
            "yaw_activation": yaw_info["yaw_activation"],
            "front_distribution_nominal": yaw_info[
                "front_distribution_nominal"],
            "front_distribution_applied": yaw_info[
                "front_distribution_applied"],
            "rear_distribution_applied": yaw_info[
                "rear_distribution_applied"],
            "front_distribution_proposed": yaw_info[
                "front_distribution_proposed"],
            "rear_distribution_proposed": yaw_info[
                "rear_distribution_proposed"],
            "road_wheel_angle_rad": yaw_info["road_wheel_angle_rad"],
            "yaw_reference_speed_mps": yaw_info["yaw_reference_speed_mps"],
            "skyhook_roll_yaw_rate_ref": yaw_info["yaw_rate_ref"],
            "skyhook_roll_yaw_rate_actual": yaw_info["yaw_rate_actual"],
            "skyhook_roll_yaw_error": yaw_info["yaw_error"],
            "skyhook_roll_yaw_activation": yaw_info["yaw_activation"],
            "skyhook_roll_front_distribution_proposed": yaw_info[
                "front_distribution_proposed"],
            "skyhook_roll_rear_distribution_proposed": yaw_info[
                "rear_distribution_proposed"],
            "spring_scale": _mean(spring_scales),
            "damper_scale": _mean(damper_scales),
        })

    def _add_roll_wheel_aliases(
        self,
        diagnostics: Dict[str, Any],
        index: int,
        side_weight: float,
        spring_scale: float,
        damper_scale: float,
        damper_add: float,
        spring_add: float,
        softening_guard_active: int,
    ) -> None:
        label = _label(index)
        diagnostics.update({
            "skyhook_roll_side_weight_%s" % label: side_weight,
            "skyhook_roll_spring_scale_%s" % label: spring_scale,
            "skyhook_roll_damper_scale_%s" % label: damper_scale,
            "skyhook_roll_damper_add_%s" % label: damper_add,
            "skyhook_roll_spring_add_%s" % label: spring_add,
            "roll_softening_guard_active_%s" % label: softening_guard_active,
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


def _sign(value: float) -> float:
    value = _safe_float(value)
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return -1.0
    return 0.0


def _smoothstep(value: float, start: float, full: float) -> float:
    value = _safe_float(value)
    start = _safe_float(start)
    full = _safe_float(full)
    if full <= start:
        return 1.0 if value > start else 0.0
    if value <= start:
        return 0.0
    if value >= full:
        return 1.0
    t = (value - start) / (full - start)
    return t * t * (3.0 - 2.0 * t)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
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
