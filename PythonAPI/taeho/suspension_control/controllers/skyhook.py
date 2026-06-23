"""State-strict semi-active skyhook damping controller."""

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


_EPS = 1.0e-9
_LABELS = ("fl", "fr", "rl", "rr")
_WHEEL_NAMES = ("FL", "FR", "RL", "RR")


@dataclass(frozen=True)
class SkyhookConfig:
    """Configuration for the state-strict skyhook baseline.

    This is a damper-only semi-active controller. It uses validated
    per-wheel suspension velocity, keeps every spring scale at identity, and
    falls back to an all-wheel identity command whenever required state is
    invalid.
    """

    controller_version: str = "skyhook_v2_state_strict"
    default_dt: float = 0.05
    wheel_count: int = 4

    half_track_m: float = 0.85
    half_wheelbase_m: float = 1.45
    front_x_sign: float = 1.0
    left_y_sign: float = -1.0

    require_suspension_state: bool = True
    identity_if_any_wheel_invalid: bool = True
    require_contact_valid: bool = True
    require_velocity: bool = True
    relative_extension_sign: float = -1.0

    skyhook_law: str = "projected_force"
    skyhook_c_scale: float = 1.0
    neutral_damper_scale: float = 1.0
    low_damper_scale: float = 0.82
    high_damper_scale: float = 1.22
    min_damper_scale: float = 0.75
    max_damper_scale: float = 1.25
    max_damper_delta_per_step: float = 0.035

    sprung_velocity_deadband: float = 0.025
    rel_velocity_deadband: float = 0.015
    product_deadband: float = 0.0005
    projection_eps: float = 1.0e-6

    angular_velocity_source: str = "actor_getter_deg_s"
    angular_velocity_unit: str = "deg_s"

    # Legacy knobs accepted by older configs/callers. They no longer define
    # the law, but keeping them avoids breaking imports that construct
    # SkyhookConfig directly.
    base_damper_scale: float = 1.0
    corner_velocity_scale: float = 1.0
    corner_velocity_deadband: float = 0.02
    skyhook_gain: float = 0.25

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SkyhookConfig":
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


class SkyhookController(SuspensionController):
    """Per-wheel projected-force semi-active skyhook controller."""

    name = "skyhook"
    requires_suspension_state = True

    def __init__(self, config: Optional[SkyhookConfig] = None):
        self.config = config or SkyhookConfig()
        self.previous_damper_scales = [
            1.0 for _ in range(max(1, int(self.config.wheel_count)))
        ]

    def reset(self, native_suspension: Any = None) -> None:
        wheel_count = _wheel_count_from_control(
            native_suspension,
            max(1, int(self.config.wheel_count)))
        self.previous_damper_scales = [1.0 for _ in range(wheel_count)]

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

        diagnostics = self._base_diagnostics(
            context=context,
            wheel_count=wheel_count,
            state_valid=state_valid,
            invalid_reason=invalid_reason,
            corner_velocities=corner_velocities,
            angular_diagnostics=angular_diagnostics)

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
            return ControllerOutput(command=command, diagnostics=diagnostics)

        damper_scales = []
        for index in range(wheel_count):
            wheel = wheels[index]
            v_sprung = corner_velocities[index]
            v_rel = (
                _safe_float(cfg.relative_extension_sign, -1.0) *
                _safe_float(getattr(wheel, "suspension_velocity_mps", 0.0)))
            native_damper = native_dampers[index]
            law_result = self._target_damper_scale(
                v_sprung=v_sprung,
                v_rel_extension=v_rel,
                native_damper=native_damper)
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

        command = SuspensionCommand(tuple(
            WheelScale(spring_scale=1.0, damper_scale=damper_scale)
            for damper_scale in damper_scales
        )).validate(expected_wheels=wheel_count)

        diagnostics.update({
            "spring_scale": 1.0,
            "damper_scale": _mean(damper_scales),
            "fallback_mode": "active",
            "identity_fallback_this_tick": 0,
            "identity_fallback_that_would_have_occurred": 0,
            "identity_fallback_reason": "",
            "soft_mode_ratio": _mean(
                _wheel_values(diagnostics, "soft_mode", wheel_count)),
            "hard_mode_ratio": _mean(
                _wheel_values(diagnostics, "hard_mode", wheel_count)),
            "neutral_mode_ratio": _mean(
                _wheel_values(diagnostics, "neutral_mode", wheel_count)),
        })
        return ControllerOutput(command=command, diagnostics=diagnostics)

    def _wheel_count(self, context: ControllerContext) -> int:
        for source_name in ("native_suspension", "current_suspension"):
            count = _wheel_count_from_control(getattr(context, source_name, None), 0)
            if count > 0:
                return count
        return max(1, int(self.config.wheel_count))

    def _ensure_previous_count(self, wheel_count: int) -> None:
        if len(self.previous_damper_scales) == wheel_count:
            return
        self.previous_damper_scales = _match_length(
            self.previous_damper_scales,
            wheel_count,
            1.0)

    def _native_damper_rates(
        self,
        context: ControllerContext,
        wheel_count: int,
    ) -> Tuple[float, ...]:
        rates = tuple(context.native_damper_rate_by_wheel or ())
        if not rates:
            rates = tuple(
                _safe_float(getattr(wheel, "spring_damper_rate", 0.0))
                for wheel in tuple(
                    getattr(context.native_suspension, "wheels", ()) or ()))
        return _match_length(rates, wheel_count, 0.0)

    def _invalid_reason(
        self,
        context: ControllerContext,
        wheels: Sequence[Any],
        native_dampers: Sequence[float],
        wheel_count: int,
    ) -> str:
        cfg = self.config
        expected = int(cfg.wheel_count)
        if wheel_count != expected:
            return "wheel_count_%d_expected_%d" % (wheel_count, expected)
        if str(cfg.skyhook_law).strip().lower() not in (
                "projected_force",
                "switching"):
            return "unsupported_skyhook_law_%s" % cfg.skyhook_law
        if not bool(cfg.require_suspension_state):
            return "suspension_state_disabled"
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
        for index, native_damper in enumerate(native_dampers):
            if not _is_finite(native_damper) or float(native_damper) <= 0.0:
                return "wheel_%d_native_damper_rate_invalid" % index
        for index, wheel in enumerate(wheels):
            if not bool(getattr(wheel, "field_valid", False)):
                return "wheel_%d_field_valid_false" % index
            if cfg.require_velocity and not bool(
                    getattr(wheel, "velocity_valid", False)):
                return "wheel_%d_velocity_valid_false" % index
            if cfg.require_contact_valid and not bool(
                    getattr(wheel, "contact_valid", False)):
                return "wheel_%d_contact_valid_false" % index
            if cfg.require_contact_valid and bool(
                    getattr(wheel, "wheel_in_air", False)):
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

    def _base_diagnostics(
        self,
        context: ControllerContext,
        wheel_count: int,
        state_valid: bool,
        invalid_reason: str,
        corner_velocities: Sequence[float],
        angular_diagnostics: Mapping[str, Any],
    ) -> Dict[str, Any]:
        state = context.state
        suspension_state = context.suspension_state
        wheels = tuple(getattr(suspension_state, "wheels", ()) or ())
        return {
            "controller": self.name,
            "controller_version": self.config.controller_version,
            "dt": _safe_dt(context.dt, self.config.default_dt),
            "state_valid": 1,
            "suspension_state_valid": int(bool(state_valid)),
            "contact_valid_all": int(self._contact_valid_all(wheels)),
            "fallback_mode": "active" if state_valid else "identity",
            "identity_fallback_this_tick": int(not state_valid),
            "identity_fallback_that_would_have_occurred": int(not state_valid),
            "identity_fallback_reason": invalid_reason,
            "angular_velocity_source": angular_diagnostics[
                "angular_velocity_source"],
            "angular_velocity_unit_converted": angular_diagnostics[
                "angular_velocity_unit_converted"],
            "roll_deg": _finite_or_empty(getattr(state, "roll", "")),
            "pitch_deg": _finite_or_empty(getattr(state, "pitch", "")),
            "yaw_deg": _finite_or_empty(getattr(state, "yaw", "")),
            "roll_rate_rad_s": angular_diagnostics["roll_rate_rad_s"],
            "pitch_rate_rad_s": angular_diagnostics["pitch_rate_rad_s"],
            "yaw_rate_rad_s": angular_diagnostics["yaw_rate_rad_s"],
            "local_vx": _finite_or_empty(getattr(state, "local_vx", "")),
            "local_vy": _finite_or_empty(getattr(state, "local_vy", "")),
            "local_vz": _finite_or_empty(getattr(state, "vz", "")),
            "local_ax": _finite_or_empty(getattr(state, "local_ax", "")),
            "local_ay": _finite_or_empty(getattr(state, "local_ay", "")),
            "local_az": _finite_or_empty(getattr(state, "az", "")),
            "throttle": _finite_or_empty(getattr(state, "throttle", "")),
            "brake": _finite_or_empty(getattr(state, "brake", "")),
            "steer": _finite_or_empty(getattr(state, "steer", "")),
            "compression_convention_validated": int(bool(getattr(
                suspension_state,
                "compression_convention_validated",
                False))),
            "suspension_state_source": getattr(
                suspension_state,
                "state_source",
                ""),
            "suspension_failure_reason": getattr(
                suspension_state,
                "failure_reason",
                ""),
            "suspension_wheel_count": len(wheels),
            "skyhook_law": self.config.skyhook_law,
            "skyhook_c_scale": self.config.skyhook_c_scale,
            "relative_extension_sign": self.config.relative_extension_sign,
            "skyhook_mean_abs_corner_vz": _mean_abs(corner_velocities),
            "skyhook_max_activity": max(
                (self._corner_activity(value) for value in corner_velocities),
                default=0.0),
            "skyhook_roll_rate_rad": angular_diagnostics["roll_rate_rad_s"],
            "skyhook_pitch_rate_rad": angular_diagnostics["pitch_rate_rad_s"],
            "soft_mode_ratio": "",
            "hard_mode_ratio": "",
            "neutral_mode_ratio": "",
        }

    def _add_wheel_diagnostics(
        self,
        diagnostics: Dict[str, Any],
        index: int,
        wheel: Any,
        v_sprung: float,
        v_roll: float,
        v_pitch: float,
        native_damper: Any,
        final_target: float,
        final_damper: float,
        law_result: Optional[Mapping[str, Any]],
    ) -> None:
        label = _label(index)
        activity = self._corner_activity(v_sprung)
        diagnostics.update({
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
            "native_spring_strength_%s" % label: "",
            "native_damper_rate_%s" % label: _finite_or_empty(native_damper),
            "v_sprung_%s" % label: v_sprung,
            "v_roll_%s" % label: v_roll,
            "v_pitch_%s" % label: v_pitch,
            "v_rel_extension_mps_%s" % label: "",
            "F_sky_ideal_%s" % label: "",
            "F_roll_ideal_%s" % label: 0.0,
            "F_total_ideal_%s" % label: "",
            "C_native_%s" % label: _finite_or_empty(native_damper),
            "C_required_%s" % label: "",
            "semi_active_feasible_%s" % label: "",
            "skyhook_only_target_damper_%s" % label: final_target,
            "final_target_damper_%s" % label: final_target,
            "final_spring_scale_%s" % label: 1.0,
            "final_damper_scale_%s" % label: final_damper,
            "rate_limited_damper_%s" % label: "",
            "clamped_damper_%s" % label: "",
            "soft_mode_%s" % label: "",
            "hard_mode_%s" % label: "",
            "neutral_mode_%s" % label: "",
            "skyhook_corner_vz_%s" % label: v_sprung,
            "skyhook_activity_%s" % label: activity,
            "skyhook_damper_scale_%s" % label: final_damper,
        })
        if wheel is not None:
            diagnostics.update({
                "wheel_index_raw_%s" % label: getattr(
                    wheel,
                    "wheel_index_raw",
                    index),
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
        if law_result:
            for key, value in law_result.items():
                diagnostics["%s_%s" % (key, label)] = value

    def _target_damper_scale(
        self,
        v_sprung: float,
        v_rel_extension: float,
        native_damper: float,
    ) -> Dict[str, Any]:
        cfg = self.config
        native = max(_safe_float(native_damper, 0.0), _EPS)
        skyhook_c = max(0.0, _safe_float(cfg.skyhook_c_scale, 1.0)) * native
        f_ideal = -skyhook_c * v_sprung
        product = v_sprung * v_rel_extension
        c_required = (
            -f_ideal / v_rel_extension
            if abs(v_rel_extension) > max(_safe_float(cfg.projection_eps), _EPS)
            else "")
        feasible = 0
        raw_target = _safe_float(cfg.neutral_damper_scale, 1.0)

        if abs(v_sprung) < max(0.0, _safe_float(cfg.sprung_velocity_deadband)):
            raw_target = _safe_float(cfg.neutral_damper_scale, 1.0)
        elif abs(v_rel_extension) < max(0.0, _safe_float(cfg.rel_velocity_deadband)):
            raw_target = _safe_float(cfg.neutral_damper_scale, 1.0)
        elif product > max(0.0, _safe_float(cfg.product_deadband)):
            feasible = 1
            if str(cfg.skyhook_law).strip().lower() == "switching":
                raw_target = _safe_float(cfg.high_damper_scale, 1.22)
            else:
                required = (
                    skyhook_c * abs(v_sprung) /
                    max(abs(v_rel_extension), _safe_float(cfg.projection_eps)))
                c_cmd = clamp(
                    required,
                    _safe_float(cfg.low_damper_scale, 0.82) * native,
                    _safe_float(cfg.high_damper_scale, 1.22) * native)
                raw_target = c_cmd / native
        else:
            raw_target = _safe_float(cfg.low_damper_scale, 0.82)

        final_target = clamp(
            raw_target,
            _safe_float(cfg.min_damper_scale, 0.75),
            _safe_float(cfg.max_damper_scale, 1.25))
        clamped = int(not math.isclose(
            raw_target,
            final_target,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12))
        neutral = _safe_float(cfg.neutral_damper_scale, 1.0)
        soft = int(final_target < neutral - 1.0e-12)
        hard = int(final_target > neutral + 1.0e-12)
        neutral_mode = int(not soft and not hard)

        return {
            "v_rel_extension_mps": v_rel_extension,
            "F_sky_ideal": f_ideal,
            "F_roll_ideal": 0.0,
            "F_total_ideal": f_ideal,
            "C_native": native,
            "C_required": c_required,
            "semi_active_feasible": feasible,
            "skyhook_only_target_damper": final_target,
            "final_target_damper": final_target,
            "final_spring_scale": 1.0,
            "final_damper_scale": final_target,
            "rate_limited_damper": 0,
            "clamped_damper": clamped,
            "soft_mode": soft,
            "hard_mode": hard,
            "neutral_mode": neutral_mode,
        }

    def _rate_limit_damper(self, index: int, desired_damper_scale: float) -> float:
        max_delta = max(0.0, _safe_float(self.config.max_damper_delta_per_step))
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

    def _corner_activity(self, corner_vz: float) -> float:
        magnitude = max(
            0.0,
            abs(_safe_float(corner_vz)) -
            max(0.0, _safe_float(self.config.sprung_velocity_deadband)))
        return magnitude / max(
            abs(_safe_float(self.config.corner_velocity_scale, 1.0)),
            _EPS)

    def _corner_vertical_velocity_terms(
        self,
        state: VehicleState,
    ) -> Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[float, ...], Dict[str, Any]]:
        roll_rate, pitch_rate, yaw_rate, angular_diagnostics = (
            self._angular_rates_rad_s(state))
        vz = _safe_float(getattr(state, "vz", 0.0))
        corners = self._corner_coordinates()
        roll_components = tuple(roll_rate * y for _, y in corners)
        pitch_components = tuple(-pitch_rate * x for x, _ in corners)
        velocities = tuple(
            vz + v_roll + v_pitch
            for v_roll, v_pitch in zip(roll_components, pitch_components))
        angular_diagnostics.update({
            "roll_rate_rad_s": roll_rate,
            "pitch_rate_rad_s": pitch_rate,
            "yaw_rate_rad_s": yaw_rate,
        })
        return velocities, roll_components, pitch_components, angular_diagnostics

    def _angular_rates_rad_s(
        self,
        state: VehicleState,
    ) -> Tuple[float, float, float, Dict[str, Any]]:
        source = str(self.config.angular_velocity_source or "actor_getter_deg_s")
        unit = str(self.config.angular_velocity_unit or "").strip().lower()
        if not unit:
            unit = (
                "rad_s"
                if source in ("snapshot_rad_s", "imu_rad_s")
                else "deg_s")
        roll = _safe_float(getattr(state, "roll_rate", 0.0))
        pitch = _safe_float(getattr(state, "pitch_rate", 0.0))
        yaw = _safe_float(getattr(state, "yaw_rate", 0.0))
        if unit in ("rad_s", "rad/s", "radians"):
            converted = "none_rad_s"
            return roll, pitch, yaw, {
                "angular_velocity_source": source,
                "angular_velocity_unit_converted": converted,
            }
        converted = "deg_s_to_rad_s"
        return math.radians(roll), math.radians(pitch), math.radians(yaw), {
            "angular_velocity_source": source,
            "angular_velocity_unit_converted": converted,
        }

    def _corner_coordinates(self) -> Tuple[Tuple[float, float], ...]:
        cfg = self.config
        x_front = (
            _safe_float(cfg.front_x_sign, 1.0) *
            abs(_safe_float(cfg.half_wheelbase_m, 1.45)))
        x_rear = -x_front
        y_left = (
            _safe_float(cfg.left_y_sign, -1.0) *
            abs(_safe_float(cfg.half_track_m, 0.85)))
        y_right = -y_left
        return (
            (x_front, y_left),
            (x_front, y_right),
            (x_rear, y_left),
            (x_rear, y_right),
        )

    def _contact_valid_all(self, wheels: Sequence[Any]) -> bool:
        if not wheels:
            return False
        return all(
            bool(getattr(wheel, "contact_valid", False)) and
            not bool(getattr(wheel, "wheel_in_air", False))
            for wheel in wheels)


def _wheel_count_from_control(control: Any, default: int) -> int:
    wheels = getattr(control, "wheels", None)
    if wheels is None:
        return default
    try:
        count = len(wheels)
    except TypeError:
        return default
    return count if count > 0 else default


def _safe_dt(value: Any, default: float) -> float:
    result = _safe_float(value, default)
    if result <= 0.0 or result > 1.0:
        return _safe_float(default, 0.05)
    return result


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
    if index < len(_LABELS):
        return _LABELS[index]
    return "w%d" % index


def _wheel_values(
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
