"""Planning preview feedforward composed with skyhook_roll feedback."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .base import (
    ControllerContext,
    ControllerOutput,
    PlanningInfo,
    SuspensionCommand,
    VehicleState,
    WheelScale,
    clamp,
)
from .planning_aware_risk_damping import (
    ema_update,
    robust_preview_risk,
    smoothstep,
)
from .skyhook_roll import SkyhookRollConfig, SkyhookRollController


_EPS = 1.0e-9
_WHEEL_LABELS = ("fl", "fr", "rl", "rr")


@dataclass(frozen=True)
class PlanningAwareSkyhookRollConfig(SkyhookRollConfig):
    """Config for preview feedforward plus the exact skyhook_roll residual."""

    controller_version: str = (
        "planning_aware_skyhook_roll_v1_feedforward_plus_skyhook_roll")

    max_preview_damper_extra: float = 0.035
    preview_tau_rise: float = 0.30
    preview_tau_fall: float = 0.90
    preview_horizon_s: float = 1.5
    near_window_s: float = 0.75
    far_window_s: float = 1.50
    far_weight: float = 0.50
    single_point_weight: float = 0.60
    max_planning_age_frames: int = 5
    min_horizon_dt: float = 0.02
    max_horizon_dt: float = 0.20
    min_preview_points_for_trajectory_curvature: int = 3
    speed_cap_mps: float = 45.0
    motion_gate_low: float = 3.0
    motion_gate_high: float = 7.0
    ay_abs_cap: float = 8.0
    ay_low: float = 2.0
    ay_high: float = 5.0
    curvature_abs_cap: float = 0.25
    curvature_low: float = 0.030
    curvature_high: float = 0.110
    steer_low: float = 0.12
    steer_high: float = 0.50
    target_speed_drop_window_s: float = 1.0
    decel_low: float = 1.5
    decel_high: float = 4.0
    brake_low: float = 0.20
    brake_high: float = 0.70
    speed_drop_low: float = 2.0
    speed_drop_high: float = 6.0
    preview_inner_side_factor: float = 0.55

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> "PlanningAwareSkyhookRollConfig":
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


class PlanningAwareSkyhookRollController(SkyhookRollController):
    """LEAD preview feedforward plus unchanged skyhook_roll feedback residual."""

    name = "planning_aware"
    requires_suspension_state = True

    def __init__(
        self,
        config: Optional[PlanningAwareSkyhookRollConfig] = None,
    ):
        super().__init__(config or PlanningAwareSkyhookRollConfig())
        self.preview_risk_smooth = 0.0
        self.previous_combined_damper_scales = [
            1.0 for _ in range(max(1, int(self.config.wheel_count)))
        ]

    def reset(self, native_suspension: Any = None) -> None:
        super().reset(native_suspension)
        self.preview_risk_smooth = 0.0
        self.previous_combined_damper_scales = [
            1.0 for _ in range(len(self.previous_damper_scales))
        ]

    def compute(self, context: ControllerContext) -> ControllerOutput:
        feedback_output = super().compute(context)
        wheel_count = len(feedback_output.command.wheels)
        self._ensure_combined_previous_count(wheel_count)

        diagnostics = dict(feedback_output.diagnostics)
        feedback_dampers = tuple(
            wheel.damper_scale for wheel in feedback_output.command.wheels)
        feedback_springs = tuple(
            wheel.spring_scale for wheel in feedback_output.command.wheels)
        for index, label in enumerate(_WHEEL_LABELS[:wheel_count]):
            diagnostics["planning_aware_feedback_damper_%s" % label] = (
                feedback_dampers[index])
            diagnostics["planning_aware_feedback_spring_%s" % label] = (
                feedback_springs[index])

        preview = self._preview_feedforward(context, wheel_count)
        feedback_fallback = int(
            diagnostics.get("identity_fallback_this_tick", 0) or 0)
        if feedback_fallback:
            self.previous_combined_damper_scales = [1.0] * wheel_count
            preview = dict(preview)
            preview["valid"] = 0
            preview["fallback_reason"] = (
                "feedback_identity_fallback:" +
                str(diagnostics.get("identity_fallback_reason", "")))
            preview["risk_raw"] = 0.0
            preview["risk_smooth"] = 0.0
            preview["extras"] = (0.0,) * wheel_count
            self.preview_risk_smooth = 0.0
            self._add_preview_diagnostics(
                diagnostics,
                preview,
                feedback_dampers,
                feedback_dampers,
                rate_limited=(0,) * wheel_count,
                clamped=(0,) * wheel_count)
            diagnostics["controller"] = self.name
            diagnostics["controller_name"] = self.name
            diagnostics["controller_version"] = self.config.controller_version
            return ControllerOutput(
                command=feedback_output.command,
                diagnostics=diagnostics)

        combined_dampers = []
        rate_limited = []
        clamped = []
        min_damper = _safe_float(self.config.min_damper_scale, 0.75)
        max_damper = _safe_float(self.config.max_damper_scale, 1.25)
        max_delta = _safe_float(self.config.max_damper_delta_per_step, 0.035)
        for index, feedback_damper in enumerate(feedback_dampers):
            desired = feedback_damper + preview["extras"][index]
            bounded = clamp(desired, min_damper, max_damper)
            clamped.append(int(not math.isclose(
                desired,
                bounded,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12)))
            previous = self.previous_combined_damper_scales[index]
            if max_delta > 0.0:
                limited = clamp(bounded, previous - max_delta, previous + max_delta)
            else:
                limited = bounded
            rate_limited.append(int(not math.isclose(
                limited,
                bounded,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12)))
            combined_dampers.append(limited)
        self.previous_combined_damper_scales = list(combined_dampers)

        command = SuspensionCommand(tuple(
            WheelScale(spring_scale=spring, damper_scale=damper)
            for spring, damper in zip(feedback_springs, combined_dampers)
        )).validate(expected_wheels=wheel_count)

        self._add_preview_diagnostics(
            diagnostics,
            preview,
            feedback_dampers,
            tuple(combined_dampers),
            rate_limited=tuple(rate_limited),
            clamped=tuple(clamped))
        diagnostics.update({
            "controller": self.name,
            "controller_name": self.name,
            "controller_version": self.config.controller_version,
            "damper_scale": _mean(combined_dampers),
            "spring_scale": _mean(feedback_springs),
        })
        return ControllerOutput(command=command, diagnostics=diagnostics)

    def _ensure_combined_previous_count(self, wheel_count: int) -> None:
        if len(self.previous_combined_damper_scales) == wheel_count:
            return
        values = tuple(self.previous_combined_damper_scales or ())
        if len(values) > wheel_count:
            self.previous_combined_damper_scales = list(values[:wheel_count])
        else:
            self.previous_combined_damper_scales = list(
                values + tuple(1.0 for _ in range(wheel_count - len(values))))

    def _preview_feedforward(
        self,
        context: ControllerContext,
        wheel_count: int,
    ) -> Dict[str, Any]:
        cfg = self.config
        planning = getattr(context, "planning", None)
        base = {
            "valid": 0,
            "fallback_reason": "",
            "risk_raw": 0.0,
            "risk_smooth": self.preview_risk_smooth,
            "r_lat_preview": 0.0,
            "r_brake_preview": 0.0,
            "r_curvature_preview": 0.0,
            "r_ay_preview": 0.0,
            "r_steer_preview": 0.0,
            "r_decel_preview": 0.0,
            "r_speed_drop": 0.0,
            "motion_gate": 0.0,
            "source_mask": "none",
            "source": getattr(planning, "source", "") if planning is not None else "",
            "age_frames": -1.0,
            "points_used": 0,
            "horizon_dt": -1.0,
            "curvature_source": "missing",
            "predicted_ay_source": "missing",
            "signed_lat_peak": 0.0,
            "extras": (0.0,) * wheel_count,
        }
        if planning is None:
            base["fallback_reason"] = "no_planning"
            self.preview_risk_smooth = 0.0
            base["risk_smooth"] = 0.0
            return base
        if not bool(getattr(planning, "available", False)):
            base["fallback_reason"] = "planning_unavailable"
            self.preview_risk_smooth = 0.0
            base["risk_smooth"] = 0.0
            return base
        metadata = dict(getattr(planning, "metadata", {}) or {})
        if _metadata_false(metadata.get("valid_prediction", None)):
            base["fallback_reason"] = "metadata_invalid_prediction"
            self.preview_risk_smooth = 0.0
            base["risk_smooth"] = 0.0
            return base

        state = getattr(context, "state", VehicleState())
        state_frame = _optional_finite_float(getattr(state, "frame", None))
        planning_frame = _optional_finite_float(getattr(planning, "frame", None))
        if state_frame is None or planning_frame is None or planning_frame < 0.0:
            base["fallback_reason"] = "missing_frame"
            self.preview_risk_smooth = 0.0
            base["risk_smooth"] = 0.0
            return base
        age_frames = int(state_frame) - int(planning_frame)
        base["age_frames"] = float(age_frames)
        if age_frames > int(cfg.max_planning_age_frames):
            base["fallback_reason"] = "stale_planning"
            self.preview_risk_smooth = 0.0
            base["risk_smooth"] = 0.0
            return base

        horizon_dt = _optional_finite_float(getattr(planning, "horizon_dt", None))
        if (
                horizon_dt is None or
                horizon_dt < cfg.min_horizon_dt or
                horizon_dt > cfg.max_horizon_dt):
            base["fallback_reason"] = "bad_horizon_dt"
            self.preview_risk_smooth = 0.0
            base["risk_smooth"] = 0.0
            return base
        base["horizon_dt"] = horizon_dt

        features = self._extract_preview_features(state, planning, horizon_dt)
        if features["fallback_reason"]:
            base["fallback_reason"] = features["fallback_reason"]
            self.preview_risk_smooth = 0.0
            base["risk_smooth"] = 0.0
            return base

        motion_gate = smoothstep(
            _safe_float(getattr(state, "speed", 0.0)),
            cfg.motion_gate_low,
            cfg.motion_gate_high)
        r_ay = robust_preview_risk(
            features["ay_abs"],
            features["times"],
            cfg.ay_low,
            cfg.ay_high,
            cfg.near_window_s,
            cfg.far_window_s,
            cfg.far_weight,
            cfg.single_point_weight)
        r_curvature = robust_preview_risk(
            features["curvature_abs"],
            features["times"],
            cfg.curvature_low,
            cfg.curvature_high,
            cfg.near_window_s,
            cfg.far_window_s,
            cfg.far_weight,
            cfg.single_point_weight)
        r_steer = robust_preview_risk(
            features["steer_abs"],
            features["times"],
            cfg.steer_low,
            cfg.steer_high,
            cfg.near_window_s,
            cfg.far_window_s,
            cfg.far_weight,
            cfg.single_point_weight)
        r_lat = motion_gate * max(r_ay, 0.65 * r_curvature, 0.25 * r_steer)
        r_decel = robust_preview_risk(
            features["decel"],
            features["times"],
            cfg.decel_low,
            cfg.decel_high,
            cfg.near_window_s,
            cfg.far_window_s,
            cfg.far_weight,
            cfg.single_point_weight)
        r_brake = robust_preview_risk(
            features["brake"],
            features["times"],
            cfg.brake_low,
            cfg.brake_high,
            cfg.near_window_s,
            cfg.far_window_s,
            cfg.far_weight,
            cfg.single_point_weight)
        r_speed_drop = smoothstep(
            features["target_speed_drop_1s"],
            cfg.speed_drop_low,
            cfg.speed_drop_high)
        r_brake_combined = motion_gate * max(r_decel, r_brake, 0.35 * r_speed_drop)
        risk_raw = clamp(max(r_lat, 0.65 * r_brake_combined), 0.0, 1.0)
        dt = _dt(context, cfg.default_dt)
        self.preview_risk_smooth = ema_update(
            self.preview_risk_smooth,
            risk_raw,
            dt,
            cfg.preview_tau_rise,
            cfg.preview_tau_fall)
        risk_scale = (
            self.preview_risk_smooth / risk_raw
            if risk_raw > _EPS else 0.0)
        max_extra = max(0.0, _safe_float(cfg.max_preview_damper_extra, 0.035))
        lat_extra = max_extra * r_lat * risk_scale
        brake_extra = max_extra * 0.65 * r_brake_combined * risk_scale
        extras = self._wheel_feedforward_extras(
            wheel_count=wheel_count,
            lat_extra=lat_extra,
            brake_extra=brake_extra,
            signed_lat_peak=features["signed_lat_peak"])

        base.update({
            "valid": 1,
            "fallback_reason": "none",
            "risk_raw": risk_raw,
            "risk_smooth": self.preview_risk_smooth,
            "r_lat_preview": r_lat,
            "r_brake_preview": r_brake_combined,
            "r_curvature_preview": r_curvature,
            "r_ay_preview": r_ay,
            "r_steer_preview": r_steer,
            "r_decel_preview": r_decel,
            "r_speed_drop": r_speed_drop,
            "motion_gate": motion_gate,
            "source_mask": features["source_mask"],
            "points_used": len(features["times"]),
            "curvature_source": features["curvature_source"],
            "predicted_ay_source": features["predicted_ay_source"],
            "signed_lat_peak": features["signed_lat_peak"],
            "extras": extras,
        })
        return base

    def _extract_preview_features(
        self,
        state: VehicleState,
        planning: PlanningInfo,
        horizon_dt: float,
    ) -> Dict[str, Any]:
        cfg = self.config
        points = tuple(getattr(planning, "points", ()) or ())
        lengths = [
            len(getattr(planning, "predicted_ay", ()) or ()),
            len(getattr(planning, "curvature", ()) or ()),
            len(getattr(planning, "trajectory_xy", ()) or ()),
            len(getattr(planning, "target_speed", ()) or ()),
            len(getattr(planning, "steer", ()) or ()),
            len(getattr(planning, "brake", ()) or ()),
            len(getattr(planning, "predicted_ax", ()) or ()),
            len(points),
        ]
        horizon_len = max(lengths or [0])
        if horizon_len <= 0:
            return _empty_features("no_valid_preview_signal")

        times_all = tuple(index * horizon_dt for index in range(horizon_len))
        indices = tuple(
            index for index, value in enumerate(times_all)
            if 0.0 <= value <= cfg.preview_horizon_s)
        if not indices:
            return _empty_features("no_valid_preview_signal")

        speeds, negative_speed = self._target_speeds(state, planning, horizon_len)
        if negative_speed:
            return _empty_features("negative_target_speed")
        curvatures, curvature_source = self._curvatures(planning, horizon_len)
        predicted_ay, predicted_ay_source = self._predicted_ay(
            planning,
            speeds,
            curvatures,
            curvature_source,
            horizon_len)
        predicted_ax = _series(getattr(planning, "predicted_ax", ()), horizon_len)
        if not any(value is not None for value in predicted_ax):
            predicted_ax = _speed_derivatives(speeds, horizon_dt)
        brake = _series(getattr(planning, "brake", ()), horizon_len)
        steer = _series(getattr(planning, "steer", ()), horizon_len)

        source_mask = []
        if any(value is not None for value in predicted_ay):
            source_mask.append("ay")
        if any(value is not None for value in curvatures):
            source_mask.append("kappa")
        if any(value is not None for value in predicted_ax):
            source_mask.append("decel")
        if any(value is not None for value in brake):
            source_mask.append("brake")
        if any(value is not None for value in steer):
            source_mask.append("steer")
        if any(value is not None for value in speeds):
            source_mask.append("target_speed")
        if not source_mask:
            return _empty_features("no_valid_preview_signal")

        times = tuple(times_all[index] for index in indices)
        curvature_abs = tuple(
            clamp(abs(curvatures[index]), 0.0, cfg.curvature_abs_cap)
            if curvatures[index] is not None else 0.0
            for index in indices)
        ay_abs = tuple(
            clamp(abs(predicted_ay[index]), 0.0, cfg.ay_abs_cap)
            if predicted_ay[index] is not None else 0.0
            for index in indices)
        decel = tuple(
            max(0.0, -predicted_ax[index])
            if predicted_ax[index] is not None else 0.0
            for index in indices)
        brake_abs = tuple(
            clamp(brake[index], 0.0, 1.0)
            if brake[index] is not None else 0.0
            for index in indices)
        steer_abs = tuple(
            abs(steer[index])
            if steer[index] is not None else 0.0
            for index in indices)
        target_speed_drop_1s = 0.0
        speeds_1s = [
            speeds[index]
            for index in indices
            if times_all[index] <= cfg.target_speed_drop_window_s and
            speeds[index] is not None
        ]
        if speeds_1s:
            target_speed_drop_1s = max(
                0.0,
                _safe_float(getattr(state, "speed", 0.0)) - min(speeds_1s))

        signed_lat_peak = _signed_peak(
            predicted_ay,
            curvatures,
            indices,
            _safe_float(getattr(state, "speed", 0.0)))
        return {
            "fallback_reason": "",
            "times": times,
            "curvature_abs": curvature_abs,
            "ay_abs": ay_abs,
            "decel": decel,
            "brake": brake_abs,
            "steer_abs": steer_abs,
            "target_speed_drop_1s": target_speed_drop_1s,
            "curvature_source": curvature_source,
            "predicted_ay_source": predicted_ay_source,
            "signed_lat_peak": signed_lat_peak,
            "source_mask": ",".join(source_mask),
        }

    def _target_speeds(
        self,
        state: VehicleState,
        planning: PlanningInfo,
        horizon_len: int,
    ) -> Tuple[Tuple[Optional[float], ...], bool]:
        cfg = self.config
        raw = tuple(getattr(planning, "target_speed", ()) or ())
        if len(raw) == 1:
            value = _optional_finite_float(raw[0])
            if value is not None and value < 0.0:
                return tuple(None for _ in range(horizon_len)), True
            if value is not None:
                bounded = clamp(value, 0.0, cfg.speed_cap_mps)
                return tuple(bounded for _ in range(horizon_len)), False
        values = _series(raw, horizon_len)
        if any(value is not None and value < 0.0 for value in values):
            return values, True
        if any(value is not None for value in values):
            return tuple(
                clamp(value, 0.0, cfg.speed_cap_mps)
                if value is not None else None
                for value in values), False
        points = tuple(getattr(planning, "points", ()) or ())
        point_speeds = _series(
            tuple(getattr(point, "speed", None) for point in points),
            horizon_len)
        if any(value is not None for value in point_speeds):
            return tuple(
                clamp(value, 0.0, cfg.speed_cap_mps)
                if value is not None else None
                for value in point_speeds), False
        fallback_speed = clamp(
            _safe_float(getattr(state, "speed", 0.0)),
            0.0,
            cfg.speed_cap_mps)
        return tuple(fallback_speed for _ in range(horizon_len)), False

    def _curvatures(
        self,
        planning: PlanningInfo,
        horizon_len: int,
    ) -> Tuple[Tuple[Optional[float], ...], str]:
        raw = _series(getattr(planning, "curvature", ()), horizon_len)
        if any(value is not None for value in raw):
            return raw, "planning"
        points = tuple(getattr(planning, "points", ()) or ())
        point_values = _series(
            tuple(getattr(point, "curvature", None) for point in points),
            horizon_len)
        if any(value is not None for value in point_values):
            return point_values, "points"
        trajectory = tuple(getattr(planning, "trajectory_xy", ()) or ())
        values = _trajectory_curvature(
            trajectory,
            self.config.min_preview_points_for_trajectory_curvature)
        if values:
            padded = tuple(
                values[index] if index < len(values) else values[-1]
                for index in range(horizon_len))
            return padded, "trajectory"
        return tuple(None for _ in range(horizon_len)), "missing"

    def _predicted_ay(
        self,
        planning: PlanningInfo,
        speeds: Sequence[Optional[float]],
        curvatures: Sequence[Optional[float]],
        curvature_source: str,
        horizon_len: int,
    ) -> Tuple[Tuple[Optional[float], ...], str]:
        raw = _series(getattr(planning, "predicted_ay", ()), horizon_len)
        if any(value is not None for value in raw):
            return raw, "planning"
        if curvature_source == "missing":
            return tuple(None for _ in range(horizon_len)), "missing"
        values = []
        for speed, curvature in zip(speeds, curvatures):
            if speed is None or curvature is None:
                values.append(None)
            else:
                values.append(speed * speed * curvature)
        if any(value is not None for value in values):
            return tuple(values), "v2kappa"
        return tuple(None for _ in range(horizon_len)), "missing"

    def _wheel_feedforward_extras(
        self,
        wheel_count: int,
        lat_extra: float,
        brake_extra: float,
        signed_lat_peak: float,
    ) -> Tuple[float, ...]:
        cfg = self.config
        max_extra = max(0.0, _safe_float(cfg.max_preview_damper_extra, 0.035))
        corners = tuple(self._corner_coordinates())[:wheel_count]
        if abs(signed_lat_peak) <= _EPS or lat_extra <= _EPS:
            return tuple(
                clamp(max(lat_extra, brake_extra), 0.0, max_extra)
                for _ in range(wheel_count))
        outer_side = _sign(
            _safe_float(cfg.outer_side_sign_from_ay, -1.0) *
            _sign(signed_lat_peak))
        inner_factor = clamp(
            _safe_float(cfg.preview_inner_side_factor, 0.55),
            0.0,
            1.0)
        extras = []
        for _, y_value in corners:
            side_factor = 1.0 if _sign(y_value) == outer_side else inner_factor
            extras.append(clamp(max(brake_extra, lat_extra * side_factor), 0.0, max_extra))
        return tuple(extras)

    def _add_preview_diagnostics(
        self,
        diagnostics: Dict[str, Any],
        preview: Mapping[str, Any],
        feedback_dampers: Sequence[float],
        combined_dampers: Sequence[float],
        rate_limited: Sequence[int],
        clamped: Sequence[int],
    ) -> None:
        wheel_count = len(combined_dampers)
        diagnostics.update({
            "planning_aware_feedback_source": "skyhook_roll",
            "planning_aware_preview_source": preview.get("source", ""),
            "planning_aware_valid": int(preview.get("valid", 0)),
            "planning_aware_fallback_reason": preview.get("fallback_reason", ""),
            "planning_aware_risk_raw": _safe_float(preview.get("risk_raw")),
            "planning_aware_risk_smooth": _safe_float(preview.get("risk_smooth")),
            "planning_aware_r_lat_preview": _safe_float(
                preview.get("r_lat_preview")),
            "planning_aware_r_brake_preview": _safe_float(
                preview.get("r_brake_preview")),
            "planning_aware_r_curvature_preview": _safe_float(
                preview.get("r_curvature_preview")),
            "planning_aware_r_ay_preview": _safe_float(
                preview.get("r_ay_preview")),
            "planning_aware_r_steer_preview": _safe_float(
                preview.get("r_steer_preview")),
            "planning_aware_r_decel_preview": _safe_float(
                preview.get("r_decel_preview")),
            "planning_aware_r_speed_drop": _safe_float(
                preview.get("r_speed_drop")),
            "planning_aware_motion_gate": _safe_float(
                preview.get("motion_gate")),
            "planning_aware_source_mask": preview.get("source_mask", ""),
            "planning_aware_planning_age_frames": _safe_float(
                preview.get("age_frames"),
                -1.0),
            "planning_aware_preview_points_used": int(
                preview.get("points_used", 0) or 0),
            "planning_aware_horizon_dt": _safe_float(
                preview.get("horizon_dt"),
                -1.0),
            "planning_aware_curvature_source": preview.get(
                "curvature_source",
                "missing"),
            "planning_aware_predicted_ay_source": preview.get(
                "predicted_ay_source",
                "missing"),
            "planning_aware_signed_lat_peak": _safe_float(
                preview.get("signed_lat_peak")),
            "planning_aware_max_preview_damper_extra": _safe_float(
                self.config.max_preview_damper_extra,
                0.035),
            "planning_aware_rate_limited_any": int(any(rate_limited)),
            "planning_aware_clamped_any": int(any(clamped)),
            "uniform_damper_cmd": _mean(combined_dampers),
            "uniform_damper_would": _mean(combined_dampers),
        })
        extras = tuple(preview.get("extras", ()) or ())
        for index, label in enumerate(_WHEEL_LABELS[:wheel_count]):
            extra = extras[index] if index < len(extras) else 0.0
            diagnostics["planning_feedforward_damper_add_%s" % label] = extra
            diagnostics["planning_aware_combined_damper_%s" % label] = (
                combined_dampers[index])
            diagnostics["planning_aware_rate_limited_%s" % label] = (
                rate_limited[index])
            diagnostics["planning_aware_clamped_%s" % label] = clamped[index]
            diagnostics["damper_%s" % label.upper()] = combined_dampers[index]
            diagnostics["spring_%s" % label.upper()] = 1.0
            diagnostics["skyhook_roll_damper_scale_%s" % label] = (
                feedback_dampers[index])
            diagnostics["final_damper_scale_%s" % label] = (
                combined_dampers[index])


def _empty_features(fallback_reason: str) -> Dict[str, Any]:
    return {
        "fallback_reason": fallback_reason,
        "times": (),
        "curvature_abs": (),
        "ay_abs": (),
        "decel": (),
        "brake": (),
        "steer_abs": (),
        "target_speed_drop_1s": 0.0,
        "curvature_source": "missing",
        "predicted_ay_source": "missing",
        "signed_lat_peak": 0.0,
        "source_mask": "none",
    }


def _dt(context: ControllerContext, default: float) -> float:
    value = _optional_finite_float(getattr(context, "dt", None))
    if value is None or value <= 0.0:
        value = _optional_finite_float(getattr(context.state, "dt", None))
    if value is None or value <= 0.0:
        value = default
    return max(float(value), 1.0e-9)


def _series(values: Sequence[Any], horizon_len: int) -> Tuple[Optional[float], ...]:
    raw = tuple(values or ())
    result = []
    for index in range(horizon_len):
        value = raw[index] if index < len(raw) else None
        result.append(_optional_finite_float(value))
    return tuple(result)


def _speed_derivatives(
    speeds: Sequence[Optional[float]],
    horizon_dt: float,
) -> Tuple[Optional[float], ...]:
    if not speeds:
        return ()
    values = []
    previous = speeds[0]
    for value in speeds:
        if value is None or previous is None:
            values.append(None)
        else:
            values.append((value - previous) / max(horizon_dt, _EPS))
        previous = value
    return tuple(values)


def _trajectory_curvature(
    trajectory_xy: Sequence[Tuple[float, float]],
    min_points: int,
) -> Tuple[float, ...]:
    points = []
    for value in trajectory_xy or ():
        try:
            x_value = _optional_finite_float(value[0])
            y_value = _optional_finite_float(value[1])
        except (TypeError, IndexError):
            continue
        if x_value is None or y_value is None:
            continue
        points.append((x_value, y_value))
    if len(points) < max(3, int(min_points)):
        return ()
    curvatures = []
    for index in range(len(points)):
        if index == 0:
            p0, p1, p2 = points[0], points[1], points[2]
        elif index == len(points) - 1:
            p0, p1, p2 = points[-3], points[-2], points[-1]
        else:
            p0, p1, p2 = points[index - 1], points[index], points[index + 1]
        curvatures.append(_three_point_curvature(p0, p1, p2))
    return tuple(curvatures)


def _three_point_curvature(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
) -> float:
    ax = p1[0] - p0[0]
    ay = p1[1] - p0[1]
    bx = p2[0] - p1[0]
    by = p2[1] - p1[1]
    cx = p2[0] - p0[0]
    cy = p2[1] - p0[1]
    twice_area = ax * by - ay * bx
    denom = (
        math.hypot(ax, ay) *
        math.hypot(bx, by) *
        math.hypot(cx, cy))
    if denom <= _EPS:
        return 0.0
    return 2.0 * twice_area / denom


def _signed_peak(
    predicted_ay: Sequence[Optional[float]],
    curvatures: Sequence[Optional[float]],
    indices: Sequence[int],
    fallback_speed: float,
) -> float:
    best = 0.0
    for index in indices:
        value = predicted_ay[index]
        if value is None and curvatures[index] is not None:
            value = fallback_speed * fallback_speed * curvatures[index]
        if value is not None and abs(value) > abs(best):
            best = value
    return _safe_float(best)


def _metadata_false(value: Any) -> bool:
    if value is False:
        return True
    if isinstance(value, str):
        return value.strip().lower() in ("false", "0", "no")
    if isinstance(value, (int, float)):
        return float(value) == 0.0
    return False


def _optional_finite_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _safe_float(value: Any, default: float = 0.0) -> float:
    result = _optional_finite_float(value)
    return default if result is None else result


def _sign(value: float) -> float:
    value = _safe_float(value)
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return -1.0
    return 0.0


def _mean(values: Sequence[float]) -> float:
    return sum(float(value) for value in values) / float(len(values)) if values else 0.0
