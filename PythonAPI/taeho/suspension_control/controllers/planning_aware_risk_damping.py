"""Planning-aware conservative damping controller for LEAD sidecar runs."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .base import (
    ControllerContext,
    ControllerOutput,
    PlanningInfo,
    SuspensionCommand,
    SuspensionController,
    VehicleState,
    WheelScale,
    clamp,
    finite_float,
)


_WHEEL_LABELS = ("FL", "FR", "RL", "RR")


@dataclass(frozen=True)
class PlanningAwareRiskDampingConfig:
    """Conservative planning-preview damper schedule configuration."""

    shadow_mode: bool = False
    output_mode: str = "uniform"
    damper_schedule_mode: str = "stiffen_only"
    damper_min: float = 1.0
    damper_max: float = 1.035
    max_uniform_extra: float = 0.035
    max_scale_rate_per_s: float = 0.08
    tau_rise: float = 0.30
    tau_fall: float = 0.90
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
    state_ay_low: float = 2.5
    state_ay_high: float = 5.5
    roll_rate_low: float = 4.0
    roll_rate_high: float = 14.0
    roll_angle_low: float = 2.0
    roll_angle_high: float = 5.5
    yaw_rate_low: float = 10.0
    yaw_rate_high: float = 40.0
    state_feedback_gain: float = 0.25
    target_speed_drop_window_s: float = 1.0
    decel_low: float = 1.5
    decel_high: float = 4.0
    brake_low: float = 0.20
    brake_high: float = 0.70
    speed_drop_low: float = 2.0
    speed_drop_high: float = 6.0
    enable_event_hold: bool = True
    event_delta_threshold: float = 0.0015
    event_force_refresh_s: float = 0.50
    use_bbox_for_brake_risk: bool = False
    wheel_count: int = 4
    default_dt: float = 0.05
    max_front_brake_bias: float = 0.008
    max_rear_lat_bias: float = 0.004

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> "PlanningAwareRiskDampingConfig":
        allowed = {item.name for item in fields(cls)}
        kwargs = {key: value for key, value in values.items() if key in allowed}
        return cls(**kwargs)


class PlanningAwareRiskDampingController(SuspensionController):
    """Damper-only uniform schedule driven by public planning preview fields."""

    name = "planning_aware_risk_damping"
    VERSION = "v2_phase1A_uniform"

    def __init__(
        self,
        config: Optional[PlanningAwareRiskDampingConfig | Mapping[str, Any]] = None,
    ):
        if config is None:
            self.config = PlanningAwareRiskDampingConfig()
        elif isinstance(config, PlanningAwareRiskDampingConfig):
            self.config = config
        else:
            self.config = PlanningAwareRiskDampingConfig.from_mapping(config)
        self._validate_config()
        self.reset()

    def reset(self, native_suspension: Any = None) -> None:
        del native_suspension
        self.risk_smooth = 0.0
        self.r_lat_smooth = 0.0
        self.r_brake_smooth = 0.0
        self.last_uniform_scale = 1.0
        self.last_front_scale = 1.0
        self.last_rear_scale = 1.0
        self.last_sent_uniform_scale = 1.0
        self.last_sent_front_scale = 1.0
        self.last_sent_rear_scale = 1.0
        self.last_event_sent_time: Optional[float] = None

    def compute(self, context: ControllerContext) -> ControllerOutput:
        try:
            return self._compute_impl(context)
        except Exception as error:
            self.reset()
            return self._fallback_output(
                context=context,
                fallback_reason="exception",
                exception_type=type(error).__name__)

    def _validate_config(self) -> None:
        cfg = self.config
        for name in (
                "damper_min",
                "damper_max",
                "max_uniform_extra",
                "max_scale_rate_per_s",
                "tau_rise",
                "tau_fall",
                "preview_horizon_s",
                "near_window_s",
                "far_window_s",
                "far_weight",
                "single_point_weight",
                "min_horizon_dt",
                "max_horizon_dt",
                "speed_cap_mps",
                "motion_gate_low",
                "motion_gate_high",
                "ay_abs_cap",
                "ay_low",
                "ay_high",
                "curvature_abs_cap",
                "curvature_low",
                "curvature_high",
                "steer_low",
                "steer_high",
                "state_ay_low",
                "state_ay_high",
                "roll_rate_low",
                "roll_rate_high",
                "roll_angle_low",
                "roll_angle_high",
                "yaw_rate_low",
                "yaw_rate_high",
                "state_feedback_gain",
                "target_speed_drop_window_s",
                "decel_low",
                "decel_high",
                "brake_low",
                "brake_high",
                "speed_drop_low",
                "speed_drop_high",
                "event_delta_threshold",
                "event_force_refresh_s",
                "default_dt",
                "max_front_brake_bias",
                "max_rear_lat_bias"):
            finite_float(getattr(cfg, name), name)
        if str(cfg.output_mode) not in ("uniform", "front_rear"):
            raise ValueError("output_mode must be uniform or front_rear")
        if str(cfg.damper_schedule_mode) not in ("stiffen_only", "centered_range"):
            raise ValueError(
                "damper_schedule_mode must be stiffen_only or centered_range")
        if cfg.damper_min <= 0.0:
            raise ValueError("damper_min must be positive")
        if cfg.damper_max < cfg.damper_min:
            raise ValueError("damper_max must be >= damper_min")
        if cfg.max_uniform_extra < 0.0:
            raise ValueError("max_uniform_extra must be nonnegative")
        if cfg.max_scale_rate_per_s < 0.0:
            raise ValueError("max_scale_rate_per_s must be nonnegative")
        if cfg.preview_horizon_s < 0.0:
            raise ValueError("preview_horizon_s must be nonnegative")
        if cfg.near_window_s < 0.0:
            raise ValueError("near_window_s must be nonnegative")
        if cfg.far_window_s < cfg.near_window_s:
            raise ValueError("far_window_s must be >= near_window_s")
        if cfg.max_planning_age_frames < 0:
            raise ValueError("max_planning_age_frames must be nonnegative")
        if cfg.min_horizon_dt <= 0.0:
            raise ValueError("min_horizon_dt must be positive")
        if cfg.max_horizon_dt < cfg.min_horizon_dt:
            raise ValueError("max_horizon_dt must be >= min_horizon_dt")
        if int(cfg.wheel_count) != 4:
            raise ValueError("PlanningAwareRiskDampingController requires 4 wheels")

    def _compute_impl(self, context: ControllerContext) -> ControllerOutput:
        cfg = self.config
        planning = getattr(context, "planning", None)
        state = getattr(context, "state", VehicleState())

        early_reason = self._early_fallback_reason(context, state, planning)
        if early_reason:
            self.reset()
            return self._fallback_output(context, early_reason)

        assert isinstance(planning, PlanningInfo)
        horizon_dt = float(planning.horizon_dt)
        state_frame = int(float(state.frame))
        planning_frame = int(float(planning.frame))
        planning_age_frames = state_frame - planning_frame
        dt = self._dt(context)

        features = self._extract_features(state, planning, horizon_dt)
        if features["fallback_reason"]:
            self.reset()
            return self._fallback_output(
                context,
                str(features["fallback_reason"]))

        motion_gate = smoothstep(
            float(state.speed),
            cfg.motion_gate_low,
            cfg.motion_gate_high)

        r_ay_preview = robust_preview_risk(
            features["ay_abs"],
            features["times"],
            cfg.ay_low,
            cfg.ay_high,
            cfg.near_window_s,
            cfg.far_window_s,
            cfg.far_weight,
            cfg.single_point_weight)
        r_kappa_preview = robust_preview_risk(
            features["curvature_abs"],
            features["times"],
            cfg.curvature_low,
            cfg.curvature_high,
            cfg.near_window_s,
            cfg.far_window_s,
            cfg.far_weight,
            cfg.single_point_weight)
        r_steer_preview = robust_preview_risk(
            features["steer_abs"],
            features["times"],
            cfg.steer_low,
            cfg.steer_high,
            cfg.near_window_s,
            cfg.far_window_s,
            cfg.far_weight,
            cfg.single_point_weight)
        r_lat_preview = motion_gate * max(
            r_ay_preview,
            0.65 * r_kappa_preview,
            0.25 * r_steer_preview)

        r_state_ay = smoothstep(
            abs(float(state.local_ay)),
            cfg.state_ay_low,
            cfg.state_ay_high)
        r_roll_rate = smoothstep(
            abs(float(state.roll_rate)),
            cfg.roll_rate_low,
            cfg.roll_rate_high)
        r_roll_angle = smoothstep(
            abs(float(state.roll)),
            cfg.roll_angle_low,
            cfg.roll_angle_high)
        r_yaw_rate = smoothstep(
            abs(float(state.yaw_rate)),
            cfg.yaw_rate_low,
            cfg.yaw_rate_high)
        r_state_raw = max(
            r_state_ay,
            0.70 * r_roll_rate,
            0.45 * r_roll_angle,
            0.25 * r_yaw_rate)

        r_pred_decel = robust_preview_risk(
            features["decel"],
            features["times"],
            cfg.decel_low,
            cfg.decel_high,
            cfg.near_window_s,
            cfg.far_window_s,
            cfg.far_weight,
            cfg.single_point_weight)
        r_brake_preview = robust_preview_risk(
            features["brake"],
            features["times"],
            cfg.brake_low,
            cfg.brake_high,
            cfg.near_window_s,
            cfg.far_window_s,
            cfg.far_weight,
            cfg.single_point_weight)
        r_current_brake = 0.40 * smoothstep(
            float(state.brake),
            cfg.brake_low,
            cfg.brake_high)
        r_speed_drop = smoothstep(
            float(features["target_speed_drop_1s"]),
            cfg.speed_drop_low,
            cfg.speed_drop_high)
        r_brake_preview_combined = motion_gate * max(
            r_pred_decel,
            r_brake_preview,
            r_current_brake,
            0.35 * r_speed_drop)

        preview_gate = max(r_lat_preview, r_brake_preview_combined)
        r_state_trim = cfg.state_feedback_gain * preview_gate * r_state_raw
        risk_preview = max(r_lat_preview, 0.65 * r_brake_preview_combined)
        risk_raw = clip(risk_preview + r_state_trim, 0.0, 1.0)

        if not math.isfinite(risk_raw):
            self.reset()
            return self._fallback_output(context, "nonfinite_command")

        self.risk_smooth = ema_update(
            self.risk_smooth,
            risk_raw,
            dt,
            cfg.tau_rise,
            cfg.tau_fall)
        self.r_lat_smooth = ema_update(
            self.r_lat_smooth,
            r_lat_preview,
            dt,
            cfg.tau_rise,
            cfg.tau_fall)
        self.r_brake_smooth = ema_update(
            self.r_brake_smooth,
            r_brake_preview_combined,
            dt,
            cfg.tau_rise,
            cfg.tau_fall)

        uniform_desired = self._desired_scale(
            risk=self.risk_smooth,
            extra=cfg.max_uniform_extra)
        max_delta = max(0.0, cfg.max_scale_rate_per_s) * dt
        uniform_limited = rate_limit(
            self.last_uniform_scale,
            uniform_desired,
            max_delta)
        rate_limit_active_uniform = int(not math.isclose(
            uniform_limited,
            uniform_desired,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12))
        self.last_uniform_scale = uniform_limited

        front_limited = uniform_limited
        rear_limited = uniform_limited
        rate_limit_active_front = 0
        rate_limit_active_rear = 0
        if cfg.output_mode == "front_rear":
            front_desired = self._desired_scale(
                risk=self.risk_smooth,
                extra=cfg.max_uniform_extra,
                additive_bias=cfg.max_front_brake_bias * self.r_brake_smooth)
            rear_desired = self._desired_scale(
                risk=self.risk_smooth,
                extra=cfg.max_uniform_extra,
                additive_bias=cfg.max_rear_lat_bias * self.r_lat_smooth)
            front_limited = rate_limit(
                self.last_front_scale,
                front_desired,
                max_delta)
            rear_limited = rate_limit(
                self.last_rear_scale,
                rear_desired,
                max_delta)
            rate_limit_active_front = int(not math.isclose(
                front_limited,
                front_desired,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12))
            rate_limit_active_rear = int(not math.isclose(
                rear_limited,
                rear_desired,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12))
            self.last_front_scale = front_limited
            self.last_rear_scale = rear_limited

        would_uniform, would_front, would_rear, event_held = self._event_command(
            elapsed_seconds=float(state.elapsed_seconds),
            uniform_limited=uniform_limited,
            front_limited=front_limited,
            rear_limited=rear_limited)

        if cfg.output_mode == "uniform":
            would_front = would_uniform
            would_rear = would_uniform

        actual_uniform = 1.0 if cfg.shadow_mode else would_uniform
        actual_front = 1.0 if cfg.shadow_mode else would_front
        actual_rear = 1.0 if cfg.shadow_mode else would_rear

        command = self._command(actual_front, actual_rear)
        if not _command_finite(command):
            self.reset()
            return self._fallback_output(context, "nonfinite_command")

        diagnostics = self._diagnostics(
            context=context,
            planning=planning,
            planning_age_frames=planning_age_frames,
            planning_valid=1,
            fallback_active=0,
            fallback_reason="none",
            features=features,
            motion_gate=motion_gate,
            r_ay_preview=r_ay_preview,
            r_kappa_preview=r_kappa_preview,
            r_steer_preview=r_steer_preview,
            r_lat_preview=r_lat_preview,
            r_state_ay=r_state_ay,
            r_roll_rate=r_roll_rate,
            r_roll_angle=r_roll_angle,
            r_yaw_rate=r_yaw_rate,
            r_state_raw=r_state_raw,
            r_state_trim=r_state_trim,
            r_pred_decel=r_pred_decel,
            r_brake_preview=r_brake_preview,
            r_current_brake=r_current_brake,
            r_speed_drop=r_speed_drop,
            r_brake_preview_combined=r_brake_preview_combined,
            risk_preview=risk_preview,
            risk_raw=risk_raw,
            uniform_damper_desired=uniform_desired,
            uniform_damper_limited=uniform_limited,
            uniform_damper_cmd=actual_uniform,
            uniform_damper_would=would_uniform,
            front_damper_cmd=actual_front,
            rear_damper_cmd=actual_rear,
            front_damper_would=would_front,
            rear_damper_would=would_rear,
            rate_limit_active_uniform=rate_limit_active_uniform,
            event_held=event_held,
            command=command)
        diagnostics["rate_limit_active_front"] = rate_limit_active_front
        diagnostics["rate_limit_active_rear"] = rate_limit_active_rear
        return ControllerOutput(command=command, diagnostics=diagnostics)

    def _desired_scale(
        self,
        risk: float,
        extra: float,
        additive_bias: float = 0.0,
    ) -> float:
        cfg = self.config
        risk_value = clip(float(risk), 0.0, 1.0)
        if cfg.damper_schedule_mode == "centered_range":
            desired = 1.0 + float(extra) * (2.0 * risk_value - 1.0)
        else:
            desired = 1.0 + float(extra) * risk_value
        return clip(
            desired + float(additive_bias),
            cfg.damper_min,
            cfg.damper_max)

    def _early_fallback_reason(
        self,
        context: ControllerContext,
        state: VehicleState,
        planning: Optional[PlanningInfo],
    ) -> str:
        cfg = self.config
        if planning is None:
            return "no_planning"
        if not bool(getattr(planning, "available", False)):
            return "planning_unavailable"
        state_frame = _optional_finite_float(getattr(state, "frame", None))
        planning_frame = _optional_finite_float(getattr(planning, "frame", None))
        if state_frame is None or planning_frame is None or planning_frame < 0.0:
            return "missing_frame"
        if int(state_frame) - int(planning_frame) > int(cfg.max_planning_age_frames):
            return "stale_planning"
        horizon_dt = _optional_finite_float(getattr(planning, "horizon_dt", None))
        if (
                horizon_dt is None or
                horizon_dt < cfg.min_horizon_dt or
                horizon_dt > cfg.max_horizon_dt):
            return "bad_horizon_dt"
        metadata = dict(getattr(planning, "metadata", {}) or {})
        if _metadata_false(metadata.get("valid_prediction", None)):
            return "metadata_invalid_prediction"
        for name in (
                "speed",
                "local_ay",
                "roll",
                "roll_rate",
                "yaw_rate",
                "brake",
                "steer"):
            if _optional_finite_float(getattr(state, name, None)) is None:
                return "nonfinite_state"
        if _optional_finite_float(getattr(context, "dt", cfg.default_dt)) is None:
            if _optional_finite_float(getattr(state, "dt", cfg.default_dt)) is None:
                return "nonfinite_state"
        return ""

    def _extract_features(
        self,
        state: VehicleState,
        planning: PlanningInfo,
        horizon_dt: float,
    ) -> Dict[str, Any]:
        cfg = self.config
        lengths = [
            len(getattr(planning, "predicted_ay", ()) or ()),
            len(getattr(planning, "curvature", ()) or ()),
            len(getattr(planning, "trajectory_xy", ()) or ()),
            len(getattr(planning, "target_speed", ()) or ()),
            len(getattr(planning, "steer", ()) or ()),
            len(getattr(planning, "brake", ()) or ()),
            len(getattr(planning, "predicted_ax", ()) or ()),
        ]
        point_count = len(getattr(planning, "points", ()) or ())
        if point_count:
            lengths.append(point_count)
        horizon_len = max(lengths or [0])
        if horizon_len <= 0:
            return self._empty_features("no_valid_preview_signal")

        times_all = tuple(index * horizon_dt for index in range(horizon_len))
        indices = tuple(
            index for index, value in enumerate(times_all)
            if 0.0 <= value <= cfg.preview_horizon_s)
        if not indices:
            return self._empty_features("no_valid_preview_signal")

        target_speed_all, target_speed_has_negative = self._target_speeds(
            state,
            planning,
            horizon_len,
            indices)
        if target_speed_has_negative:
            return self._empty_features("negative_target_speed")

        curvature_all, curvature_source = self._curvature(
            planning,
            horizon_len)
        predicted_ay_all, predicted_ay_source = self._predicted_ay(
            planning,
            target_speed_all,
            curvature_all,
            curvature_source,
            horizon_len)
        predicted_ax_all = _series(
            getattr(planning, "predicted_ax", ()),
            horizon_len)
        brake_all = _series(getattr(planning, "brake", ()), horizon_len)
        steer_all = _series(getattr(planning, "steer", ()), horizon_len)

        times = tuple(times_all[index] for index in indices)
        curvature = tuple(
            clamp(abs(curvature_all[index]), 0.0, cfg.curvature_abs_cap)
            if curvature_all[index] is not None else 0.0
            for index in indices)
        ay_abs = tuple(
            clamp(abs(predicted_ay_all[index]), 0.0, cfg.ay_abs_cap)
            if predicted_ay_all[index] is not None else 0.0
            for index in indices)
        decel = tuple(
            max(0.0, -predicted_ax_all[index])
            if predicted_ax_all[index] is not None else 0.0
            for index in indices)
        brake = tuple(
            clamp(brake_all[index], 0.0, 1.0)
            if brake_all[index] is not None else 0.0
            for index in indices)
        steer_abs = tuple(
            abs(steer_all[index])
            if steer_all[index] is not None else 0.0
            for index in indices)
        target_speeds = tuple(
            target_speed_all[index]
            for index in indices
            if target_speed_all[index] is not None)

        valid_sources = []
        if any(value is not None for value in predicted_ay_all):
            valid_sources.append("ay")
        if any(value is not None for value in curvature_all):
            valid_sources.append("kappa")
        if any(value is not None for value in predicted_ax_all):
            valid_sources.append("decel")
        if any(value is not None for value in brake_all):
            valid_sources.append("brake")
        if any(value is not None for value in steer_all):
            valid_sources.append("steer")
        if target_speeds:
            valid_sources.append("drop")
        if not valid_sources:
            return self._empty_features("no_valid_preview_signal")

        target_speed_min_1s = 0.0
        target_speed_drop_1s = 0.0
        speeds_1s = [
            target_speed_all[index]
            for index in indices
            if (
                times_all[index] <= cfg.target_speed_drop_window_s and
                target_speed_all[index] is not None)
        ]
        if speeds_1s:
            target_speed_min_1s = min(speeds_1s)
            target_speed_drop_1s = max(0.0, float(state.speed) - target_speed_min_1s)

        source_count = max(1, len(valid_sources))
        risk_confidence = min(1.0, 0.50 + 0.15 * source_count)
        if len(indices) == 1:
            risk_confidence *= cfg.single_point_weight
        if curvature_source == "trajectory":
            risk_confidence *= 0.90
        if predicted_ay_source == "v2kappa":
            risk_confidence *= 0.90
        risk_confidence = clip(risk_confidence, 0.0, 1.0)

        extra = dict(getattr(planning, "extra", {}) or {})
        bbox = _bbox_summary(extra)

        return {
            "fallback_reason": "",
            "times": times,
            "preview_points_used": len(indices),
            "curvature_abs": curvature,
            "curvature_source": curvature_source,
            "curvature_abs_max": max(curvature) if curvature else 0.0,
            "curvature_abs_near_topk": _topk_window_value(
                curvature,
                times,
                0.0,
                cfg.near_window_s),
            "ay_abs": ay_abs,
            "predicted_ay_source": predicted_ay_source,
            "predicted_ay_abs_max": max(ay_abs) if ay_abs else 0.0,
            "predicted_ay_abs_near_topk": _topk_window_value(
                ay_abs,
                times,
                0.0,
                cfg.near_window_s),
            "predicted_ax_min": min(
                (
                    predicted_ax_all[index]
                    for index in indices
                    if predicted_ax_all[index] is not None),
                default=0.0),
            "decel": decel,
            "brake": brake,
            "brake_preview_max": max(brake) if brake else 0.0,
            "steer_abs": steer_abs,
            "steer_preview_abs_max": max(steer_abs) if steer_abs else 0.0,
            "target_speed_min_1s": target_speed_min_1s,
            "target_speed_drop_1s": target_speed_drop_1s,
            "risk_source_mask": ",".join(valid_sources),
            "risk_confidence": risk_confidence,
            "bbox_num_boxes": bbox["bbox_num_boxes"],
            "bbox_num_vehicle_boxes": bbox["bbox_num_vehicle_boxes"],
            "bbox_num_pedestrian_boxes": bbox["bbox_num_pedestrian_boxes"],
            "bbox_min_forward_distance_m": bbox["bbox_min_forward_distance_m"],
        }

    def _empty_features(self, fallback_reason: str) -> Dict[str, Any]:
        return {
            "fallback_reason": fallback_reason,
            "times": (),
            "preview_points_used": 0,
            "curvature_abs": (),
            "curvature_source": "missing",
            "curvature_abs_max": 0.0,
            "curvature_abs_near_topk": 0.0,
            "ay_abs": (),
            "predicted_ay_source": "missing",
            "predicted_ay_abs_max": 0.0,
            "predicted_ay_abs_near_topk": 0.0,
            "predicted_ax_min": 0.0,
            "decel": (),
            "brake": (),
            "brake_preview_max": 0.0,
            "steer_abs": (),
            "steer_preview_abs_max": 0.0,
            "target_speed_min_1s": 0.0,
            "target_speed_drop_1s": 0.0,
            "risk_source_mask": "none",
            "risk_confidence": 0.0,
            "bbox_num_boxes": 0.0,
            "bbox_num_vehicle_boxes": 0.0,
            "bbox_num_pedestrian_boxes": 0.0,
            "bbox_min_forward_distance_m": 0.0,
        }

    def _target_speeds(
        self,
        state: VehicleState,
        planning: PlanningInfo,
        horizon_len: int,
        indices: Sequence[int],
    ) -> Tuple[Tuple[Optional[float], ...], bool]:
        cfg = self.config
        raw = getattr(planning, "target_speed", ()) or ()
        values = _series(raw, horizon_len)
        has_negative = False
        for index in indices:
            value = values[index]
            if value is not None and value < 0.0:
                has_negative = True
                break
        if len(raw) == 1:
            only_value = _optional_finite_float(raw[0])
            if only_value is not None and only_value < 0.0:
                return tuple(None for _ in range(horizon_len)), True
            if only_value is not None:
                value = clamp(only_value, 0.0, cfg.speed_cap_mps)
                return tuple(value for _ in range(horizon_len)), False
        if any(value is not None for value in values):
            filled = tuple(
                clamp(value, 0.0, cfg.speed_cap_mps)
                if value is not None and value >= 0.0 else None
                for value in values)
            return filled, has_negative
        fallback_speed = clamp(float(state.speed), 0.0, cfg.speed_cap_mps)
        return tuple(fallback_speed for _ in range(horizon_len)), False

    def _curvature(
        self,
        planning: PlanningInfo,
        horizon_len: int,
    ) -> Tuple[Tuple[Optional[float], ...], str]:
        raw = _series(getattr(planning, "curvature", ()), horizon_len)
        if any(value is not None for value in raw):
            return raw, "planning"
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
        speed: Sequence[Optional[float]],
        curvature: Sequence[Optional[float]],
        curvature_source: str,
        horizon_len: int,
    ) -> Tuple[Tuple[Optional[float], ...], str]:
        raw = _series(getattr(planning, "predicted_ay", ()), horizon_len)
        if any(value is not None for value in raw):
            return raw, "planning"
        if curvature_source == "missing":
            return tuple(None for _ in range(horizon_len)), "missing"
        values: List[Optional[float]] = []
        for speed_value, curvature_value in zip(speed, curvature):
            if speed_value is None or curvature_value is None:
                values.append(None)
            else:
                values.append(speed_value * speed_value * curvature_value)
        if any(value is not None for value in values):
            return tuple(values), "v2kappa"
        return tuple(None for _ in range(horizon_len)), "missing"

    def _dt(self, context: ControllerContext) -> float:
        cfg = self.config
        dt = _optional_finite_float(getattr(context, "dt", None))
        if dt is None or dt <= 0.0:
            dt = _optional_finite_float(getattr(context.state, "dt", None))
        if dt is None or dt <= 0.0:
            dt = cfg.default_dt
        return max(float(dt), 1.0e-9)

    def _event_command(
        self,
        elapsed_seconds: float,
        uniform_limited: float,
        front_limited: float,
        rear_limited: float,
    ) -> Tuple[float, float, float, int]:
        cfg = self.config
        if self.last_event_sent_time is None:
            self.last_sent_uniform_scale = uniform_limited
            self.last_sent_front_scale = front_limited
            self.last_sent_rear_scale = rear_limited
            self.last_event_sent_time = elapsed_seconds
            return uniform_limited, front_limited, rear_limited, 0

        if cfg.output_mode == "uniform":
            change = abs(uniform_limited - self.last_sent_uniform_scale)
        else:
            change = max(
                abs(front_limited - self.last_sent_front_scale),
                abs(rear_limited - self.last_sent_rear_scale))
        time_since_sent = elapsed_seconds - self.last_event_sent_time
        force_refresh = (
            not math.isfinite(time_since_sent) or
            time_since_sent >= cfg.event_force_refresh_s)
        if (
                cfg.enable_event_hold and
                not force_refresh and
                change < cfg.event_delta_threshold):
            return (
                self.last_sent_uniform_scale,
                self.last_sent_front_scale,
                self.last_sent_rear_scale,
                1)

        self.last_sent_uniform_scale = uniform_limited
        self.last_sent_front_scale = front_limited
        self.last_sent_rear_scale = rear_limited
        self.last_event_sent_time = elapsed_seconds
        return uniform_limited, front_limited, rear_limited, 0

    def _command(self, front_damper: float, rear_damper: float) -> SuspensionCommand:
        return SuspensionCommand((
            WheelScale(1.0, front_damper),
            WheelScale(1.0, front_damper),
            WheelScale(1.0, rear_damper),
            WheelScale(1.0, rear_damper),
        )).validate(expected_wheels=4)

    def _fallback_output(
        self,
        context: Optional[ControllerContext],
        fallback_reason: str,
        exception_type: str = "",
    ) -> ControllerOutput:
        command = SuspensionCommand.identity(4).validate(expected_wheels=4)
        planning = getattr(context, "planning", None) if context is not None else None
        state = getattr(context, "state", VehicleState()) if context is not None else VehicleState()
        planning_age_frames = self._planning_age(state, planning)
        diagnostics = self._diagnostics(
            context=context,
            planning=planning,
            planning_age_frames=planning_age_frames,
            planning_valid=0,
            fallback_active=1,
            fallback_reason=fallback_reason,
            features=self._empty_features(fallback_reason),
            motion_gate=0.0,
            r_ay_preview=0.0,
            r_kappa_preview=0.0,
            r_steer_preview=0.0,
            r_lat_preview=0.0,
            r_state_ay=0.0,
            r_roll_rate=0.0,
            r_roll_angle=0.0,
            r_yaw_rate=0.0,
            r_state_raw=0.0,
            r_state_trim=0.0,
            r_pred_decel=0.0,
            r_brake_preview=0.0,
            r_current_brake=0.0,
            r_speed_drop=0.0,
            r_brake_preview_combined=0.0,
            risk_preview=0.0,
            risk_raw=0.0,
            uniform_damper_desired=1.0,
            uniform_damper_limited=1.0,
            uniform_damper_cmd=1.0,
            uniform_damper_would=1.0,
            front_damper_cmd=1.0,
            rear_damper_cmd=1.0,
            front_damper_would=1.0,
            rear_damper_would=1.0,
            rate_limit_active_uniform=0,
            event_held=0,
            command=command)
        diagnostics["exception_type"] = exception_type
        return ControllerOutput(command=command, diagnostics=diagnostics)

    def _diagnostics(
        self,
        context: Optional[ControllerContext],
        planning: Optional[PlanningInfo],
        planning_age_frames: float,
        planning_valid: int,
        fallback_active: int,
        fallback_reason: str,
        features: Mapping[str, Any],
        motion_gate: float,
        r_ay_preview: float,
        r_kappa_preview: float,
        r_steer_preview: float,
        r_lat_preview: float,
        r_state_ay: float,
        r_roll_rate: float,
        r_roll_angle: float,
        r_yaw_rate: float,
        r_state_raw: float,
        r_state_trim: float,
        r_pred_decel: float,
        r_brake_preview: float,
        r_current_brake: float,
        r_speed_drop: float,
        r_brake_preview_combined: float,
        risk_preview: float,
        risk_raw: float,
        uniform_damper_desired: float,
        uniform_damper_limited: float,
        uniform_damper_cmd: float,
        uniform_damper_would: float,
        front_damper_cmd: float,
        rear_damper_cmd: float,
        front_damper_would: float,
        rear_damper_would: float,
        rate_limit_active_uniform: int,
        event_held: int,
        command: SuspensionCommand,
    ) -> Dict[str, Any]:
        cfg = self.config
        state = getattr(context, "state", VehicleState()) if context is not None else VehicleState()
        wheels = command.wheels
        planning_frame = (
            _optional_finite_float(getattr(planning, "frame", None))
            if planning is not None else None)
        planning_horizon_dt = (
            _optional_finite_float(getattr(planning, "horizon_dt", None))
            if planning is not None else None)
        diagnostics: Dict[str, Any] = {
            "controller": self.name,
            "controller_name": self.name,
            "controller_version": self.VERSION,
            "spring_scale": 1.0,
            "damper_scale": _finite_or_default(uniform_damper_cmd, 1.0),
            "shadow_mode": int(cfg.shadow_mode),
            "output_mode": str(cfg.output_mode),
            "damper_schedule_mode": str(cfg.damper_schedule_mode),
            "planning_available": int(bool(
                planning is not None and getattr(planning, "available", False))),
            "planning_valid": int(planning_valid),
            "planning_frame": (
                -1.0 if planning_frame is None else planning_frame),
            "planning_age_frames": _finite_or_default(
                planning_age_frames,
                -1.0),
            "planning_horizon_dt": (
                -1.0 if planning_horizon_dt is None else planning_horizon_dt),
            "fallback_active": int(fallback_active),
            "fallback_reason": fallback_reason,
            "suspension_state_available": int(bool(
                context is not None and
                getattr(context, "suspension_state", None) is not None and
                getattr(context, "suspension_state_valid", False))),
            "preview_points_used": int(features.get("preview_points_used", 0)),
            "motion_gate": _finite_or_default(motion_gate),
            "curvature_source": features.get("curvature_source", "missing"),
            "curvature_abs_max": _finite_or_default(
                features.get("curvature_abs_max", 0.0)),
            "curvature_abs_near_topk": _finite_or_default(
                features.get("curvature_abs_near_topk", 0.0)),
            "predicted_ay_source": features.get("predicted_ay_source", "missing"),
            "predicted_ay_abs_max": _finite_or_default(
                features.get("predicted_ay_abs_max", 0.0)),
            "predicted_ay_abs_near_topk": _finite_or_default(
                features.get("predicted_ay_abs_near_topk", 0.0)),
            "predicted_ax_min": _finite_or_default(
                features.get("predicted_ax_min", 0.0)),
            "brake_preview_max": _finite_or_default(
                features.get("brake_preview_max", 0.0)),
            "target_speed_min_1s": _finite_or_default(
                features.get("target_speed_min_1s", 0.0)),
            "target_speed_drop_1s": _finite_or_default(
                features.get("target_speed_drop_1s", 0.0)),
            "steer_preview_abs_max": _finite_or_default(
                features.get("steer_preview_abs_max", 0.0)),
            "risk_source_mask": features.get("risk_source_mask", "none"),
            "risk_confidence": _finite_or_default(
                features.get("risk_confidence", 0.0)),
            "r_ay_preview": _finite_or_default(r_ay_preview),
            "r_kappa_preview": _finite_or_default(r_kappa_preview),
            "r_steer_preview": _finite_or_default(r_steer_preview),
            "r_lat_preview": _finite_or_default(r_lat_preview),
            "r_state_ay": _finite_or_default(r_state_ay),
            "r_roll_rate": _finite_or_default(r_roll_rate),
            "r_roll_angle": _finite_or_default(r_roll_angle),
            "r_yaw_rate": _finite_or_default(r_yaw_rate),
            "r_state_raw": _finite_or_default(r_state_raw),
            "r_state_trim": _finite_or_default(r_state_trim),
            "r_pred_decel": _finite_or_default(r_pred_decel),
            "r_brake_preview": _finite_or_default(r_brake_preview),
            "r_current_brake": _finite_or_default(r_current_brake),
            "r_speed_drop": _finite_or_default(r_speed_drop),
            "r_brake_preview_combined": _finite_or_default(
                r_brake_preview_combined),
            "risk_preview": _finite_or_default(risk_preview),
            "risk_raw": _finite_or_default(risk_raw),
            "risk_smooth": _finite_or_default(self.risk_smooth),
            "r_lat_smooth": _finite_or_default(self.r_lat_smooth),
            "r_brake_smooth": _finite_or_default(self.r_brake_smooth),
            "uniform_damper_desired": _finite_or_default(
                uniform_damper_desired,
                1.0),
            "uniform_damper_limited": _finite_or_default(
                uniform_damper_limited,
                1.0),
            "uniform_damper_cmd": _finite_or_default(uniform_damper_cmd, 1.0),
            "uniform_damper_would": _finite_or_default(
                uniform_damper_would,
                1.0),
            "front_damper_cmd": _finite_or_default(front_damper_cmd, 1.0),
            "rear_damper_cmd": _finite_or_default(rear_damper_cmd, 1.0),
            "front_damper_would": _finite_or_default(
                front_damper_would,
                1.0),
            "rear_damper_would": _finite_or_default(rear_damper_would, 1.0),
            "rate_limit_active_uniform": int(rate_limit_active_uniform),
            "event_held": int(event_held),
            "bbox_num_boxes": _finite_or_default(
                features.get("bbox_num_boxes", 0.0)),
            "bbox_num_vehicle_boxes": _finite_or_default(
                features.get("bbox_num_vehicle_boxes", 0.0)),
            "bbox_num_pedestrian_boxes": _finite_or_default(
                features.get("bbox_num_pedestrian_boxes", 0.0)),
            "bbox_min_forward_distance_m": _finite_or_default(
                features.get("bbox_min_forward_distance_m", 0.0)),
            "state_speed": _finite_or_default(getattr(state, "speed", 0.0)),
            "state_local_ay": _finite_or_default(
                getattr(state, "local_ay", 0.0)),
            "state_roll": _finite_or_default(getattr(state, "roll", 0.0)),
            "state_roll_rate": _finite_or_default(
                getattr(state, "roll_rate", 0.0)),
            "state_pitch": _finite_or_default(getattr(state, "pitch", 0.0)),
            "state_pitch_rate": _finite_or_default(
                getattr(state, "pitch_rate", 0.0)),
            "state_yaw_rate": _finite_or_default(
                getattr(state, "yaw_rate", 0.0)),
            "state_steer": _finite_or_default(getattr(state, "steer", 0.0)),
            "state_brake": _finite_or_default(getattr(state, "brake", 0.0)),
        }
        for label, wheel in zip(_WHEEL_LABELS, wheels):
            diagnostics["spring_%s" % label] = _finite_or_default(
                wheel.spring_scale,
                1.0)
            diagnostics["damper_%s" % label] = _finite_or_default(
                wheel.damper_scale,
                1.0)
        return diagnostics

    def _planning_age(
        self,
        state: VehicleState,
        planning: Optional[PlanningInfo],
    ) -> float:
        if planning is None:
            return -1.0
        state_frame = _optional_finite_float(getattr(state, "frame", None))
        planning_frame = _optional_finite_float(getattr(planning, "frame", None))
        if state_frame is None or planning_frame is None:
            return -1.0
        return float(int(state_frame) - int(planning_frame))


def smoothstep(value: float, low: float, high: float) -> float:
    if not math.isfinite(value):
        return 0.0
    if high <= low:
        return 1.0 if value >= high else 0.0
    t = clip((value - low) / (high - low), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def clip(value: float, low: float, high: float) -> float:
    return clamp(value, low, high)


def ema_update(
    previous: float,
    raw: float,
    dt: float,
    tau_rise: float,
    tau_fall: float,
) -> float:
    tau = tau_rise if raw > previous else tau_fall
    tau = max(float(tau), 1.0e-6)
    alpha = clip(dt / (tau + dt), 0.0, 1.0)
    return previous + alpha * (raw - previous)


def rate_limit(previous: float, desired: float, max_delta: float) -> float:
    return previous + clip(desired - previous, -max_delta, max_delta)


def topk_mean(values: Iterable[float], k: int = 2) -> float:
    finite_values = sorted(
        (float(value) for value in values if _is_finite_number(value)),
        reverse=True)
    if not finite_values:
        return 0.0
    count = max(1, min(int(k), len(finite_values)))
    return sum(finite_values[:count]) / float(count)


def robust_window_risk(
    values_abs: Sequence[float],
    times: Sequence[float],
    low: float,
    high: float,
    window_start: float,
    window_end: float,
    single_point_weight: float,
) -> float:
    risks = [
        smoothstep(float(value), low, high)
        for value, time_value in zip(values_abs, times)
        if (
            window_start <= time_value <= window_end and
            _is_finite_number(value))
    ]
    if not risks:
        return 0.0
    if len(risks) == 1:
        return single_point_weight * risks[0]
    return topk_mean(risks, k=2)


def robust_preview_risk(
    values_abs: Sequence[float],
    times: Sequence[float],
    low: float,
    high: float,
    near_window_s: float,
    far_window_s: float,
    far_weight: float,
    single_point_weight: float,
) -> float:
    near = robust_window_risk(
        values_abs,
        times,
        low,
        high,
        0.0,
        near_window_s,
        single_point_weight)
    far = robust_window_risk(
        values_abs,
        times,
        low,
        high,
        near_window_s,
        far_window_s,
        single_point_weight)
    return clip(max(near, far_weight * far), 0.0, 1.0)


def _metadata_false(value: Any) -> bool:
    if value is False:
        return True
    if isinstance(value, str):
        return value.strip().lower() in ("false", "0", "no")
    if isinstance(value, (int, float)):
        return float(value) == 0.0
    return False


def _is_finite_number(value: Any) -> bool:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(result)


def _optional_finite_float(value: Any) -> Optional[float]:
    if not _is_finite_number(value):
        return None
    return float(value)


def _finite_or_default(value: Any, default: float = 0.0) -> float:
    result = _optional_finite_float(value)
    return default if result is None else result


def _series(values: Sequence[Any], horizon_len: int) -> Tuple[Optional[float], ...]:
    raw = tuple(values or ())
    result: List[Optional[float]] = []
    for index in range(horizon_len):
        value = raw[index] if index < len(raw) else None
        result.append(_optional_finite_float(value))
    return tuple(result)


def _trajectory_curvature(
    trajectory_xy: Sequence[Tuple[float, float]],
    min_points: int,
) -> Tuple[float, ...]:
    points: List[Tuple[float, float]] = []
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
    a = math.hypot(ax, ay)
    b = math.hypot(bx, by)
    c = math.hypot(cx, cy)
    area2 = ax * cy - ay * cx
    denominator = max(a * b * c, 1.0e-9)
    return 2.0 * area2 / denominator


def _topk_window_value(
    values: Sequence[float],
    times: Sequence[float],
    start: float,
    end: float,
) -> float:
    return topk_mean(
        value
        for value, time_value in zip(values, times)
        if start <= time_value <= end)


def _bbox_summary(extra: Mapping[str, Any]) -> Dict[str, float]:
    summary = extra.get("bbox_summary", extra) if extra else {}
    if not isinstance(summary, Mapping):
        summary = {}
    return {
        "bbox_num_boxes": _finite_or_default(
            _first_present(summary, "num_boxes", "bbox_num_boxes")),
        "bbox_num_vehicle_boxes": _finite_or_default(
            _first_present(
                summary,
                "num_vehicle_boxes",
                "vehicle_boxes",
                "bbox_num_vehicle_boxes")),
        "bbox_num_pedestrian_boxes": _finite_or_default(
            _first_present(
                summary,
                "num_pedestrian_boxes",
                "pedestrian_boxes",
                "bbox_num_pedestrian_boxes")),
        "bbox_min_forward_distance_m": _finite_or_default(
            _first_present(
                summary,
                "min_forward_distance_m",
                "bbox_min_forward_distance_m")),
    }


def _first_present(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return 0.0


def _command_finite(command: SuspensionCommand) -> bool:
    for wheel in command.wheels:
        if (
                not _is_finite_number(wheel.spring_scale) or
                not _is_finite_number(wheel.damper_scale)):
            return False
    return True
