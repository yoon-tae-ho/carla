"""Residual-MPC Step 02 identification probe over Phase A."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .base import ControllerContext, ControllerOutput, VehicleState, clamp
from .planning_aware_mpc_phaseA import (
    PLANNING_AWARE_V4_MPC_PHASEA_VERSION,
    PlanningAwareV4MpcPhaseAConfig,
    PlanningAwareV4MpcPhaseAController,
)
from .planning_aware_mpc_primary import (
    PlanningAwareV3MpcPrimaryController,
    _WHEEL_LABELS,
    _all_finite,
    _context_dt,
    _match_length,
    _optional_finite_float,
    _safe_float,
)
from .residual_mpc_contracts import (
    MODAL_BASIS_NAMES,
    WHEEL_ORDER,
    modal_to_wheel_residual,
    wheel_residual_to_modal,
)


PLANNING_AWARE_V5_RESIDUAL_ID_PROBE_VERSION = (
    "planning_aware_v5_residual_id_probe_walsh_step02_v1")
RESIDUAL_ID_PROBE_TELEMETRY_SCHEMA_VERSION = (
    "residual_mpc_id_probe_telemetry_v1")

_WALSH_HADAMARD_8: Tuple[Tuple[float, ...], ...] = (
    (1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0),
    (1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0),
    (1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0),
    (1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0),
)


RESIDUAL_ID_PROBE_DIAGNOSTIC_FIELDS: Tuple[str, ...] = (
    "residual_mpc_schema_version",
    "residual_mpc_controller_version",
    "residual_mpc_profile_name",
    "residual_mpc_enabled",
    "residual_mpc_mode",
    "residual_mpc_fallback_reason",
    "residual_mpc_solver_status",
    "residual_mpc_compute_ms",
    "residual_mpc_run_id",
    "residual_mpc_route_id",
    "residual_mpc_ad_seed",
    "residual_mpc_excitation_seed",
    "residual_mpc_excitation_family",
    "residual_mpc_excitation_sequence_hash",
    "residual_mpc_excitation_block_index",
    "residual_mpc_excitation_cycle_index",
    "residual_mpc_excitation_active",
    "residual_mpc_suppression_reason",
    "residual_mpc_carla_frame",
    "residual_mpc_lead_producer_frame",
    "residual_mpc_sidecar_step",
    "residual_mpc_sim_time_s",
    "residual_mpc_wall_timestamp",
    "residual_mpc_measured_dt_s",
    "residual_mpc_planning_age_frames",
    "residual_mpc_planning_age_s",
    "residual_mpc_preview_curvature_max_abs",
    "residual_mpc_preview_lateral_acc_max_abs",
    "residual_mpc_preview_target_speed_now",
    "residual_mpc_preview_target_speed_semantics",
    "residual_mpc_preview_control_semantics",
    "residual_mpc_preview_validity_reason",
    "residual_mpc_preview_source_mask",
    "residual_mpc_phase_a_commit_match",
    "residual_mpc_bounds_guard_active",
    "residual_mpc_projection_active",
    "residual_mpc_final_guard_active",
    "residual_mpc_spring_scale_invariant",
    "residual_mpc_modal_mean",
    "residual_mpc_modal_roll_front",
    "residual_mpc_modal_roll_rear",
    "residual_mpc_modal_pitch",
    "residual_mpc_phase_a_modal_mean",
    "residual_mpc_phase_a_modal_roll_front",
    "residual_mpc_phase_a_modal_roll_rear",
    "residual_mpc_phase_a_modal_pitch",
    "residual_mpc_roll_deg_raw",
    "residual_mpc_roll_deg_filt",
    "residual_mpc_roll_rate_degps_raw",
    "residual_mpc_roll_rate_degps_filt",
    "residual_mpc_pitch_deg_raw",
    "residual_mpc_pitch_deg_filt",
    "residual_mpc_pitch_rate_degps_raw",
    "residual_mpc_pitch_rate_degps_filt",
    "residual_mpc_yaw_rate_degps_raw",
    "residual_mpc_yaw_rate_degps_filt",
    "residual_mpc_ax_world_mps2_raw",
    "residual_mpc_ax_world_mps2_filt",
    "residual_mpc_ay_world_mps2_raw",
    "residual_mpc_ay_world_mps2_filt",
    "residual_mpc_ax_body_mps2_raw",
    "residual_mpc_ax_body_mps2_filt",
    "residual_mpc_ay_body_mps2_raw",
    "residual_mpc_ay_body_mps2_filt",
) + tuple(
    "%s_%s" % (field, label)
    for field in (
        "residual_mpc_phase_a_shadow_damper",
        "residual_mpc_requested_residual",
        "residual_mpc_bounded_residual",
        "residual_mpc_rate_limited_residual",
        "residual_mpc_projected_residual",
        "residual_mpc_final_residual",
        "residual_mpc_final_damper",
    )
    for label in _WHEEL_LABELS
)


@dataclass(frozen=True)
class PlanningAwareV5ResidualIdProbeConfig(PlanningAwareV4MpcPhaseAConfig):
    """Step 02 deterministic residual excitation profile."""

    controller_version: str = PLANNING_AWARE_V5_RESIDUAL_ID_PROBE_VERSION
    profile_name: str = "conservative"
    telemetry_schema_version: str = RESIDUAL_ID_PROBE_TELEMETRY_SCHEMA_VERSION

    excitation_family: str = "walsh_hadamard_8"
    excitation_seed: int = 111
    excitation_mode_names: Tuple[str, ...] = MODAL_BASIS_NAMES
    excitation_amplitude_by_mode: Tuple[float, ...] = (
        0.010,
        0.012,
        0.012,
        0.010,
    )
    block_duration_s: float = 0.20
    cycle_length_blocks: int = 8
    warmup_s: float = 1.0
    active_duration_s: float = 0.0
    cooldown_s: float = 0.0

    speed_gate_min_mps: float = 1.0
    speed_gate_max_mps: float = 35.0
    max_dt_s: float = 0.20
    min_command_margin_scale: float = 0.005
    require_valid_planning: bool = False
    require_suspension_state_for_projection: bool = True

    run_id: str = ""
    route_id: str = ""
    ad_seed: str = ""
    output_filter_tau_s: float = 0.15

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> "PlanningAwareV5ResidualIdProbeConfig":
        allowed = {item.name for item in fields(cls)}
        incoming = dict(values or {})
        if "amplitude_by_mode" in incoming:
            incoming.setdefault(
                "excitation_amplitude_by_mode",
                incoming["amplitude_by_mode"])
        kwargs = {
            key: value
            for key, value in incoming.items()
            if key in allowed
        }
        if "excitation_mode_names" in kwargs:
            kwargs["excitation_mode_names"] = _parse_string_tuple(
                kwargs["excitation_mode_names"])
        if "excitation_amplitude_by_mode" in kwargs:
            kwargs["excitation_amplitude_by_mode"] = _parse_float_tuple(
                kwargs["excitation_amplitude_by_mode"])
        return cls(**kwargs)


class PlanningAwareV5ResidualIdProbeController(
        PlanningAwareV3MpcPrimaryController):
    """Phase-A-based deterministic residual ID probe."""

    name = "planning_aware_v5_residual_id_probe"
    requires_suspension_state = True

    def __init__(
        self,
        config: Optional[PlanningAwareV5ResidualIdProbeConfig] = None,
    ):
        self.config = config or PlanningAwareV5ResidualIdProbeConfig()
        super().__init__(self.config)
        self.phase_a_controller = PlanningAwareV4MpcPhaseAController(
            PlanningAwareV4MpcPhaseAConfig.from_mapping(
                _phase_a_config_values(self.config)))
        self._filtered_outputs: Dict[str, float] = {}

    def reset(self, native_suspension: Any = None) -> None:
        super().reset(native_suspension)
        self.phase_a_controller.reset(native_suspension)
        self._filtered_outputs = {}

    def compute(self, context: ControllerContext) -> ControllerOutput:
        start_ns = time.perf_counter_ns()
        cfg = self.config
        wheel_count = self._wheel_count(context)
        self._ensure_previous_count(wheel_count)

        phase_a_shadow = self.phase_a_controller.compute_shadow(context)
        phase_a_dampers = self._command_dampers(
            phase_a_shadow.command,
            wheel_count,
            default=1.0)
        phase_a_dampers = self._clamped_finite_dampers(
            phase_a_dampers,
            wheel_count)
        phase_a_modal = wheel_residual_to_modal(
            tuple(value - 1.0 for value in _match_length(
                phase_a_dampers,
                4,
                1.0)))

        suppression_reason = self._suppression_reason(context, phase_a_dampers)
        modal_residual, block_index, cycle_index = self._modal_excitation(
            context,
            suppressed=bool(suppression_reason))
        wheel_residual = modal_to_wheel_residual(modal_residual)
        wheel_residual = _match_length(wheel_residual, wheel_count, 0.0)

        requested = tuple(
            phase_a_dampers[index] + wheel_residual[index]
            for index in range(wheel_count))
        bounded = self._bounded_dampers(requested, wheel_count)
        rate_limited, bounds_guard_active = (
            self._candidate_bounds_and_rate_dampers(
                requested,
                self.previous_final_damper_scales,
                context))
        projected, projection_active = self._semi_active_project_dampers(
            rate_limited,
            context)
        final, final_guard_active = self._final_safety_dampers(
            projected,
            context)
        mode = "id_probe"
        if suppression_reason:
            mode = "phase_a_suppressed"
        if not _all_finite(final):
            final = phase_a_dampers
            projected = final
            rate_limited = final
            bounded = final
            suppression_reason = _join_reasons(
                suppression_reason,
                "final_nonfinite_safety_guard")
            mode = "phase_a_fallback"

        command = self._command_from_dampers(final)
        phase_a_commit = self.phase_a_controller.compute(context)
        phase_a_commit_match = int(
            phase_a_commit.command.is_close(phase_a_shadow.command))

        final_residual = tuple(
            final[index] - phase_a_dampers[index]
            for index in range(wheel_count))
        final_modal = wheel_residual_to_modal(_match_length(
            final_residual,
            4,
            0.0))
        diagnostics = self._diagnostics(
            context=context,
            phase_a_shadow_diagnostics=phase_a_shadow.diagnostics,
            phase_a_dampers=phase_a_dampers,
            phase_a_modal=phase_a_modal,
            requested_residual=tuple(
                requested[index] - phase_a_dampers[index]
                for index in range(wheel_count)),
            bounded_residual=tuple(
                bounded[index] - phase_a_dampers[index]
                for index in range(wheel_count)),
            rate_limited_residual=tuple(
                rate_limited[index] - phase_a_dampers[index]
                for index in range(wheel_count)),
            projected_residual=tuple(
                projected[index] - phase_a_dampers[index]
                for index in range(wheel_count)),
            final_residual=final_residual,
            final_dampers=final,
            final_modal=final_modal,
            mode=mode,
            suppression_reason=suppression_reason,
            block_index=block_index,
            cycle_index=cycle_index,
            excitation_active=int(
                not suppression_reason and any(
                    abs(value) > 1.0e-12 for value in modal_residual)),
            bounds_guard_active=bounds_guard_active,
            projection_active=projection_active,
            final_guard_active=final_guard_active,
            phase_a_commit_match=phase_a_commit_match,
            compute_ms=max(
                0.0,
                (time.perf_counter_ns() - start_ns) / 1.0e6),
        )
        self.previous_final_damper_scales = list(final)
        return ControllerOutput(command=command, diagnostics=diagnostics)

    def _modal_excitation(
        self,
        context: ControllerContext,
        *,
        suppressed: bool,
    ) -> Tuple[Tuple[float, ...], int, int]:
        if suppressed:
            return (0.0, 0.0, 0.0, 0.0), -1, -1
        cfg = self.config
        elapsed = _safe_float(getattr(context.state, "elapsed_seconds", 0.0), 0.0)
        warmup_s = max(_safe_float(cfg.warmup_s, 1.0), 0.0)
        if elapsed < warmup_s:
            return (0.0, 0.0, 0.0, 0.0), -1, -1
        active_duration_s = max(_safe_float(cfg.active_duration_s, 0.0), 0.0)
        if active_duration_s > 0.0 and elapsed >= warmup_s + active_duration_s:
            return (0.0, 0.0, 0.0, 0.0), -1, -1
        block_duration = max(_safe_float(cfg.block_duration_s, 0.20), 1.0e-6)
        block_index = int(math.floor((elapsed - warmup_s) / block_duration))
        cycle_len = _cycle_length(cfg)
        cycle_index = (
            block_index + int(cfg.excitation_seed)) % cycle_len
        signs = _seed_signs(int(cfg.excitation_seed))
        amplitudes = _amplitudes(cfg)
        enabled = set(_mode_names(cfg))
        values: List[float] = []
        for index, mode_name in enumerate(MODAL_BASIS_NAMES):
            if mode_name not in enabled:
                values.append(0.0)
                continue
            values.append(
                amplitudes[index] *
                signs[index] *
                _WALSH_HADAMARD_8[index][cycle_index])
        return tuple(values), block_index, cycle_index

    def _suppression_reason(
        self,
        context: ControllerContext,
        phase_a_dampers: Sequence[float],
    ) -> str:
        cfg = self.config
        state = getattr(context, "state", VehicleState())
        reasons: List[str] = []
        dt = _context_dt(context, cfg.default_dt)
        if dt <= 0.0 or dt > max(_safe_float(cfg.max_dt_s, 0.20), 1.0e-6):
            reasons.append("dt_gap")
        speed = _safe_float(getattr(state, "speed", 0.0), 0.0)
        if speed < _safe_float(cfg.speed_gate_min_mps, 1.0):
            reasons.append("speed_below_gate")
        if speed > _safe_float(cfg.speed_gate_max_mps, 35.0):
            reasons.append("speed_above_gate")
        for field_name in (
                "roll",
                "pitch",
                "yaw_rate",
                "local_ax",
                "local_ay",
                "ax",
                "ay"):
            if _optional_finite_float(getattr(state, field_name, None)) is None:
                reasons.append("nonfinite_telemetry")
                break
        planning = getattr(context, "planning", None)
        if bool(getattr(cfg, "require_valid_planning", False)):
            if planning is None or not bool(getattr(planning, "available", False)):
                reasons.append("planning_invalid")
            else:
                age = _planning_age_frames(context)
                if (
                        age is not None and
                        age > int(_safe_float(cfg.max_planning_age_frames, 5))):
                    reasons.append("planning_stale")
        if (
                planning is not None and
                int(getattr(planning, "collision_count", 0) or 0) > 0):
            reasons.append("collision_state")
        if (
                bool(getattr(
                    cfg,
                    "require_suspension_state_for_projection",
                    True)) and
                not bool(getattr(context, "suspension_state_valid", False))):
            reasons.append("suspension_state_invalid")
        if not _all_finite(phase_a_dampers):
            reasons.append("phase_a_nonfinite")
        low = _safe_float(cfg.min_damper_scale, 0.75)
        high = _safe_float(cfg.max_damper_scale, 1.25)
        margin = min(
            min(abs(value - low), abs(high - value))
            for value in phase_a_dampers)
        if margin < max(_safe_float(cfg.min_command_margin_scale, 0.005), 0.0):
            reasons.append("command_margin_insufficient")
        return ";".join(dict.fromkeys(reasons))

    def _bounded_dampers(
        self,
        values: Sequence[float],
        wheel_count: int,
    ) -> Tuple[float, ...]:
        low = _safe_float(self.config.min_damper_scale, 0.75)
        high = _safe_float(self.config.max_damper_scale, 1.25)
        return tuple(
            clamp(_safe_float(value, 1.0), low, high)
            for value in _match_length(values, wheel_count, 1.0))

    def _diagnostics(
        self,
        *,
        context: ControllerContext,
        phase_a_shadow_diagnostics: Mapping[str, Any],
        phase_a_dampers: Sequence[float],
        phase_a_modal: Sequence[float],
        requested_residual: Sequence[float],
        bounded_residual: Sequence[float],
        rate_limited_residual: Sequence[float],
        projected_residual: Sequence[float],
        final_residual: Sequence[float],
        final_dampers: Sequence[float],
        final_modal: Sequence[float],
        mode: str,
        suppression_reason: str,
        block_index: int,
        cycle_index: int,
        excitation_active: int,
        bounds_guard_active: bool,
        projection_active: bool,
        final_guard_active: bool,
        phase_a_commit_match: int,
        compute_ms: float,
    ) -> Dict[str, Any]:
        state = context.state
        planning = getattr(context, "planning", None)
        metadata = dict(getattr(planning, "metadata", {}) or {})
        preview_summary = (
            planning.preview_summary()
            if planning is not None else {})
        planning_age_frames = _planning_age_frames(context)
        planning_age_s = (
            planning_age_frames * _safe_float(
                getattr(planning, "horizon_dt", 0.1),
                0.1)
            if planning_age_frames is not None and planning is not None else "")
        diagnostics: Dict[str, Any] = {
            "residual_mpc_schema_version": (
                self.config.telemetry_schema_version),
            "residual_mpc_controller_version": self.config.controller_version,
            "residual_mpc_profile_name": self.config.profile_name,
            "residual_mpc_enabled": 1,
            "residual_mpc_mode": mode,
            "residual_mpc_fallback_reason": suppression_reason or "none",
            "residual_mpc_solver_status": "not_used_id_probe",
            "residual_mpc_compute_ms": compute_ms,
            "residual_mpc_run_id": self.config.run_id,
            "residual_mpc_route_id": self.config.route_id,
            "residual_mpc_ad_seed": self.config.ad_seed,
            "residual_mpc_excitation_seed": int(self.config.excitation_seed),
            "residual_mpc_excitation_family": self.config.excitation_family,
            "residual_mpc_excitation_sequence_hash": sequence_hash(self.config),
            "residual_mpc_excitation_block_index": block_index,
            "residual_mpc_excitation_cycle_index": cycle_index,
            "residual_mpc_excitation_active": excitation_active,
            "residual_mpc_suppression_reason": suppression_reason or "none",
            "residual_mpc_carla_frame": getattr(state, "frame", -1),
            "residual_mpc_lead_producer_frame": getattr(planning, "frame", -1),
            "residual_mpc_sidecar_step": getattr(context, "step", 0),
            "residual_mpc_sim_time_s": getattr(state, "elapsed_seconds", 0.0),
            "residual_mpc_wall_timestamp": time.time(),
            "residual_mpc_measured_dt_s": _context_dt(
                context,
                self.config.default_dt),
            "residual_mpc_planning_age_frames": (
                planning_age_frames if planning_age_frames is not None else ""),
            "residual_mpc_planning_age_s": planning_age_s,
            "residual_mpc_preview_curvature_max_abs": preview_summary.get(
                "preview_curvature_max_abs",
                0.0),
            "residual_mpc_preview_lateral_acc_max_abs": preview_summary.get(
                "preview_lateral_acc_max_abs",
                0.0),
            "residual_mpc_preview_target_speed_now": preview_summary.get(
                "preview_target_speed_now",
                0.0),
            "residual_mpc_preview_target_speed_semantics": metadata.get(
                "speed_semantics",
                "unverified"),
            "residual_mpc_preview_control_semantics": metadata.get(
                "control_semantics",
                "unverified"),
            "residual_mpc_preview_validity_reason": metadata.get(
                "validity_reason",
                ""),
            "residual_mpc_preview_source_mask": metadata.get(
                "preview_source_mask",
                metadata.get("trajectory_source", "")),
            "residual_mpc_phase_a_commit_match": phase_a_commit_match,
            "residual_mpc_bounds_guard_active": int(bounds_guard_active),
            "residual_mpc_projection_active": int(projection_active),
            "residual_mpc_final_guard_active": int(final_guard_active),
            "residual_mpc_spring_scale_invariant": 1.0,
        }
        diagnostics.update(dict(phase_a_shadow_diagnostics or {}))
        self._modal_diagnostics(
            diagnostics,
            "residual_mpc_modal",
            final_modal)
        self._modal_diagnostics(
            diagnostics,
            "residual_mpc_phase_a_modal",
            phase_a_modal)
        self._wheel_diagnostics(
            diagnostics,
            "residual_mpc_phase_a_shadow_damper",
            phase_a_dampers)
        self._wheel_diagnostics(
            diagnostics,
            "residual_mpc_requested_residual",
            requested_residual)
        self._wheel_diagnostics(
            diagnostics,
            "residual_mpc_bounded_residual",
            bounded_residual)
        self._wheel_diagnostics(
            diagnostics,
            "residual_mpc_rate_limited_residual",
            rate_limited_residual)
        self._wheel_diagnostics(
            diagnostics,
            "residual_mpc_projected_residual",
            projected_residual)
        self._wheel_diagnostics(
            diagnostics,
            "residual_mpc_final_residual",
            final_residual)
        self._wheel_diagnostics(
            diagnostics,
            "residual_mpc_final_damper",
            final_dampers)
        diagnostics.update(self._output_channel_diagnostics(context))
        self._ensure_residual_diagnostics(diagnostics)
        return diagnostics

    def _output_channel_diagnostics(
        self,
        context: ControllerContext,
    ) -> Dict[str, Any]:
        state = context.state
        raw = {
            "residual_mpc_roll_deg_raw": _safe_float(state.roll),
            "residual_mpc_roll_rate_degps_raw": _safe_float(state.roll_rate),
            "residual_mpc_pitch_deg_raw": _safe_float(state.pitch),
            "residual_mpc_pitch_rate_degps_raw": _safe_float(state.pitch_rate),
            "residual_mpc_yaw_rate_degps_raw": _safe_float(state.yaw_rate),
            "residual_mpc_ax_world_mps2_raw": _safe_float(state.ax),
            "residual_mpc_ay_world_mps2_raw": _safe_float(state.ay),
            "residual_mpc_ax_body_mps2_raw": _safe_float(state.local_ax),
            "residual_mpc_ay_body_mps2_raw": _safe_float(state.local_ay),
        }
        dt = _context_dt(context, self.config.default_dt)
        tau = max(_safe_float(self.config.output_filter_tau_s, 0.15), 0.0)
        alpha = 1.0 if tau <= 1.0e-12 else clamp(dt / (tau + dt), 0.0, 1.0)
        out: Dict[str, Any] = {}
        for raw_key, raw_value in raw.items():
            filt_key = raw_key.replace("_raw", "_filt")
            previous = self._filtered_outputs.get(filt_key)
            filtered = (
                raw_value if previous is None else
                previous + alpha * (raw_value - previous))
            self._filtered_outputs[filt_key] = filtered
            out[raw_key] = raw_value
            out[filt_key] = filtered
        return out

    @staticmethod
    def _modal_diagnostics(
        diagnostics: Dict[str, Any],
        prefix: str,
        values: Sequence[float],
    ) -> None:
        modal_values = _match_length(values, len(MODAL_BASIS_NAMES), 0.0)
        for mode_name, value in zip(MODAL_BASIS_NAMES, modal_values):
            diagnostics["%s_%s" % (prefix, mode_name)] = value

    @staticmethod
    def _wheel_diagnostics(
        diagnostics: Dict[str, Any],
        prefix: str,
        values: Sequence[float],
    ) -> None:
        wheel_values = _match_length(values, len(_WHEEL_LABELS), 0.0)
        for label, value in zip(_WHEEL_LABELS, wheel_values):
            diagnostics["%s_%s" % (prefix, label)] = value

    @staticmethod
    def _ensure_residual_diagnostics(diagnostics: Dict[str, Any]) -> None:
        for key in RESIDUAL_ID_PROBE_DIAGNOSTIC_FIELDS:
            if key in diagnostics:
                continue
            if (
                    key.endswith("_version") or
                    key.endswith("_name") or
                    key.endswith("_mode") or
                    key.endswith("_reason") or
                    key.endswith("_status") or
                    key.endswith("_family") or
                    key.endswith("_hash") or
                    key.endswith("_id") or
                    key.endswith("_semantics") or
                    key.endswith("_mask")):
                diagnostics[key] = ""
            else:
                diagnostics[key] = 0.0


def sequence_hash(config: PlanningAwareV5ResidualIdProbeConfig) -> str:
    payload = {
        "family": config.excitation_family,
        "seed": int(config.excitation_seed),
        "modes": _mode_names(config),
        "amplitudes": _amplitudes(config),
        "cycle": [
            _cycle_modal_values(config, cycle_index)
            for cycle_index in range(_cycle_length(config))
        ],
    }
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]


def _cycle_modal_values(
    config: PlanningAwareV5ResidualIdProbeConfig,
    cycle_index: int,
) -> Tuple[float, ...]:
    signs = _seed_signs(int(config.excitation_seed))
    amplitudes = _amplitudes(config)
    enabled = set(_mode_names(config))
    return tuple(
        (
            amplitudes[index] *
            signs[index] *
            _WALSH_HADAMARD_8[index][cycle_index]
        )
        if mode_name in enabled else 0.0
        for index, mode_name in enumerate(MODAL_BASIS_NAMES))


def _seed_signs(seed: int) -> Tuple[float, ...]:
    digest = hashlib.sha256(str(seed).encode("ascii")).digest()
    return tuple(-1.0 if digest[index] & 1 else 1.0 for index in range(4))


def _amplitudes(
    config: PlanningAwareV5ResidualIdProbeConfig,
) -> Tuple[float, float, float, float]:
    return tuple(_match_length(
        tuple(float(value) for value in config.excitation_amplitude_by_mode),
        len(MODAL_BASIS_NAMES),
        0.0))  # type: ignore[return-value]


def _mode_names(config: PlanningAwareV5ResidualIdProbeConfig) -> Tuple[str, ...]:
    names = tuple(str(name) for name in config.excitation_mode_names)
    return tuple(name for name in names if name in MODAL_BASIS_NAMES)


def _cycle_length(config: PlanningAwareV5ResidualIdProbeConfig) -> int:
    requested = int(_safe_float(config.cycle_length_blocks, 8))
    return 8 if requested <= 0 else min(8, requested)


def _phase_a_config_values(
    config: PlanningAwareV5ResidualIdProbeConfig,
) -> Dict[str, Any]:
    phase_fields = {field.name for field in fields(PlanningAwareV4MpcPhaseAConfig)}
    values = {
        key: getattr(config, key)
        for key in phase_fields
        if hasattr(config, key)
    }
    values["controller_version"] = PLANNING_AWARE_V4_MPC_PHASEA_VERSION
    return values


def _planning_age_frames(context: ControllerContext) -> Optional[int]:
    planning = getattr(context, "planning", None)
    state = getattr(context, "state", None)
    planning_frame = _optional_finite_float(getattr(planning, "frame", None))
    state_frame = _optional_finite_float(getattr(state, "frame", None))
    if planning_frame is None or state_frame is None or planning_frame < 0:
        return None
    return int(state_frame) - int(planning_frame)


def _join_reasons(left: str, right: str) -> str:
    return ";".join(reason for reason in (left, right) if reason)


def _parse_string_tuple(value: Any) -> Tuple[str, ...]:
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    return tuple(str(item) for item in (value or ()))


def _parse_float_tuple(value: Any) -> Tuple[float, ...]:
    if isinstance(value, str):
        return tuple(
            float(item.strip())
            for item in value.split(",")
            if item.strip())
    return tuple(float(item) for item in (value or ()))
