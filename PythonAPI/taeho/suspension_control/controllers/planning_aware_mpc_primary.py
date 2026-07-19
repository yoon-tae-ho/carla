"""Planning-aware v3 preview and guard helpers.

This module intentionally stops short of implementing or registering the
MPC-primary controller.  It provides deterministic CARLA-free inputs for that
future controller: fixed-step planning previews, source diagnostics, and a
rolling comfort guard.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .base import (
    ControllerContext,
    ControllerOutput,
    PlanningInfo,
    SuspensionCommand,
    SuspensionController,
    VehicleState,
    WheelScale,
    clamp,
)
from .skyhook import SkyhookConfig, canonical_skyhook_v3_projection
from .skyhook_roll_v3 import (
    SkyhookRollV3CanonicalModalConfig,
    SkyhookRollV3CanonicalModalController,
)


PLANNING_AWARE_V3_MPC_PRIMARY_VERSION = (
    "planning_aware_v3_mpc_primary_b2d_threshold_modal_v1")

_EPS = 1.0e-9
_FROZEN_SPRING_SCALE = 1.0
_WHEEL_LABELS = ("fl", "fr", "rl", "rr")


@dataclass(frozen=True)
class PlanningAwareV3MpcPrimaryConfig:
    """Config shared by the future MPC-primary controller and its helpers."""

    controller_version: str = PLANNING_AWARE_V3_MPC_PRIMARY_VERSION

    default_dt: float = 0.05
    model_dt: float = 0.05
    horizon_s: float = 1.5
    max_horizon_s: float = 2.0
    max_planning_age_frames: int = 5
    min_horizon_dt: float = 0.02
    max_horizon_dt: float = 0.20
    wheel_count: int = 4

    min_damper_scale: float = 0.75
    max_damper_scale: float = 1.25
    max_rate_up_scale_per_s: float = 1.2
    max_rate_down_scale_per_s: float = 1.2

    target_speed_ramp_s: float = 1.0
    brake_decel_scale: float = 6.0
    throttle_accel_scale: float = 2.0
    speed_cap_mps: float = 45.0

    min_preview_points_for_trajectory_curvature: int = 3
    curvature_abs_cap: float = 0.30
    predicted_accel_abs_cap: float = 30.0
    yaw_rate_abs_cap: float = 5.0
    yaw_acc_abs_cap: float = 20.0

    half_wheelbase_m: float = 1.45
    half_track_m: float = 0.85
    front_x_sign: float = 1.0
    left_y_sign: float = -1.0
    outer_side_sign_from_ay: float = -1.0
    g: float = 9.81

    use_skyhook_prior: bool = False

    u_mean_grid: Tuple[float, ...] = (-0.10, -0.05, 0.0, 0.05, 0.10)
    u_roll_front_grid: Tuple[float, ...] = (
        -0.25, -0.16, -0.08, 0.0, 0.08, 0.16, 0.25)
    u_roll_rear_grid: Tuple[float, ...] = (
        -0.25, -0.16, -0.08, 0.0, 0.08, 0.16, 0.25)
    u_pitch_grid: Tuple[float, ...] = (
        -0.20, -0.12, -0.06, 0.0, 0.06, 0.12, 0.20)

    omega_roll_rad_s: float = 7.0
    omega_pitch_rad_s: float = 8.0
    zeta_roll_base: float = 0.36
    zeta_pitch_base: float = 0.38
    zeta_mean_gain: float = 0.60
    zeta_roll_gain: float = 2.20
    zeta_pitch_gain: float = 1.80
    K_phi_ay: float = 0.010
    K_theta_ax: float = -0.007
    k_phi_acc: float = 0.22
    k_theta_acc: float = 0.22
    k_phi_g: float = 0.18
    k_theta_g: float = 0.18

    W_b2d: float = 4.0
    W_body: float = 1.2
    W_ltr: float = 2.5
    W_command: float = 0.18
    W_slew: float = 5.0
    W_mean: float = 2.8
    W_feasible: float = 0.50
    W_prior: float = 0.0
    W_roll_preview: float = 5.0
    W_pitch_preview: float = 2.5

    roll_preview_full_lat_acc: float = 5.0
    roll_preview_max_u: float = 0.10
    pitch_preview_full_decel: float = 4.0
    pitch_preview_max_u: float = 0.08

    h_cg_proxy_m: float = 0.55
    ltr_soft_threshold: float = 0.45
    ltr_hard_threshold: float = 0.75
    ltr_soft_roll_weight_gain: float = 3.0

    relative_extension_sign: float = -1.0
    skyhook_law: str = "projected_force"
    skyhook_c_scale: float = 1.0
    neutral_damper_scale: float = 1.0
    low_damper_scale: float = 0.82
    high_damper_scale: float = 1.22
    sprung_velocity_deadband: float = 0.025
    rel_velocity_deadband: float = 0.015
    product_deadband: float = 0.0005
    projection_eps: float = 1.0e-6
    projection_blend_infeasible: float = 0.35
    projection_blend_low_rel: float = 0.50

    comfort_history_s: float = 2.0
    comfort_filter_tau_s: float = 0.15
    comfort_margin_activation: float = 0.10
    comfort_slew_penalty_gain: float = 2.0
    comfort_mean_damper_penalty_gain: float = 1.0

    comfort_lon_acc_upper: float = 2.40
    comfort_lon_acc_lower: float = -4.05
    comfort_abs_lat_acc: float = 4.89
    comfort_abs_lon_jerk: float = 4.13
    comfort_abs_mag_jerk: float = 8.37
    comfort_abs_yaw_acc: float = 1.93
    comfort_abs_yaw_rate: float = 0.95

    debug: bool = False

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> "PlanningAwareV3MpcPrimaryConfig":
        allowed = {item.name for item in fields(cls)}
        incoming = dict(values or {})
        if "preview_horizon_s" in incoming and "horizon_s" not in incoming:
            incoming["horizon_s"] = incoming["preview_horizon_s"]
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


@dataclass(frozen=True)
class PlanningAwareV3Preview:
    """Fixed-step planning preview consumed by the future MPC-lite optimizer."""

    valid: bool = False
    fallback_reason: str = "not_built"
    source: str = ""
    frame: int = -1
    planning_age_frames: int = -1

    dt: float = 0.05
    horizon_s: float = 0.0
    times: Tuple[float, ...] = ()
    speed_profile: Tuple[float, ...] = ()
    curvature: Tuple[float, ...] = ()
    predicted_ax: Tuple[float, ...] = ()
    predicted_ay: Tuple[float, ...] = ()
    yaw_rate_ref: Tuple[float, ...] = ()
    yaw_acc_ref: Tuple[float, ...] = ()

    predicted_ax_source: str = "missing"
    predicted_ay_source: str = "missing"
    speed_profile_source: str = "missing"
    curvature_source: str = "missing"

    def as_diagnostics(self) -> Dict[str, Any]:
        return {
            "planning_aware_v3_valid": 1 if self.valid else 0,
            "planning_aware_v3_fallback_reason": self.fallback_reason,
            "planning_aware_v3_mode": "preview" if self.valid else "fallback",
            "mpc_horizon_s": _safe_float(self.horizon_s),
            "mpc_dt": _safe_float(self.dt),
            "predicted_ax_source": self.predicted_ax_source,
            "predicted_ay_source": self.predicted_ay_source,
            "speed_profile_source": self.speed_profile_source,
            "predicted_curvature_source": self.curvature_source,
            "curvature_source": self.curvature_source,
            "planning_aware_v3_planning_age_frames": self.planning_age_frames,
            "mpc_preview_points": len(self.times),
            "predicted_ay_peak": _signed_peak(self.predicted_ay),
            "predicted_ax_min": min(self.predicted_ax) if self.predicted_ax else 0.0,
            "predicted_curvature_peak": _signed_peak(self.curvature),
            "predicted_yaw_rate_peak": _signed_peak(self.yaw_rate_ref),
            "predicted_yaw_acc_peak": _signed_peak(self.yaw_acc_ref),
        }

    @property
    def diagnostics(self) -> Dict[str, Any]:
        return self.as_diagnostics()


@dataclass(frozen=True)
class ComfortGuardResult:
    """Rolling comfort margins and scheduling hints for later optimization."""

    lon_acc_margin: float = 0.0
    lat_acc_margin: float = 0.0
    lon_jerk_margin: float = 0.0
    mag_jerk_margin: float = 0.0
    yaw_acc_margin: float = 0.0
    yaw_rate_margin: float = 0.0
    active: bool = False
    slew_penalty_multiplier: float = 1.0
    mean_damper_penalty_multiplier: float = 1.0

    def as_diagnostics(self) -> Dict[str, Any]:
        return {
            "comfort_guard_lon_acc_margin": self.lon_acc_margin,
            "comfort_guard_lat_acc_margin": self.lat_acc_margin,
            "comfort_guard_lon_jerk_margin": self.lon_jerk_margin,
            "comfort_guard_mag_jerk_margin": self.mag_jerk_margin,
            "comfort_guard_yaw_acc_margin": self.yaw_acc_margin,
            "comfort_guard_yaw_rate_margin": self.yaw_rate_margin,
            "comfort_guard_active": 1 if self.active else 0,
            "comfort_guard_slew_penalty_multiplier": (
                self.slew_penalty_multiplier),
            "comfort_guard_mean_damper_penalty_multiplier": (
                self.mean_damper_penalty_multiplier),
        }

    @property
    def diagnostics(self) -> Dict[str, Any]:
        return self.as_diagnostics()


@dataclass
class ComfortGuardState:
    """Two-second filtered local acceleration and yaw-rate history."""

    samples: List[Tuple[float, float, float, float]] = field(default_factory=list)
    filtered_local_ax: Optional[float] = None
    filtered_local_ay: Optional[float] = None
    filtered_yaw_rate: Optional[float] = None
    last_time: Optional[float] = None

    def reset(self) -> None:
        self.samples.clear()
        self.filtered_local_ax = None
        self.filtered_local_ay = None
        self.filtered_yaw_rate = None
        self.last_time = None

    def update(
        self,
        context: ControllerContext,
        config: Optional[PlanningAwareV3MpcPrimaryConfig] = None,
    ) -> ComfortGuardResult:
        cfg = config or PlanningAwareV3MpcPrimaryConfig()
        dt = _context_dt(context, cfg.default_dt)
        state = getattr(context, "state", VehicleState())
        now = _optional_finite_float(getattr(state, "elapsed_seconds", None))
        if now is None or (self.last_time is not None and now <= self.last_time):
            now = (self.last_time if self.last_time is not None else 0.0) + dt

        local_ax = _safe_float(getattr(state, "local_ax", 0.0))
        local_ay = _safe_float(getattr(state, "local_ay", 0.0))
        yaw_rate = math.radians(_safe_float(getattr(state, "yaw_rate", 0.0)))

        tau = max(_safe_float(cfg.comfort_filter_tau_s, 0.15), 0.0)
        alpha = 1.0 if tau <= _EPS else clamp(dt / (tau + dt), 0.0, 1.0)
        self.filtered_local_ax = _low_pass(
            self.filtered_local_ax,
            local_ax,
            alpha)
        self.filtered_local_ay = _low_pass(
            self.filtered_local_ay,
            local_ay,
            alpha)
        self.filtered_yaw_rate = _low_pass(
            self.filtered_yaw_rate,
            yaw_rate,
            alpha)
        self.samples.append((
            now,
            self.filtered_local_ax,
            self.filtered_local_ay,
            self.filtered_yaw_rate))
        history_s = max(_safe_float(cfg.comfort_history_s, 2.0), dt)
        oldest = now - history_s
        self.samples = [sample for sample in self.samples if sample[0] >= oldest]
        self.last_time = now

        return _comfort_result_from_samples(self.samples, cfg)


PLANNING_AWARE_V3_REQUIRED_DIAGNOSTIC_FIELDS = (
    "planning_aware_v3_valid",
    "planning_aware_v3_fallback_reason",
    "planning_aware_v3_mode",
    "mpc_horizon_s",
    "mpc_dt",
    "mpc_candidate_count",
    "mpc_best_cost",
    "mpc_second_best_cost",
    "mpc_cost_margin",
    "predicted_ax_source",
    "predicted_ay_source",
    "speed_profile_source",
    "predicted_curvature_source",
    "predicted_ay_peak",
    "predicted_ax_min",
    "predicted_curvature_peak",
    "predicted_yaw_rate_peak",
    "predicted_yaw_acc_peak",
    "comfort_guard_lon_acc_margin",
    "comfort_guard_lat_acc_margin",
    "comfort_guard_lon_jerk_margin",
    "comfort_guard_mag_jerk_margin",
    "comfort_guard_yaw_acc_margin",
    "comfort_guard_yaw_rate_margin",
    "comfort_guard_active",
    "mpc_u_mean",
    "mpc_u_roll_front",
    "mpc_u_roll_rear",
    "mpc_u_pitch",
    "mpc_ltr_proxy_peak",
    "mpc_rate_constraint_active",
    "mpc_projection_active",
    "desired_damper_fl",
    "desired_damper_fr",
    "desired_damper_rl",
    "desired_damper_rr",
    "projected_damper_fl",
    "projected_damper_fr",
    "projected_damper_rl",
    "projected_damper_rr",
    "final_damper_fl",
    "final_damper_fr",
    "final_damper_rl",
    "final_damper_rr",
    "final_spring_fl",
    "final_spring_fr",
    "final_spring_rl",
    "final_spring_rr",
    "mean_damper",
    "front_roll_contrast",
    "rear_roll_contrast",
    "pitch_contrast",
    "damper_delta_fl",
    "damper_delta_fr",
    "damper_delta_rl",
    "damper_delta_rr",
    "skyhook_roll_v3_shadow_damper_fl",
    "skyhook_roll_v3_shadow_damper_fr",
    "skyhook_roll_v3_shadow_damper_rl",
    "skyhook_roll_v3_shadow_damper_rr",
    "skyhook_prior_penalty",
)


@dataclass(frozen=True)
class MpcDamperCandidate:
    """Reusable first-step damper candidate for v3-compatible MPC search."""

    kind: str
    raw_dampers: Tuple[float, ...]
    u_mean: Optional[float] = None
    u_roll_front: Optional[float] = None
    u_roll_rear: Optional[float] = None
    u_pitch: Optional[float] = None
    raw_cost: Optional[float] = None
    projected_cost: Optional[float] = None
    constrained_dampers: Optional[Tuple[float, ...]] = None
    projected_dampers: Optional[Tuple[float, ...]] = None
    final_dampers: Optional[Tuple[float, ...]] = None
    selected: bool = False
    rejection_reason: str = ""
    components: Mapping[str, float] = field(default_factory=dict)
    projected_components: Mapping[str, float] = field(default_factory=dict)
    ltr_proxy_peak: float = 0.0
    projected_ltr_proxy_peak: float = 0.0
    raw_index: int = 0

    @property
    def desired_dampers(self) -> Tuple[float, ...]:
        """Compatibility alias for the original v3 candidate API."""

        return self.raw_dampers

    @property
    def cost(self) -> float:
        """Compatibility alias for the original v3 raw candidate cost."""

        return _safe_float(self.raw_cost, 1.0e12)


_MpcCandidate = MpcDamperCandidate


@dataclass(frozen=True)
class _MpcOptimizationResult:
    selected: Optional[_MpcCandidate]
    candidate_count: int
    feasible_candidate_count: int
    best_cost: float
    second_best_cost: float
    cost_margin: float
    failed_reason: str
    rate_rejection_count: int
    ltr_soft_guard_active: bool
    ltr_hard_guard_active: bool
    raw_candidates: Tuple[MpcDamperCandidate, ...] = ()


class PlanningAwareV3MpcPrimaryController(SuspensionController):
    """Solver-free planning-primary damper controller.

    The controller keeps ``skyhook_roll_v3`` updated as a shadow/fallback, but
    valid planning drives the final damper scales directly through a small
    deterministic MPC-lite candidate search.
    """

    name = "planning_aware_v3_mpc_primary"
    requires_suspension_state = True

    def __init__(
        self,
        config: Optional[PlanningAwareV3MpcPrimaryConfig] = None,
    ):
        self.config = config or PlanningAwareV3MpcPrimaryConfig()
        self.comfort_guard = ComfortGuardState()
        self.skyhook_roll_v3_shadow = SkyhookRollV3CanonicalModalController(
            self._shadow_config())
        wheel_count = max(1, int(_safe_float(self.config.wheel_count, 4)))
        self.previous_final_damper_scales = [1.0 for _ in range(wheel_count)]

    def reset(self, native_suspension: Any = None) -> None:
        self.comfort_guard.reset()
        self.skyhook_roll_v3_shadow.reset(native_suspension)
        wheel_count = self._wheel_count_from_native(native_suspension)
        if wheel_count <= 0:
            wheel_count = max(1, int(_safe_float(self.config.wheel_count, 4)))
        self.previous_final_damper_scales = [1.0 for _ in range(wheel_count)]

    def compute_shadow(self, context: ControllerContext) -> ControllerOutput:
        """Compute the same tick command without mutating controller state."""

        snapshot = self._state_snapshot()
        try:
            return self.compute(context)
        finally:
            self._restore_state_snapshot(snapshot)

    def _state_snapshot(self) -> Dict[str, Any]:
        return {
            "comfort_samples": list(self.comfort_guard.samples),
            "comfort_filtered_local_ax": self.comfort_guard.filtered_local_ax,
            "comfort_filtered_local_ay": self.comfort_guard.filtered_local_ay,
            "comfort_filtered_yaw_rate": self.comfort_guard.filtered_yaw_rate,
            "comfort_last_time": self.comfort_guard.last_time,
            "previous_final_damper_scales": list(
                self.previous_final_damper_scales),
            "shadow_previous_damper_scales": list(getattr(
                self.skyhook_roll_v3_shadow,
                "previous_damper_scales",
                ())),
        }

    def _restore_state_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        self.comfort_guard.samples = list(snapshot["comfort_samples"])
        self.comfort_guard.filtered_local_ax = snapshot[
            "comfort_filtered_local_ax"]
        self.comfort_guard.filtered_local_ay = snapshot[
            "comfort_filtered_local_ay"]
        self.comfort_guard.filtered_yaw_rate = snapshot[
            "comfort_filtered_yaw_rate"]
        self.comfort_guard.last_time = snapshot["comfort_last_time"]
        self.previous_final_damper_scales = list(
            snapshot["previous_final_damper_scales"])
        if hasattr(self.skyhook_roll_v3_shadow, "previous_damper_scales"):
            self.skyhook_roll_v3_shadow.previous_damper_scales = list(
                snapshot["shadow_previous_damper_scales"])

    def compute(self, context: ControllerContext) -> ControllerOutput:
        cfg = self.config
        wheel_count = self._wheel_count(context)
        self._ensure_previous_count(wheel_count)

        shadow_output = self.skyhook_roll_v3_shadow.compute(context)
        shadow_dampers = self._command_dampers(
            shadow_output.command,
            wheel_count,
            default=1.0)
        shadow_usable, shadow_reason = self._shadow_output_usable(
            shadow_output,
            wheel_count)

        preview = build_planning_preview(context, cfg)
        comfort = self.comfort_guard.update(context, cfg)
        diagnostics = self._base_diagnostics(
            preview=preview,
            comfort=comfort,
            shadow_output=shadow_output,
            shadow_dampers=shadow_dampers,
            wheel_count=wheel_count)

        if not preview.valid:
            if shadow_usable:
                final_dampers = self._clamped_finite_dampers(
                    shadow_dampers,
                    wheel_count)
                command = self._command_from_dampers(final_dampers)
                diagnostics.update({
                    "planning_aware_v3_mode": "skyhook_roll_v3_fallback",
                    "planning_aware_v3_fallback_reason": (
                        preview.fallback_reason),
                    "mpc_best_cost": 0.0,
                    "mpc_second_best_cost": 0.0,
                    "mpc_cost_margin": 0.0,
                })
                self._set_command_diagnostics(
                    diagnostics,
                    desired=final_dampers,
                    projected=final_dampers,
                    final=final_dampers,
                    previous=self.previous_final_damper_scales,
                    projection_active=False)
                self.previous_final_damper_scales = list(final_dampers)
                self._ensure_required_diagnostics(diagnostics)
                return ControllerOutput(command=command, diagnostics=diagnostics)

            final_dampers = tuple(1.0 for _ in range(wheel_count))
            command = self._command_from_dampers(final_dampers)
            reason = "%s;skyhook_roll_v3_invalid:%s" % (
                preview.fallback_reason,
                shadow_reason)
            diagnostics.update({
                "planning_aware_v3_mode": "identity_fallback",
                "planning_aware_v3_fallback_reason": reason,
                "mpc_best_cost": 0.0,
                "mpc_second_best_cost": 0.0,
                "mpc_cost_margin": 0.0,
            })
            self._set_command_diagnostics(
                diagnostics,
                desired=final_dampers,
                projected=final_dampers,
                final=final_dampers,
                previous=self.previous_final_damper_scales,
                projection_active=False)
            self.previous_final_damper_scales = list(final_dampers)
            self._ensure_required_diagnostics(diagnostics)
            return ControllerOutput(command=command, diagnostics=diagnostics)

        optimizer = self._optimize_mpc(
            context=context,
            preview=preview,
            comfort=comfort,
            shadow_dampers=shadow_dampers)
        diagnostics.update(self._optimizer_diagnostics(optimizer, comfort))

        if optimizer.selected is None:
            reason = "mpc_optimizer_failed:%s" % optimizer.failed_reason
            if shadow_usable:
                final_dampers = self._clamped_finite_dampers(
                    shadow_dampers,
                    wheel_count)
                command = self._command_from_dampers(final_dampers)
                diagnostics.update({
                    "planning_aware_v3_valid": 0,
                    "planning_aware_v3_mode": "skyhook_roll_v3_fallback",
                    "planning_aware_v3_fallback_reason": reason,
                })
            else:
                final_dampers = tuple(1.0 for _ in range(wheel_count))
                command = self._command_from_dampers(final_dampers)
                diagnostics.update({
                    "planning_aware_v3_valid": 0,
                    "planning_aware_v3_mode": "identity_fallback",
                    "planning_aware_v3_fallback_reason": (
                        "%s;skyhook_roll_v3_invalid:%s" %
                        (reason, shadow_reason)),
                })
            self._set_command_diagnostics(
                diagnostics,
                desired=final_dampers,
                projected=final_dampers,
                final=final_dampers,
                previous=self.previous_final_damper_scales,
                projection_active=False)
            self.previous_final_damper_scales = list(final_dampers)
            self._ensure_required_diagnostics(diagnostics)
            return ControllerOutput(command=command, diagnostics=diagnostics)

        selected = optimizer.selected
        projected, projection_active = self._semi_active_project_dampers(
            selected.desired_dampers,
            context)
        final_dampers, final_guard_active = self._final_safety_dampers(
            projected,
            context)
        if not _all_finite(final_dampers):
            final_dampers = tuple(1.0 for _ in range(wheel_count))
            projected = final_dampers
            diagnostics.update({
                "planning_aware_v3_valid": 0,
                "planning_aware_v3_mode": "identity_fallback",
                "planning_aware_v3_fallback_reason": (
                    "final_nonfinite_safety_guard"),
                "mpc_final_finite_guard_active": 1,
            })
        else:
            diagnostics.update({
                "planning_aware_v3_mode": "mpc_primary",
                "planning_aware_v3_fallback_reason": "none",
                "mpc_final_finite_guard_active": 0,
            })

        command = self._command_from_dampers(final_dampers)
        diagnostics.update({
            "mpc_u_mean": selected.u_mean,
            "mpc_u_roll_front": selected.u_roll_front,
            "mpc_u_roll_rear": selected.u_roll_rear,
            "mpc_u_pitch": selected.u_pitch,
            "mpc_ltr_proxy_peak": selected.ltr_proxy_peak,
            "mpc_projection_active": int(projection_active),
            "mpc_final_rate_guard_active": int(final_guard_active),
        })
        diagnostics.update({
            "mpc_b2d_threshold_cost": selected.components.get("b2d", 0.0),
            "mpc_body_cost": selected.components.get("body", 0.0),
            "mpc_ltr_cost": selected.components.get("ltr", 0.0),
            "mpc_command_effort_cost": selected.components.get("command", 0.0),
            "mpc_slew_cost": selected.components.get("slew", 0.0),
            "mpc_mean_damper_cost": selected.components.get("mean", 0.0),
            "mpc_semi_active_feasibility_cost": selected.components.get(
                "feasible",
                0.0),
            "skyhook_prior_penalty": selected.components.get("prior", 0.0),
        })
        self._set_command_diagnostics(
            diagnostics,
            desired=selected.desired_dampers,
            projected=projected,
            final=final_dampers,
            previous=self.previous_final_damper_scales,
            projection_active=projection_active)
        self.previous_final_damper_scales = list(final_dampers)
        self._ensure_required_diagnostics(diagnostics)
        return ControllerOutput(command=command, diagnostics=diagnostics)

    def _shadow_config(self) -> SkyhookRollV3CanonicalModalConfig:
        allowed = {item.name for item in fields(SkyhookRollV3CanonicalModalConfig)}
        values = {
            name: getattr(self.config, name)
            for name in allowed
            if hasattr(self.config, name)
        }
        return SkyhookRollV3CanonicalModalConfig.from_mapping(values)

    def _projection_config(self) -> SkyhookConfig:
        allowed = {item.name for item in fields(SkyhookConfig)}
        values = {
            name: getattr(self.config, name)
            for name in allowed
            if hasattr(self.config, name)
        }
        return SkyhookConfig.from_mapping(values)

    def _wheel_count(self, context: ControllerContext) -> int:
        for values in (
                getattr(context, "native_damper_rate_by_wheel", ()),
                getattr(context, "native_spring_strength_by_wheel", ())):
            if values:
                return max(1, len(values))
        state = getattr(context, "suspension_state", None)
        wheels = tuple(getattr(state, "wheels", ()) or ())
        if wheels:
            return max(1, len(wheels))
        return max(1, int(_safe_float(self.config.wheel_count, 4)))

    def _wheel_count_from_native(self, native_suspension: Any) -> int:
        wheels = tuple(getattr(native_suspension, "wheels", ()) or ())
        return len(wheels)

    def _ensure_previous_count(self, wheel_count: int) -> None:
        self.previous_final_damper_scales = list(_match_length(
            self.previous_final_damper_scales,
            wheel_count,
            1.0))

    def _shadow_output_usable(
        self,
        output: ControllerOutput,
        wheel_count: int,
    ) -> Tuple[bool, str]:
        diagnostics = dict(output.diagnostics or {})
        if int(diagnostics.get("identity_fallback_this_tick", 0) or 0):
            return False, str(diagnostics.get(
                "identity_fallback_reason",
                "identity_fallback"))
        try:
            output.command.validate(expected_wheels=wheel_count)
        except ValueError as exc:
            return False, str(exc)
        dampers = self._command_dampers(output.command, wheel_count, default=math.nan)
        if not _all_finite(dampers):
            return False, "nonfinite_shadow_command"
        return True, ""

    def _base_diagnostics(
        self,
        *,
        preview: PlanningAwareV3Preview,
        comfort: ComfortGuardResult,
        shadow_output: ControllerOutput,
        shadow_dampers: Sequence[float],
        wheel_count: int,
    ) -> Dict[str, Any]:
        diagnostics: Dict[str, Any] = {}
        diagnostics.update(preview.as_diagnostics())
        diagnostics.update(comfort.as_diagnostics())
        diagnostics.update({
            "controller": self.name,
            "controller_name": self.name,
            "controller_version": self.config.controller_version,
            "planning_aware_v3_mode": (
                "preview" if preview.valid else "fallback"),
            "planning_aware_v3_fallback_reason": preview.fallback_reason,
            "mpc_candidate_count": 0,
            "mpc_feasible_candidate_count": 0,
            "mpc_best_cost": 0.0,
            "mpc_second_best_cost": 0.0,
            "mpc_cost_margin": 0.0,
            "mpc_u_mean": 0.0,
            "mpc_u_roll_front": 0.0,
            "mpc_u_roll_rear": 0.0,
            "mpc_u_pitch": 0.0,
            "mpc_ltr_proxy_peak": 0.0,
            "mpc_rate_constraint_active": 0,
            "mpc_projection_active": 0,
            "mpc_final_rate_guard_active": 0,
            "mpc_final_finite_guard_active": 0,
            "mpc_ltr_soft_guard_active": 0,
            "mpc_ltr_hard_guard_active": 0,
            "mpc_slew_weight_scheduled": (
                self.config.W_slew * comfort.slew_penalty_multiplier),
            "mpc_mean_damper_weight_scheduled": (
                self.config.W_mean *
                comfort.mean_damper_penalty_multiplier),
            "skyhook_prior_penalty": 0.0,
            "spring_scale": 1.0,
            "damper_scale": 1.0,
            "planning_aware_v3_shadow_identity_fallback": int(
                dict(shadow_output.diagnostics or {}).get(
                    "identity_fallback_this_tick",
                    0) or 0),
            "planning_aware_v3_shadow_fallback_reason": str(
                dict(shadow_output.diagnostics or {}).get(
                    "identity_fallback_reason",
                    "")),
            "planning_aware_v3_outer_side_sign_from_ay": (
                self.config.outer_side_sign_from_ay),
        })
        for index, label in enumerate(_WHEEL_LABELS[:wheel_count]):
            value = shadow_dampers[index] if index < len(shadow_dampers) else 1.0
            diagnostics["skyhook_roll_v3_shadow_damper_%s" % label] = (
                _safe_float(value, 1.0))
        return diagnostics

    def _optimizer_diagnostics(
        self,
        optimizer: _MpcOptimizationResult,
        comfort: ComfortGuardResult,
    ) -> Dict[str, Any]:
        second = optimizer.second_best_cost
        best = optimizer.best_cost
        margin = optimizer.cost_margin
        return {
            "mpc_candidate_count": optimizer.candidate_count,
            "mpc_feasible_candidate_count": optimizer.feasible_candidate_count,
            "mpc_best_cost": _safe_float(best, 0.0),
            "mpc_second_best_cost": _safe_float(second, 0.0),
            "mpc_cost_margin": _safe_float(margin, 0.0),
            "mpc_rate_constraint_active": int(
                optimizer.rate_rejection_count > 0),
            "mpc_rate_rejection_count": optimizer.rate_rejection_count,
            "mpc_ltr_soft_guard_active": int(optimizer.ltr_soft_guard_active),
            "mpc_ltr_hard_guard_active": int(optimizer.ltr_hard_guard_active),
            "mpc_optimizer_failed_reason": optimizer.failed_reason,
            "mpc_slew_weight_scheduled": (
                self.config.W_slew * comfort.slew_penalty_multiplier),
            "mpc_mean_damper_weight_scheduled": (
                self.config.W_mean *
                comfort.mean_damper_penalty_multiplier),
        }

    def _optimize_mpc(
        self,
        *,
        context: ControllerContext,
        preview: PlanningAwareV3Preview,
        comfort: ComfortGuardResult,
        shadow_dampers: Sequence[float],
    ) -> _MpcOptimizationResult:
        return self._generate_v3_raw_mpc_grid_candidates(
            context=context,
            preview=preview,
            comfort=comfort,
            shadow_dampers=shadow_dampers)

    def _generate_v3_raw_mpc_grid_candidates(
        self,
        *,
        context: ControllerContext,
        preview: PlanningAwareV3Preview,
        comfort: ComfortGuardResult,
        shadow_dampers: Sequence[float],
        top_k: Optional[int] = None,
    ) -> _MpcOptimizationResult:
        cfg = self.config
        controller_dt = _context_dt(context, cfg.default_dt)
        max_up = max(0.0, _safe_float(cfg.max_rate_up_scale_per_s, 1.2)) * controller_dt
        max_down = max(0.0, _safe_float(cfg.max_rate_down_scale_per_s, 1.2)) * controller_dt
        rate_delta = max(_EPS, min(max_up, max_down))
        ay_peak = _signed_peak(preview.predicted_ay)
        open_loop_ltr = self._ltr_proxy_peak(
            preview=preview,
            roll_values=None)
        ltr_soft = open_loop_ltr >= _safe_float(cfg.ltr_soft_threshold, 0.45)
        ltr_hard = (
            not math.isfinite(open_loop_ltr) or
            open_loop_ltr >= _safe_float(cfg.ltr_hard_threshold, 0.75))
        if ltr_hard:
            return _MpcOptimizationResult(
                selected=None,
                candidate_count=0,
                feasible_candidate_count=0,
                best_cost=0.0,
                second_best_cost=0.0,
                cost_margin=0.0,
                failed_reason="ltr_hard_threshold",
                rate_rejection_count=0,
                ltr_soft_guard_active=ltr_soft,
                ltr_hard_guard_active=True,
                raw_candidates=())

        u_mean_values = self._rate_refined_grid(cfg.u_mean_grid, rate_delta)
        u_front_values = self._rate_refined_grid(
            cfg.u_roll_front_grid,
            rate_delta)
        u_rear_values = self._rate_refined_grid(
            cfg.u_roll_rear_grid,
            rate_delta)
        u_pitch_values = self._rate_refined_grid(cfg.u_pitch_grid, rate_delta)

        candidate_count = 0
        feasible_count = 0
        rate_rejections = 0
        previous = tuple(self.previous_final_damper_scales)
        raw_candidates: List[MpcDamperCandidate] = []

        for u_mean in u_mean_values:
            for u_roll_front in u_front_values:
                for u_roll_rear in u_rear_values:
                    for u_pitch in u_pitch_values:
                        candidate_count += 1
                        if (
                                ltr_soft and
                                abs(ay_peak) > _EPS and
                                (u_roll_front < -_EPS or
                                 u_roll_rear < -_EPS)):
                            continue
                        desired = self._modal_to_dampers(
                            preview=preview,
                            u_mean=u_mean,
                            u_roll_front=u_roll_front,
                            u_roll_rear=u_roll_rear,
                            u_pitch=u_pitch)
                        if not _all_finite(desired):
                            continue
                        if not self._within_damper_limits(desired):
                            continue
                        if not self._within_rate_limits(
                                desired,
                                previous,
                                max_up,
                                max_down):
                            rate_rejections += 1
                            continue
                        feasible_count += 1
                        cost, components, ltr_peak = self._candidate_cost(
                            context=context,
                            preview=preview,
                            desired=desired,
                            u_mean=u_mean,
                            u_roll_front=u_roll_front,
                            u_roll_rear=u_roll_rear,
                            u_pitch=u_pitch,
                            comfort=comfort,
                            shadow_dampers=shadow_dampers,
                            ltr_soft_active=ltr_soft)
                        candidate = MpcDamperCandidate(
                            kind="mpc_grid",
                            raw_dampers=desired,
                            u_mean=u_mean,
                            u_roll_front=u_roll_front,
                            u_roll_rear=u_roll_rear,
                            u_pitch=u_pitch,
                            raw_cost=cost,
                            components=components,
                            ltr_proxy_peak=ltr_peak,
                            raw_index=candidate_count - 1)
                        raw_candidates.append(candidate)

        sorted_candidates = tuple(sorted(
            raw_candidates,
            key=self._candidate_sort_key_for_raw_cost))
        if not sorted_candidates:
            return _MpcOptimizationResult(
                selected=None,
                candidate_count=candidate_count,
                feasible_candidate_count=feasible_count,
                best_cost=0.0,
                second_best_cost=0.0,
                cost_margin=0.0,
                failed_reason="all_candidates_infeasible",
                rate_rejection_count=rate_rejections,
                ltr_soft_guard_active=ltr_soft,
                ltr_hard_guard_active=False,
                raw_candidates=())

        selected = replace(sorted_candidates[0], selected=True)
        second = (
            sorted_candidates[1].cost
            if len(sorted_candidates) > 1 else selected.cost)
        retained = (selected,) + sorted_candidates[1:]
        if top_k is not None:
            limit = max(2, int(_safe_float(top_k, 2)))
            retained = retained[:limit]
        return _MpcOptimizationResult(
            selected=selected,
            candidate_count=candidate_count,
            feasible_candidate_count=feasible_count,
            best_cost=selected.cost,
            second_best_cost=second,
            cost_margin=max(0.0, second - selected.cost),
            failed_reason="",
            rate_rejection_count=rate_rejections,
            ltr_soft_guard_active=ltr_soft,
            ltr_hard_guard_active=False,
            raw_candidates=retained)

    def _candidate_sort_key_for_raw_cost(
        self,
        candidate: MpcDamperCandidate,
    ) -> Tuple[float, int]:
        return (
            _safe_float(candidate.raw_cost, 1.0e12),
            int(_safe_float(candidate.raw_index, 0)))

    def _candidate_cost(
        self,
        *,
        context: ControllerContext,
        preview: PlanningAwareV3Preview,
        desired: Sequence[float],
        u_mean: float,
        u_roll_front: float,
        u_roll_rear: float,
        u_pitch: float,
        comfort: ComfortGuardResult,
        shadow_dampers: Sequence[float],
        ltr_soft_active: bool,
    ) -> Tuple[float, Mapping[str, float], float]:
        cfg = self.config
        state = getattr(context, "state", VehicleState())
        model_dt = max(_safe_float(preview.dt, cfg.model_dt), _EPS)
        phi = math.radians(_safe_float(getattr(state, "roll", 0.0)))
        phi_dot = math.radians(_safe_float(getattr(state, "roll_rate", 0.0)))
        theta = math.radians(_safe_float(getattr(state, "pitch", 0.0)))
        theta_dot = math.radians(_safe_float(getattr(state, "pitch_rate", 0.0)))

        ay_peak = _signed_peak(preview.predicted_ay)
        ax_min = min(preview.predicted_ax) if preview.predicted_ax else 0.0
        ax_max = max(preview.predicted_ax) if preview.predicted_ax else 0.0
        pitch_need_sign = 0.0
        if abs(ax_min) >= abs(ax_max) and ax_min < -_EPS:
            pitch_need_sign = 1.0
        elif ax_max > _EPS:
            pitch_need_sign = -1.0

        mean_damper = sum(desired) / float(len(desired))
        roll_control = 0.55 * max(0.0, u_roll_front) + 0.45 * max(
            0.0,
            u_roll_rear)
        pitch_control = max(0.0, pitch_need_sign * u_pitch)
        zeta_roll = max(
            0.05,
            cfg.zeta_roll_base +
            cfg.zeta_mean_gain * max(0.0, mean_damper - 1.0) +
            cfg.zeta_roll_gain * roll_control)
        zeta_pitch = max(
            0.05,
            cfg.zeta_pitch_base +
            cfg.zeta_mean_gain * max(0.0, mean_damper - 1.0) +
            cfg.zeta_pitch_gain * pitch_control)
        omega_roll = max(_safe_float(cfg.omega_roll_rad_s, 7.0), _EPS)
        omega_pitch = max(_safe_float(cfg.omega_pitch_rad_s, 8.0), _EPS)

        b2d = 0.0
        body = 0.0
        ltr = 0.0
        ltr_peak = 0.0
        previous_lon: Optional[float] = None
        previous_lat: Optional[float] = None
        roll_values: List[float] = []
        count = max(1, len(preview.times))

        for index in range(count):
            ax = preview.predicted_ax[index]
            ay = preview.predicted_ay[index]
            phi_eq = cfg.K_phi_ay * ay
            theta_eq = cfg.K_theta_ax * ax
            phi_ddot = (
                -2.0 * zeta_roll * omega_roll * phi_dot -
                omega_roll * omega_roll * (phi - phi_eq))
            theta_ddot = (
                -2.0 * zeta_pitch * omega_pitch * theta_dot -
                omega_pitch * omega_pitch * (theta - theta_eq))
            lon_acc = (
                ax +
                cfg.k_theta_acc * theta_ddot +
                cfg.k_theta_g * cfg.g * math.sin(theta))
            lat_acc = (
                ay +
                cfg.k_phi_acc * phi_ddot +
                cfg.k_phi_g * cfg.g * math.sin(phi))
            if previous_lon is None:
                lon_jerk = 0.0
                mag_jerk = 0.0
            else:
                lon_jerk = (lon_acc - previous_lon) / model_dt
                lat_jerk = (lat_acc - previous_lat) / model_dt
                mag_jerk = math.hypot(lon_jerk, lat_jerk)
            yaw_rate = (
                preview.yaw_rate_ref[index]
                if index < len(preview.yaw_rate_ref) else 0.0)
            yaw_acc = (
                preview.yaw_acc_ref[index]
                if index < len(preview.yaw_acc_ref) else 0.0)
            b2d += (
                _hinge(lon_acc - cfg.comfort_lon_acc_upper) ** 2 +
                _hinge(cfg.comfort_lon_acc_lower - lon_acc) ** 2 +
                _hinge(abs(lat_acc) - cfg.comfort_abs_lat_acc) ** 2 +
                _hinge(abs(lon_jerk) - cfg.comfort_abs_lon_jerk) ** 2 +
                _hinge(abs(mag_jerk) - cfg.comfort_abs_mag_jerk) ** 2 +
                _hinge(abs(yaw_acc) - cfg.comfort_abs_yaw_acc) ** 2 +
                _hinge(abs(yaw_rate) - cfg.comfort_abs_yaw_rate) ** 2)
            ltr_value = self._ltr_proxy_value(ay=ay, roll_phi=phi)
            ltr_peak = max(ltr_peak, ltr_value)
            ltr += _hinge(ltr_value - cfg.ltr_soft_threshold) ** 2
            body += (
                phi * phi +
                0.08 * phi_dot * phi_dot +
                theta * theta +
                0.08 * theta_dot * theta_dot)
            roll_values.append(phi)
            phi += model_dt * phi_dot
            phi_dot += model_dt * phi_ddot
            theta += model_dt * theta_dot
            theta_dot += model_dt * theta_ddot
            previous_lon = lon_acc
            previous_lat = lat_acc

        roll_demand = clamp(
            abs(ay_peak) / max(cfg.roll_preview_full_lat_acc, _EPS) *
            cfg.roll_preview_max_u,
            0.0,
            cfg.roll_preview_max_u)
        pitch_demand = clamp(
            max(abs(ax_min), abs(ax_max)) /
            max(cfg.pitch_preview_full_decel, _EPS) *
            cfg.pitch_preview_max_u,
            0.0,
            cfg.pitch_preview_max_u)
        roll_tracking = (
            (max(0.0, u_roll_front) - roll_demand) ** 2 +
            (max(0.0, u_roll_rear) - roll_demand) ** 2)
        pitch_tracking = (
            (pitch_need_sign * u_pitch - pitch_demand) ** 2
            if pitch_need_sign else u_pitch * u_pitch)
        if ltr_soft_active:
            roll_tracking *= (1.0 + cfg.ltr_soft_roll_weight_gain)

        previous = tuple(self.previous_final_damper_scales)
        command = sum((value - 1.0) ** 2 for value in desired)
        slew = sum(
            (value - previous[index]) ** 2
            for index, value in enumerate(desired))
        mean = (
            (mean_damper - 1.0) ** 2 +
            2.0 * _hinge(mean_damper - 1.0) ** 2)
        feasible = self._semi_active_feasibility_cost(desired, context)
        prior = 0.0
        if bool(cfg.use_skyhook_prior) and cfg.W_prior > 0.0:
            shadow = _match_length(shadow_dampers, len(desired), 1.0)
            prior = sum(
                (value - shadow[index]) ** 2
                for index, value in enumerate(desired))

        inv_count = 1.0 / float(count)
        components = {
            "b2d": b2d * inv_count,
            "body": body * inv_count + cfg.W_roll_preview * roll_tracking +
            cfg.W_pitch_preview * pitch_tracking,
            "ltr": ltr * inv_count,
            "command": command,
            "slew": slew,
            "mean": mean,
            "feasible": feasible,
            "prior": prior,
        }
        total = (
            cfg.W_b2d * components["b2d"] +
            cfg.W_body * components["body"] +
            cfg.W_ltr * components["ltr"] +
            cfg.W_command * components["command"] +
            cfg.W_slew * comfort.slew_penalty_multiplier *
            components["slew"] +
            cfg.W_mean * comfort.mean_damper_penalty_multiplier *
            components["mean"] +
            cfg.W_feasible * components["feasible"] +
            (cfg.W_prior if bool(cfg.use_skyhook_prior) else 0.0) *
            components["prior"])
        return _safe_float(total, 1.0e12), components, ltr_peak

    def _modal_to_dampers(
        self,
        *,
        preview: PlanningAwareV3Preview,
        u_mean: float,
        u_roll_front: float,
        u_roll_rear: float,
        u_pitch: float,
    ) -> Tuple[float, ...]:
        side_signs = self._side_signs(preview)
        pitch_signs = self._pitch_signs()
        return (
            1.0 + u_mean + side_signs[0] * u_roll_front +
            pitch_signs[0] * u_pitch,
            1.0 + u_mean + side_signs[1] * u_roll_front +
            pitch_signs[1] * u_pitch,
            1.0 + u_mean + side_signs[2] * u_roll_rear +
            pitch_signs[2] * u_pitch,
            1.0 + u_mean + side_signs[3] * u_roll_rear +
            pitch_signs[3] * u_pitch,
        )

    def _side_signs(self, preview: PlanningAwareV3Preview) -> Tuple[float, ...]:
        ay_peak = _signed_peak(preview.predicted_ay)
        corners = self._corner_coordinates()
        if abs(ay_peak) <= _EPS:
            return tuple(_sign(y) for _, y in corners)
        outer_side = (
            _sign(self.config.outer_side_sign_from_ay) *
            _sign(ay_peak))
        if outer_side == 0.0:
            outer_side = -1.0
        return tuple(1.0 if _sign(y) == outer_side else -1.0
                     for _, y in corners)

    def _pitch_signs(self) -> Tuple[float, ...]:
        return tuple(_sign(x) for x, _ in self._corner_coordinates())

    def _rate_refined_grid(
        self,
        values: Sequence[float],
        rate_delta: float,
    ) -> Tuple[float, ...]:
        raw = [_safe_float(value) for value in values]
        raw.extend([
            -rate_delta,
            -0.5 * rate_delta,
            0.0,
            0.5 * rate_delta,
            rate_delta,
        ])
        return tuple(sorted({round(value, 8) for value in raw}))

    def _within_damper_limits(self, dampers: Sequence[float]) -> bool:
        low = _safe_float(self.config.min_damper_scale, 0.75)
        high = _safe_float(self.config.max_damper_scale, 1.25)
        return all(low - 1.0e-12 <= value <= high + 1.0e-12
                   for value in dampers)

    def _within_rate_limits(
        self,
        desired: Sequence[float],
        previous: Sequence[float],
        max_up: float,
        max_down: float,
    ) -> bool:
        for index, value in enumerate(desired):
            prev = previous[index] if index < len(previous) else 1.0
            if value - prev > max_up + 1.0e-12:
                return False
            if prev - value > max_down + 1.0e-12:
                return False
        return True

    def _candidate_bounds_and_rate_dampers(
        self,
        raw: Sequence[float],
        previous: Sequence[float],
        context: ControllerContext,
    ) -> Tuple[Tuple[float, ...], bool]:
        cfg = self.config
        raw_values = tuple(raw or ())
        previous_values = tuple(previous or ())
        wheel_count = len(raw_values) if raw_values else len(previous_values)
        if wheel_count <= 0:
            wheel_count = max(1, int(_safe_float(cfg.wheel_count, 4)))
        controller_dt = _context_dt(context, cfg.default_dt)
        max_up = max(
            0.0,
            _safe_float(cfg.max_rate_up_scale_per_s, 1.2)) * controller_dt
        max_down = max(
            0.0,
            _safe_float(cfg.max_rate_down_scale_per_s, 1.2)) * controller_dt
        low = _safe_float(cfg.min_damper_scale, 0.75)
        high = _safe_float(cfg.max_damper_scale, 1.25)
        previous_matched = _match_length(previous_values, wheel_count, 1.0)
        constrained: List[float] = []
        guard_active = False
        for index in range(wheel_count):
            raw_value = raw_values[index] if index < len(raw_values) else 1.0
            finite = _safe_float(raw_value, 1.0)
            bounded = clamp(finite, low, high)
            limited = clamp(
                bounded,
                previous_matched[index] - max_down,
                previous_matched[index] + max_up)
            raw_finite = _optional_finite_float(raw_value)
            if raw_finite is None or not math.isclose(
                    limited,
                    raw_finite,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12):
                guard_active = True
            constrained.append(limited)
        return tuple(constrained), guard_active

    def _evaluate_candidate_as_applied(
        self,
        candidate: MpcDamperCandidate,
        context: ControllerContext,
        preview: PlanningAwareV3Preview,
        comfort: ComfortGuardResult,
        shadow_dampers: Sequence[float],
    ) -> MpcDamperCandidate:
        wheel_count = len(self.previous_final_damper_scales)
        raw_dampers = _match_length(candidate.raw_dampers, wheel_count, 1.0)
        constrained, _ = self._candidate_bounds_and_rate_dampers(
            raw_dampers,
            self.previous_final_damper_scales,
            context)
        projected, _ = self._semi_active_project_dampers(constrained, context)
        final, _ = self._final_safety_dampers(projected, context)
        rejection_reason = candidate.rejection_reason
        if not _all_finite(final):
            final = tuple(1.0 for _ in range(wheel_count))
            projected = final
            rejection_reason = "final_nonfinite_safety_guard"
        modal = self._effective_modal_from_dampers(final, preview)
        open_loop_ltr = self._ltr_proxy_peak(preview=preview, roll_values=None)
        ltr_soft = open_loop_ltr >= _safe_float(
            self.config.ltr_soft_threshold,
            0.45)
        projected_cost, projected_components, projected_ltr = (
            self._candidate_cost(
                context=context,
                preview=preview,
                desired=final,
                u_mean=modal[0],
                u_roll_front=modal[1],
                u_roll_rear=modal[2],
                u_pitch=modal[3],
                comfort=comfort,
                shadow_dampers=shadow_dampers,
                ltr_soft_active=ltr_soft))
        return replace(
            candidate,
            raw_dampers=raw_dampers,
            constrained_dampers=constrained,
            projected_dampers=projected,
            final_dampers=final,
            projected_cost=projected_cost,
            projected_components=projected_components,
            projected_ltr_proxy_peak=projected_ltr,
            rejection_reason=rejection_reason)

    def _effective_modal_from_dampers(
        self,
        dampers: Sequence[float],
        preview: PlanningAwareV3Preview,
    ) -> Tuple[float, float, float, float]:
        values = _match_length(dampers, 4, 1.0)
        side_signs = self._side_signs(preview)
        pitch_signs = self._pitch_signs()
        u_mean = _mean(values) - 1.0
        front_roll_den = side_signs[0] - side_signs[1]
        rear_roll_den = side_signs[2] - side_signs[3]
        front_pitch = 0.5 * (pitch_signs[0] + pitch_signs[1])
        rear_pitch = 0.5 * (pitch_signs[2] + pitch_signs[3])
        pitch_den = front_pitch - rear_pitch
        u_roll_front = (
            (values[0] - values[1]) / front_roll_den
            if abs(front_roll_den) > _EPS else 0.0)
        u_roll_rear = (
            (values[2] - values[3]) / rear_roll_den
            if abs(rear_roll_den) > _EPS else 0.0)
        u_pitch = (
            (0.5 * (values[0] + values[1]) -
             0.5 * (values[2] + values[3])) / pitch_den
            if abs(pitch_den) > _EPS else 0.0)
        return (
            _safe_float(u_mean),
            _safe_float(u_roll_front),
            _safe_float(u_roll_rear),
            _safe_float(u_pitch))

    def _candidate_sort_key_for_projected_cost(
        self,
        candidate: MpcDamperCandidate,
    ) -> Tuple[int, int, float, float, int]:
        projected_cost = _safe_float(
            candidate.projected_cost,
            _safe_float(candidate.raw_cost, 1.0e12))
        cost_bucket = int(round(projected_cost / 1.0e-9))
        kind_priority = {
            "previous_hold": 0,
            "neutral_return": 1,
        }.get(candidate.kind, 2)
        final = (
            candidate.final_dampers or
            candidate.projected_dampers or
            candidate.constrained_dampers or
            candidate.raw_dampers)
        previous = _match_length(
            self.previous_final_damper_scales,
            len(final),
            1.0)
        final_to_previous = (
            self._l2_distance(final, previous)
            if candidate.kind == "mpc_grid" else 0.0)
        return (
            cost_bucket,
            kind_priority,
            final_to_previous,
            _safe_float(candidate.raw_cost, 1.0e12),
            int(_safe_float(candidate.raw_index, 0)))

    def _l2_distance(
        self,
        left: Sequence[float],
        right: Sequence[float],
    ) -> float:
        length = max(len(left), len(right))
        left_values = _match_length(left, length, 1.0)
        right_values = _match_length(right, length, 1.0)
        return math.sqrt(sum(
            (left_values[index] - right_values[index]) ** 2
            for index in range(length)))

    def _semi_active_project_dampers(
        self,
        desired: Sequence[float],
        context: ControllerContext,
    ) -> Tuple[Tuple[float, ...], bool]:
        cfg = self.config
        projection_config = self._projection_config()
        native_dampers = self._native_dampers(context, len(desired))
        rel_velocities = self._relative_extension_velocities(context, len(desired))
        corner_velocities = self._corner_vertical_velocities(context.state)
        corner_velocities = _match_length(corner_velocities, len(desired), 0.0)
        projected: List[float] = []
        active = False
        neutral = _safe_float(cfg.neutral_damper_scale, 1.0)
        low_rel = max(_safe_float(cfg.rel_velocity_deadband, 0.015),
                      _safe_float(cfg.projection_eps, 1.0e-6))
        for index, desired_value in enumerate(desired):
            native = max(_safe_float(native_dampers[index], 4500.0), _EPS)
            v_eff = _safe_float(corner_velocities[index], 0.0)
            v_rel = _safe_float(rel_velocities[index], 0.0)
            c_sky = native * max(abs(desired_value - neutral), 0.05)
            result = canonical_skyhook_v3_projection(
                v_eff_i=v_eff,
                v_rel_extension_i=v_rel,
                C_native_i=native,
                C_sky_i=c_sky,
                config=projection_config)
            target = _safe_float(result["final_target_damper"], neutral)
            value = _safe_float(desired_value, neutral)
            if abs(v_rel) <= low_rel:
                blend = clamp(cfg.projection_blend_low_rel, 0.0, 1.0)
                value = value + blend * (neutral - value)
            elif int(result.get("semi_active_feasible", 0) or 0) != 1:
                blend = clamp(cfg.projection_blend_infeasible, 0.0, 1.0)
                value = value + blend * (target - value)
            value = clamp(
                value,
                _safe_float(cfg.min_damper_scale, 0.75),
                _safe_float(cfg.max_damper_scale, 1.25))
            if not math.isclose(
                    value,
                    desired_value,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12):
                active = True
            projected.append(value)
        return tuple(projected), active

    def _final_safety_dampers(
        self,
        projected: Sequence[float],
        context: ControllerContext,
    ) -> Tuple[Tuple[float, ...], bool]:
        return self._candidate_bounds_and_rate_dampers(
            projected,
            self.previous_final_damper_scales,
            context)

    def _semi_active_feasibility_cost(
        self,
        desired: Sequence[float],
        context: ControllerContext,
    ) -> float:
        rel_velocities = self._relative_extension_velocities(context, len(desired))
        cost = 0.0
        deadband = max(_safe_float(self.config.rel_velocity_deadband, 0.015), _EPS)
        for desired_value, v_rel in zip(desired, rel_velocities):
            deviation = desired_value - 1.0
            if abs(v_rel) < deadband:
                cost += deviation * deviation
        return cost

    def _ltr_proxy_peak(
        self,
        *,
        preview: PlanningAwareV3Preview,
        roll_values: Optional[Sequence[float]],
    ) -> float:
        peak = 0.0
        for index, ay in enumerate(preview.predicted_ay):
            roll_phi = (
                roll_values[index]
                if roll_values is not None and index < len(roll_values)
                else 0.0)
            peak = max(peak, self._ltr_proxy_value(ay=ay, roll_phi=roll_phi))
        return peak

    def _ltr_proxy_value(self, *, ay: float, roll_phi: float) -> float:
        cfg = self.config
        track_width = max(2.0 * abs(_safe_float(cfg.half_track_m, 0.85)), _EPS)
        value = (
            (2.0 * _safe_float(cfg.h_cg_proxy_m, 0.55) /
             (track_width * max(_safe_float(cfg.g, 9.81), _EPS))) *
            (abs(_safe_float(ay, 0.0)) +
             _safe_float(cfg.g, 9.81) * abs(math.sin(_safe_float(roll_phi)))))
        return _safe_float(value, 1.0e12)

    def _set_command_diagnostics(
        self,
        diagnostics: Dict[str, Any],
        *,
        desired: Sequence[float],
        projected: Sequence[float],
        final: Sequence[float],
        previous: Sequence[float],
        projection_active: bool,
    ) -> None:
        wheel_count = len(final)
        desired = _match_length(desired, wheel_count, 1.0)
        projected = _match_length(projected, wheel_count, 1.0)
        previous = _match_length(previous, wheel_count, 1.0)
        for index, label in enumerate(_WHEEL_LABELS[:wheel_count]):
            diagnostics["desired_damper_%s" % label] = desired[index]
            diagnostics["projected_damper_%s" % label] = projected[index]
            diagnostics["final_damper_%s" % label] = final[index]
            diagnostics["final_spring_%s" % label] = _FROZEN_SPRING_SCALE
            diagnostics["damper_delta_%s" % label] = final[index] - previous[index]
        diagnostics.update({
            "mpc_projection_active": int(projection_active),
            "mean_damper": _mean(final),
            "damper_scale": _mean(final),
            "spring_scale": _FROZEN_SPRING_SCALE,
            "front_roll_contrast": (
                final[0] - final[1] if wheel_count >= 2 else 0.0),
            "rear_roll_contrast": (
                final[2] - final[3] if wheel_count >= 4 else 0.0),
            "pitch_contrast": (
                ((final[0] + final[1]) * 0.5 -
                 (final[2] + final[3]) * 0.5)
                if wheel_count >= 4 else 0.0),
        })

    def _ensure_required_diagnostics(self, diagnostics: Dict[str, Any]) -> None:
        for key in PLANNING_AWARE_V3_REQUIRED_DIAGNOSTIC_FIELDS:
            if key in diagnostics:
                continue
            if (
                    key.endswith("_source") or
                    key.endswith("_reason") or
                    key.endswith("_mode")):
                diagnostics[key] = ""
            else:
                diagnostics[key] = 0.0

    def _command_from_dampers(
        self,
        dampers: Sequence[float],
    ) -> SuspensionCommand:
        return SuspensionCommand(tuple(
            WheelScale(
                spring_scale=_FROZEN_SPRING_SCALE,
                damper_scale=_safe_float(damper, 1.0))
            for damper in dampers
        )).validate(expected_wheels=len(dampers))

    def _command_dampers(
        self,
        command: SuspensionCommand,
        wheel_count: int,
        default: float,
    ) -> Tuple[float, ...]:
        wheels = tuple(getattr(command, "wheels", ()) or ())
        values = tuple(
            _safe_float(getattr(wheel, "damper_scale", default), default)
            for wheel in wheels)
        return _match_length(values, wheel_count, default)

    def _clamped_finite_dampers(
        self,
        dampers: Sequence[float],
        wheel_count: int,
    ) -> Tuple[float, ...]:
        values = _match_length(dampers, wheel_count, 1.0)
        return tuple(
            clamp(
                _safe_float(value, 1.0),
                _safe_float(self.config.min_damper_scale, 0.75),
                _safe_float(self.config.max_damper_scale, 1.25))
            for value in values)

    def _native_dampers(
        self,
        context: ControllerContext,
        wheel_count: int,
    ) -> Tuple[float, ...]:
        values = tuple(getattr(context, "native_damper_rate_by_wheel", ()) or ())
        if not values:
            native = getattr(context, "native_suspension", None)
            wheels = tuple(getattr(native, "wheels", ()) or ())
            values = tuple(
                _safe_float(getattr(wheel, "spring_damper_rate", 4500.0), 4500.0)
                for wheel in wheels)
        return _match_length(values, wheel_count, 4500.0)

    def _relative_extension_velocities(
        self,
        context: ControllerContext,
        wheel_count: int,
    ) -> Tuple[float, ...]:
        state = getattr(context, "suspension_state", None)
        wheels = tuple(getattr(state, "wheels", ()) or ())
        sign = _safe_float(self.config.relative_extension_sign, -1.0)
        values = tuple(
            sign * _safe_float(
                getattr(wheel, "suspension_velocity_mps", 0.0),
                0.0)
            for wheel in wheels)
        return _match_length(values, wheel_count, 0.0)

    def _corner_vertical_velocities(
        self,
        state: VehicleState,
    ) -> Tuple[float, ...]:
        corners = self._corner_coordinates()
        roll_rate = math.radians(_safe_float(getattr(state, "roll_rate", 0.0)))
        pitch_rate = math.radians(_safe_float(getattr(state, "pitch_rate", 0.0)))
        vz = _safe_float(getattr(state, "vz", 0.0), 0.0)
        return tuple(
            vz + roll_rate * y - pitch_rate * x
            for x, y in corners)

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


def build_planning_preview(
    context: ControllerContext,
    config: Optional[PlanningAwareV3MpcPrimaryConfig] = None,
) -> PlanningAwareV3Preview:
    """Build a finite fixed-step horizon from one ``ControllerContext``."""

    cfg = config or PlanningAwareV3MpcPrimaryConfig()
    model_dt = _positive_or_default(cfg.model_dt, cfg.default_dt)
    horizon_s = min(
        max(_safe_float(cfg.horizon_s, 1.5), model_dt),
        max(_safe_float(cfg.max_horizon_s, 2.0), model_dt))
    step_count = max(2, int(math.floor(horizon_s / model_dt + _EPS)) + 1)
    times = tuple(index * model_dt for index in range(step_count))
    planning = getattr(context, "planning", None)

    def invalid(reason: str, age_frames: int = -1) -> PlanningAwareV3Preview:
        return PlanningAwareV3Preview(
            valid=False,
            fallback_reason=reason,
            source=getattr(planning, "source", "") if planning is not None else "",
            frame=_safe_int(getattr(planning, "frame", -1)),
            planning_age_frames=age_frames,
            dt=model_dt,
            horizon_s=horizon_s)

    if planning is None:
        return invalid("no_planning")
    if not bool(getattr(planning, "available", False)):
        return invalid("planning_unavailable")
    metadata = dict(getattr(planning, "metadata", {}) or {})
    if _metadata_false(metadata.get("valid_prediction", None)):
        return invalid("metadata_invalid_prediction")

    state = getattr(context, "state", VehicleState())
    state_frame = _optional_finite_float(getattr(state, "frame", None))
    planning_frame = _optional_finite_float(getattr(planning, "frame", None))
    if state_frame is None or planning_frame is None or planning_frame < 0.0:
        return invalid("missing_frame")
    age_frames = int(state_frame) - int(planning_frame)
    if age_frames > int(cfg.max_planning_age_frames):
        return invalid("stale_planning", age_frames=age_frames)

    source_dt = _optional_finite_float(getattr(planning, "horizon_dt", None))
    if (
            source_dt is None or
            source_dt < cfg.min_horizon_dt or
            source_dt > cfg.max_horizon_dt):
        return invalid("bad_horizon_dt", age_frames=age_frames)

    if _planning_has_nonfinite(planning):
        return invalid("nonfinite_preview_value", age_frames=age_frames)
    if _planning_signal_length(planning) <= 0:
        return invalid("no_valid_preview_signal", age_frames=age_frames)

    speed_profile, speed_source, speed_error = _build_speed_profile(
        state,
        planning,
        source_dt,
        times,
        cfg)
    if speed_error:
        return invalid(speed_error, age_frames=age_frames)
    curvature, curvature_source = _build_curvature_profile(
        planning,
        source_dt,
        times,
        cfg)
    brake_profile = _resampled_numeric(
        getattr(planning, "brake", ()),
        source_dt,
        times,
        default=0.0,
        low=0.0,
        high=1.0)
    throttle_profile = _resampled_numeric(
        getattr(planning, "throttle", ()),
        source_dt,
        times,
        default=0.0,
        low=0.0,
        high=1.0)
    predicted_ax, predicted_ax_source = _build_predicted_ax(
        planning,
        source_dt,
        times,
        speed_profile,
        brake_profile,
        throttle_profile,
        cfg)
    predicted_ay, predicted_ay_source = _build_predicted_ay(
        planning,
        source_dt,
        times,
        speed_profile,
        curvature,
        cfg)
    yaw_rate_ref = tuple(
        clamp(
            speed * kappa,
            -cfg.yaw_rate_abs_cap,
            cfg.yaw_rate_abs_cap)
        for speed, kappa in zip(speed_profile, curvature))
    yaw_acc_ref = tuple(
        clamp(value, -cfg.yaw_acc_abs_cap, cfg.yaw_acc_abs_cap)
        for value in _finite_difference(yaw_rate_ref, model_dt))

    arrays = (
        speed_profile,
        curvature,
        predicted_ax,
        predicted_ay,
        yaw_rate_ref,
        yaw_acc_ref,
    )
    if not _all_arrays_finite(arrays):
        return invalid("nonfinite_preview_output", age_frames=age_frames)

    return PlanningAwareV3Preview(
        valid=True,
        fallback_reason="none",
        source=getattr(planning, "source", ""),
        frame=int(planning_frame),
        planning_age_frames=age_frames,
        dt=model_dt,
        horizon_s=times[-1] if times else 0.0,
        times=times,
        speed_profile=speed_profile,
        curvature=curvature,
        predicted_ax=predicted_ax,
        predicted_ay=predicted_ay,
        yaw_rate_ref=yaw_rate_ref,
        yaw_acc_ref=yaw_acc_ref,
        predicted_ax_source=predicted_ax_source,
        predicted_ay_source=predicted_ay_source,
        speed_profile_source=speed_source,
        curvature_source=curvature_source)


def _build_speed_profile(
    state: VehicleState,
    planning: PlanningInfo,
    source_dt: float,
    times: Sequence[float],
    cfg: PlanningAwareV3MpcPrimaryConfig,
) -> Tuple[Tuple[float, ...], str, str]:
    current_speed = clamp(
        _safe_float(getattr(state, "speed", 0.0)),
        0.0,
        cfg.speed_cap_mps)
    target_speed = _numeric_tuple(getattr(planning, "target_speed", ()))
    if any(value < 0.0 for value in target_speed):
        return (), "missing", "negative_target_speed"

    if len(target_speed) == 1:
        target = clamp(target_speed[0], 0.0, cfg.speed_cap_mps)
        ramp_s = max(_safe_float(cfg.target_speed_ramp_s, 1.0), _EPS)
        profile = tuple(
            clamp(
                current_speed + (target - current_speed) *
                _smoothstep01(min(max(time_value / ramp_s, 0.0), 1.0)),
                0.0,
                cfg.speed_cap_mps)
            for time_value in times)
        source = "target_speed_ramp"
    elif len(target_speed) > 1:
        profile = _resampled_values(
            target_speed,
            source_dt,
            times,
            low=0.0,
            high=cfg.speed_cap_mps)
        source = "target_speed_sequence"
    else:
        point_speeds = _numeric_tuple(
            tuple(getattr(point, "speed", 0.0)
                  for point in getattr(planning, "points", ()) or ()))
        if any(value < 0.0 for value in point_speeds):
            return (), "missing", "negative_target_speed"
        if len(point_speeds) == 1:
            target = clamp(point_speeds[0], 0.0, cfg.speed_cap_mps)
            ramp_s = max(_safe_float(cfg.target_speed_ramp_s, 1.0), _EPS)
            profile = tuple(
                clamp(
                    current_speed + (target - current_speed) *
                    _smoothstep01(min(max(time_value / ramp_s, 0.0), 1.0)),
                    0.0,
                    cfg.speed_cap_mps)
                for time_value in times)
            source = "points_speed_ramp"
        elif len(point_speeds) > 1:
            profile = _resampled_values(
                point_speeds,
                source_dt,
                times,
                low=0.0,
                high=cfg.speed_cap_mps)
            source = "points_speed_sequence"
        else:
            profile = tuple(current_speed for _ in times)
            source = "state_speed"

    brake_profile = _resampled_numeric(
        getattr(planning, "brake", ()),
        source_dt,
        times,
        default=0.0,
        low=0.0,
        high=1.0)
    throttle_profile = _resampled_numeric(
        getattr(planning, "throttle", ()),
        source_dt,
        times,
        default=0.0,
        low=0.0,
        high=1.0)
    if any(value > _EPS for value in brake_profile):
        control_speed = _integrate_control_speed(
            current_speed,
            brake_profile,
            throttle_profile,
            cfg,
            times)
        profile = tuple(min(base, control) for base, control in zip(
            profile,
            control_speed))
        source = source + "+control"
    elif source == "state_speed" and any(value > _EPS for value in throttle_profile):
        profile = _integrate_control_speed(
            current_speed,
            brake_profile,
            throttle_profile,
            cfg,
            times)
        source = "state_speed+control"

    return tuple(profile), source, ""


def _build_curvature_profile(
    planning: PlanningInfo,
    source_dt: float,
    times: Sequence[float],
    cfg: PlanningAwareV3MpcPrimaryConfig,
) -> Tuple[Tuple[float, ...], str]:
    curvature = _numeric_tuple(getattr(planning, "curvature", ()))
    if curvature:
        return (
            _resampled_values(
                curvature,
                source_dt,
                times,
                low=-cfg.curvature_abs_cap,
                high=cfg.curvature_abs_cap),
            "planning")

    point_curvature = _numeric_tuple(
        tuple(getattr(point, "curvature", 0.0)
              for point in getattr(planning, "points", ()) or ()))
    if point_curvature:
        return (
            _resampled_values(
                point_curvature,
                source_dt,
                times,
                low=-cfg.curvature_abs_cap,
                high=cfg.curvature_abs_cap),
            "points")

    trajectory_curvature = _trajectory_curvature(
        getattr(planning, "trajectory_xy", ()) or (),
        cfg.min_preview_points_for_trajectory_curvature)
    if trajectory_curvature:
        return (
            _resampled_values(
                trajectory_curvature,
                source_dt,
                times,
                low=-cfg.curvature_abs_cap,
                high=cfg.curvature_abs_cap),
            "trajectory")
    return tuple(0.0 for _ in times), "missing"


def _build_predicted_ax(
    planning: PlanningInfo,
    source_dt: float,
    times: Sequence[float],
    speed_profile: Sequence[float],
    brake_profile: Sequence[float],
    throttle_profile: Sequence[float],
    cfg: PlanningAwareV3MpcPrimaryConfig,
) -> Tuple[Tuple[float, ...], str]:
    direct = _numeric_tuple(getattr(planning, "predicted_ax", ()))
    if direct:
        return (
            _resampled_values(
                direct,
                source_dt,
                times,
                low=-cfg.predicted_accel_abs_cap,
                high=cfg.predicted_accel_abs_cap),
            "planning")

    model_dt = times[1] - times[0] if len(times) > 1 else cfg.model_dt
    finite_diff = tuple(
        clamp(value, -cfg.predicted_accel_abs_cap, cfg.predicted_accel_abs_cap)
        for value in _finite_difference(speed_profile, model_dt))
    control_active = any(
        abs(brake) > _EPS or abs(throttle) > _EPS
        for brake, throttle in zip(brake_profile, throttle_profile))
    if control_active:
        synthetic = []
        for index, (brake, throttle) in enumerate(zip(
                brake_profile,
                throttle_profile)):
            value = (
                throttle * cfg.throttle_accel_scale -
                brake * cfg.brake_decel_scale)
            if finite_diff[index] < -_EPS:
                value = min(value, finite_diff[index])
            elif finite_diff[index] > _EPS and brake <= _EPS:
                value = max(value, finite_diff[index])
            synthetic.append(clamp(
                value,
                -cfg.predicted_accel_abs_cap,
                cfg.predicted_accel_abs_cap))
        return tuple(synthetic), "synthetic_control"

    if max(speed_profile) - min(speed_profile) > 1.0e-6:
        return finite_diff, "speed_finite_difference"

    return tuple(0.0 for _ in times), "zero"


def _build_predicted_ay(
    planning: PlanningInfo,
    source_dt: float,
    times: Sequence[float],
    speed_profile: Sequence[float],
    curvature: Sequence[float],
    cfg: PlanningAwareV3MpcPrimaryConfig,
) -> Tuple[Tuple[float, ...], str]:
    direct = _numeric_tuple(getattr(planning, "predicted_ay", ()))
    if direct:
        return (
            _resampled_values(
                direct,
                source_dt,
                times,
                low=-cfg.predicted_accel_abs_cap,
                high=cfg.predicted_accel_abs_cap),
            "planning")
    return tuple(
        clamp(
            speed * speed * kappa,
            -cfg.predicted_accel_abs_cap,
            cfg.predicted_accel_abs_cap)
        for speed, kappa in zip(speed_profile, curvature)), "v2kappa"


def _integrate_control_speed(
    current_speed: float,
    brake_profile: Sequence[float],
    throttle_profile: Sequence[float],
    cfg: PlanningAwareV3MpcPrimaryConfig,
    times: Sequence[float],
) -> Tuple[float, ...]:
    speeds = []
    speed = current_speed
    previous_time = times[0] if times else 0.0
    for index, time_value in enumerate(times):
        if index > 0:
            dt = max(time_value - previous_time, _EPS)
            accel = (
                throttle_profile[index - 1] * cfg.throttle_accel_scale -
                brake_profile[index - 1] * cfg.brake_decel_scale)
            speed = clamp(speed + accel * dt, 0.0, cfg.speed_cap_mps)
        speeds.append(speed)
        previous_time = time_value
    return tuple(speeds)


def _resampled_numeric(
    values: Sequence[Any],
    source_dt: float,
    times: Sequence[float],
    default: float,
    low: Optional[float] = None,
    high: Optional[float] = None,
) -> Tuple[float, ...]:
    numeric = _numeric_tuple(values)
    if not numeric:
        return tuple(default for _ in times)
    return _resampled_values(numeric, source_dt, times, low=low, high=high)


def _resampled_values(
    values: Sequence[float],
    source_dt: float,
    times: Sequence[float],
    low: Optional[float] = None,
    high: Optional[float] = None,
) -> Tuple[float, ...]:
    raw = tuple(float(value) for value in values)
    if not raw:
        return ()
    if len(raw) == 1:
        value = _bounded(raw[0], low, high)
        return tuple(value for _ in times)
    result = []
    for time_value in times:
        position = max(0.0, time_value / max(source_dt, _EPS))
        if position >= len(raw) - 1:
            value = raw[-1]
        else:
            left = int(math.floor(position))
            right = min(left + 1, len(raw) - 1)
            fraction = position - left
            value = raw[left] + (raw[right] - raw[left]) * fraction
        result.append(_bounded(value, low, high))
    return tuple(result)


def _finite_difference(values: Sequence[float], dt: float) -> Tuple[float, ...]:
    if not values:
        return ()
    if len(values) == 1:
        return (0.0,)
    step = max(dt, _EPS)
    result = []
    for index, value in enumerate(values):
        if index == 0:
            result.append((values[1] - value) / step)
        else:
            result.append((value - values[index - 1]) / step)
    return tuple(result)


def _comfort_result_from_samples(
    samples: Sequence[Tuple[float, float, float, float]],
    cfg: PlanningAwareV3MpcPrimaryConfig,
) -> ComfortGuardResult:
    if not samples:
        return ComfortGuardResult()

    ax_values = tuple(sample[1] for sample in samples)
    ay_values = tuple(sample[2] for sample in samples)
    yaw_values = tuple(sample[3] for sample in samples)
    lon_jerk = []
    mag_jerk = []
    yaw_acc = []
    for previous, current in zip(samples[:-1], samples[1:]):
        dt = max(current[0] - previous[0], _EPS)
        jx = (current[1] - previous[1]) / dt
        jy = (current[2] - previous[2]) / dt
        lon_jerk.append(jx)
        mag_jerk.append(math.hypot(jx, jy))
        yaw_acc.append((current[3] - previous[3]) / dt)

    lon_margin = min(
        cfg.comfort_lon_acc_upper - max(ax_values),
        min(ax_values) - cfg.comfort_lon_acc_lower)
    lat_margin = cfg.comfort_abs_lat_acc - _max_abs(ay_values)
    lon_jerk_margin = cfg.comfort_abs_lon_jerk - _max_abs(lon_jerk)
    mag_jerk_margin = cfg.comfort_abs_mag_jerk - _max_abs(mag_jerk)
    yaw_acc_margin = cfg.comfort_abs_yaw_acc - _max_abs(yaw_acc)
    yaw_rate_margin = cfg.comfort_abs_yaw_rate - _max_abs(yaw_values)
    margins = (
        lon_margin,
        lat_margin,
        lon_jerk_margin,
        mag_jerk_margin,
        yaw_acc_margin,
        yaw_rate_margin,
    )
    activation = max(_safe_float(cfg.comfort_margin_activation, 0.10), 0.0)
    smallest_margin = min(margins)
    active = smallest_margin <= activation
    severity = 0.0
    if active:
        severity = (activation - smallest_margin) / max(activation, 1.0)
        severity = clamp(severity, 0.0, 10.0)
    return ComfortGuardResult(
        lon_acc_margin=_safe_float(lon_margin),
        lat_acc_margin=_safe_float(lat_margin),
        lon_jerk_margin=_safe_float(lon_jerk_margin),
        mag_jerk_margin=_safe_float(mag_jerk_margin),
        yaw_acc_margin=_safe_float(yaw_acc_margin),
        yaw_rate_margin=_safe_float(yaw_rate_margin),
        active=active,
        slew_penalty_multiplier=1.0 + cfg.comfort_slew_penalty_gain * severity,
        mean_damper_penalty_multiplier=(
            1.0 + cfg.comfort_mean_damper_penalty_gain * severity))


def _planning_signal_length(planning: PlanningInfo) -> int:
    lengths = [
        len(getattr(planning, "trajectory_xy", ()) or ()),
        len(getattr(planning, "target_speed", ()) or ()),
        len(getattr(planning, "curvature", ()) or ()),
        len(getattr(planning, "steer", ()) or ()),
        len(getattr(planning, "throttle", ()) or ()),
        len(getattr(planning, "brake", ()) or ()),
        len(getattr(planning, "predicted_ax", ()) or ()),
        len(getattr(planning, "predicted_ay", ()) or ()),
        len(getattr(planning, "points", ()) or ()),
    ]
    return max(lengths or [0])


def _planning_has_nonfinite(planning: PlanningInfo) -> bool:
    for name in (
            "target_speed",
            "curvature",
            "steer",
            "throttle",
            "brake",
            "predicted_ax",
            "predicted_ay",
            "trajectory_yaw"):
        for value in getattr(planning, name, ()) or ():
            if _optional_finite_float(value) is None:
                return True
    for xy in getattr(planning, "trajectory_xy", ()) or ():
        try:
            x_value, y_value = xy[0], xy[1]
        except (TypeError, IndexError):
            return True
        if (
                _optional_finite_float(x_value) is None or
                _optional_finite_float(y_value) is None):
            return True
    for point in getattr(planning, "points", ()) or ():
        for name in ("time_seconds", "x", "y", "z", "speed", "yaw", "curvature"):
            if _optional_finite_float(getattr(point, name, 0.0)) is None:
                return True
    return False


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
    denom = math.hypot(ax, ay) * math.hypot(bx, by) * math.hypot(cx, cy)
    if denom <= _EPS:
        return 0.0
    return 2.0 * twice_area / denom


def _numeric_tuple(values: Sequence[Any]) -> Tuple[float, ...]:
    return tuple(float(value) for value in (values or ()))


def _context_dt(context: ControllerContext, default: float) -> float:
    value = _optional_finite_float(getattr(context, "dt", None))
    if value is None or value <= 0.0:
        value = _optional_finite_float(getattr(context.state, "dt", None))
    if value is None or value <= 0.0:
        value = _safe_float(default, 0.05)
    return max(value, _EPS)


def _positive_or_default(value: Any, default: float) -> float:
    result = _optional_finite_float(value)
    if result is None or result <= 0.0:
        result = _optional_finite_float(default)
    if result is None or result <= 0.0:
        result = 0.05
    return result


def _bounded(
    value: float,
    low: Optional[float],
    high: Optional[float],
) -> float:
    result = value
    if low is not None:
        result = max(float(low), result)
    if high is not None:
        result = min(float(high), result)
    return result


def _low_pass(
    previous: Optional[float],
    current: float,
    alpha: float,
) -> float:
    if previous is None:
        return current
    return previous + alpha * (current - previous)


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


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _match_length(
    values: Sequence[Any],
    length: int,
    default: float,
) -> Tuple[float, ...]:
    result = tuple(_safe_float(value, default) for value in (values or ()))
    if len(result) >= length:
        return result[:length]
    return result + tuple(default for _ in range(length - len(result)))


def _all_finite(values: Sequence[float]) -> bool:
    return all(_optional_finite_float(value) is not None for value in values)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(_safe_float(value) for value in values) / float(len(values))


def _hinge(value: float) -> float:
    return max(0.0, _safe_float(value, 0.0))


def _sign(value: float) -> float:
    result = _safe_float(value, 0.0)
    if result > _EPS:
        return 1.0
    if result < -_EPS:
        return -1.0
    return 0.0


def _smoothstep01(value: float) -> float:
    value = clamp(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _max_abs(values: Sequence[float]) -> float:
    return max((abs(float(value)) for value in values), default=0.0)


def _signed_peak(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return max((float(value) for value in values), key=lambda value: abs(value))


def _all_arrays_finite(arrays: Sequence[Sequence[float]]) -> bool:
    for values in arrays:
        for value in values:
            if _optional_finite_float(value) is None:
                return False
    return True
