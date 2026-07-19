"""Planning-aware v5 applied-residual QP-MPC over Phase A.

The controller is intentionally manifest-gated: without a Step 04
``validated_live`` artifact it computes diagnostics and returns the same-tick
Phase A command. This keeps Step 05 wiring testable without inventing model
modes or signals.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import sys
import time
import zipfile
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

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
from .skyhook import canonical_skyhook_v3_projection
from .residual_mpc_contracts import (
    MODAL_BASIS_NAMES,
    modal_to_wheel_residual,
    wheel_residual_to_modal,
)


PLANNING_AWARE_V5_RESIDUAL_QP_MPC_VERSION = (
    "planning_aware_v5_residual_qp_mpc_step05_v1")
PLANNING_AWARE_V5_RESIDUAL_QP_MPC_SCHEMA_VERSION = (
    "residual_mpc_qp_controller_step05_v1")
RESIDUAL_MPC_PREPROCESSING_SCHEMA_VERSION = "residual_mpc_preprocessing_v1"


RESIDUAL_MPC_LEGACY_TIMING_BUDGET_METRIC = "residual_mpc_compute_ms"
RESIDUAL_MPC_END_TO_END_TIMING_BUDGET_METRIC = (
    "residual_mpc_controller_end_to_end_ms")
RESIDUAL_MPC_INCREMENTAL_TIMING_BUDGET_METRIC = (
    "residual_mpc_incremental_compute_ms")
RESIDUAL_MPC_TIMING_BUDGET_METRICS = (
    RESIDUAL_MPC_LEGACY_TIMING_BUDGET_METRIC,
    RESIDUAL_MPC_END_TO_END_TIMING_BUDGET_METRIC,
    RESIDUAL_MPC_INCREMENTAL_TIMING_BUDGET_METRIC,
)


RESIDUAL_QP_MPC_FALLBACK_REASONS = (
    "model_unavailable",
    "model_hash_mismatch",
    "history_warmup",
    "preview_invalid",
    "preview_stale",
    "telemetry_invalid",
    "out_of_support",
    "solver_unavailable",
    "solver_error",
    "solver_infeasible",
    "solver_timeout",
    "nonfinite_solution",
    "first_action_mismatch",
    "compute_budget_exceeded",
)


RESIDUAL_QP_MPC_DIAGNOSTIC_FIELDS: Tuple[str, ...] = (
    "residual_qp_mpc_shadow_config_path",
    "residual_qp_mpc_shadow_config_exists",
    "residual_mpc_schema_version",
    "residual_mpc_controller_version",
    "residual_mpc_enabled",
    "residual_mpc_shadow_mode",
    "residual_mpc_mode",
    "residual_mpc_fallback_active",
    "residual_mpc_fallback_reason",
    "residual_mpc_model_artifact_dir",
    "residual_mpc_model_artifact_dir_exists",
    "residual_mpc_model_artifact_loaded",
    "residual_mpc_model_id",
    "residual_mpc_model_family",
    "residual_mpc_model_validation_status",
    "residual_mpc_model_hash_status",
    "residual_mpc_active_modes",
    "residual_mpc_output_channels",
    "residual_mpc_exogenous_channels",
    "residual_mpc_input_delay_ticks",
    "residual_mpc_output_lags",
    "residual_mpc_input_lags",
    "residual_mpc_horizon_steps",
    "residual_mpc_control_block_steps",
    "residual_mpc_control_blocks",
    "residual_mpc_history_ready",
    "residual_mpc_history_samples",
    "residual_mpc_support_distance",
    "residual_mpc_support_scale",
    "residual_mpc_operating_mode",
    "residual_mpc_weight_calm",
    "residual_mpc_weight_lateral",
    "residual_mpc_weight_braking",
    "residual_mpc_solver_backend",
    "residual_mpc_solver_name",
    "residual_mpc_solver_status",
    "residual_mpc_solver_iterations",
    "residual_mpc_solver_setup_ms",
    "residual_mpc_solver_solve_ms",
    "residual_mpc_solver_total_ms",
    "residual_mpc_solver_primal_residual",
    "residual_mpc_solver_dual_residual",
    "residual_mpc_solver_primal_residual_available",
    "residual_mpc_solver_dual_residual_available",
    "residual_mpc_solver_primal_residual_provenance",
    "residual_mpc_solver_dual_residual_provenance",
    # Legacy aggregate boolean.  It now means whether the initial QP was
    # given a warm-start vector, rather than reflecting controller state after
    # a solve.  The immutable per-stage fields below carry the complete truth.
    "residual_mpc_solver_warm_start_used",
    "residual_mpc_initial_qp_warm_start_used",
    "residual_mpc_initial_qp_warm_start_dimension",
    "residual_mpc_initial_qp_warm_start_provenance",
    "residual_mpc_first_action_resolve_warm_start_used",
    "residual_mpc_first_action_resolve_warm_start_dimension",
    "residual_mpc_first_action_resolve_warm_start_provenance",
    "residual_mpc_qp_objective",
    "residual_mpc_re_solve_count",
    "residual_mpc_re_solve_attempted",
    "residual_mpc_re_solve_succeeded",
    "residual_mpc_re_solve_failed",
    "residual_mpc_first_action_correction_active",
    "residual_mpc_raw_model_linf",
    "residual_mpc_raw_model_nonzero",
    "residual_mpc_qp_solution_linf",
    "residual_mpc_qp_solution_nonzero",
    "residual_mpc_budget_accepted_linf",
    "residual_mpc_budget_accepted_nonzero",
    "residual_mpc_final_applied_linf",
    "residual_mpc_final_applied_nonzero",
    "residual_mpc_qp_solution_finite",
    "residual_mpc_would_propose_without_timing_gate",
    "residual_mpc_timing_gate_pass",
    "residual_mpc_safety_projection_pass",
    "residual_mpc_osqp_status",
    "residual_mpc_osqp_status_val",
    "residual_mpc_osqp_iter",
    "residual_mpc_osqp_prim_res",
    "residual_mpc_osqp_dual_res",
    "residual_mpc_osqp_prim_res_provenance",
    "residual_mpc_osqp_dual_res_provenance",
    "residual_mpc_osqp_obj_val",
    "residual_mpc_osqp_rho_updates",
    "residual_mpc_osqp_setup_time_ms",
    "residual_mpc_osqp_solve_time_ms",
    "residual_mpc_osqp_update_time_ms",
    "residual_mpc_initial_qp_status",
    "residual_mpc_first_action_resolve_status",
    "residual_mpc_controller_solver_outcome",
    "residual_mpc_zero_feasible",
    "residual_mpc_zero_feas_max_violation",
    "residual_mpc_zero_feas_worst_group",
    "residual_mpc_zero_feas_worst_lower",
    "residual_mpc_zero_feas_worst_value",
    "residual_mpc_zero_feas_worst_upper",
    "residual_mpc_first_action_resolve_zero_feasible",
    "residual_mpc_first_action_resolve_zero_feas_max_violation",
    "residual_mpc_first_action_resolve_zero_feas_worst_group",
    "residual_mpc_first_action_resolve_zero_feas_worst_lower",
    "residual_mpc_first_action_resolve_zero_feas_worst_value",
    "residual_mpc_first_action_resolve_zero_feas_worst_upper",
    "residual_mpc_compute_ms",
    "residual_mpc_controller_end_to_end_ms",
    "residual_mpc_incremental_compute_ms",
    "residual_mpc_budget_metric_name",
    "residual_mpc_budget_metric_ms",
    "residual_mpc_budget_ms",
    "residual_mpc_stage_phase_a_shadow_ms",
    "residual_mpc_stage_phase_a_commit_ms",
    "residual_mpc_stage_feature_build_ms",
    "residual_mpc_stage_model_predict_ms",
    "residual_mpc_stage_qp_matrix_build_ms",
    "residual_mpc_stage_qp_setup_or_update_ms",
    "residual_mpc_stage_qp_solve_wall_ms",
    "residual_mpc_stage_qp_postprocess_ms",
    "residual_mpc_stage_safety_projection_ms",
    "residual_mpc_compute_budget_exceeded",
    "residual_mpc_baseline_predicted_comfort_cost",
    "residual_mpc_optimized_predicted_comfort_cost",
    "residual_mpc_predicted_cost_gain_abs",
    "residual_mpc_predicted_cost_gain_rel",
    "residual_mpc_baseline_predicted_body_cost",
    "residual_mpc_optimized_predicted_body_cost",
    "residual_mpc_baseline_predicted_safety_cost",
    "residual_mpc_optimized_predicted_safety_cost",
    "residual_mpc_max_slack",
    "residual_mpc_first_action_qp_to_applied_l2",
    "residual_mpc_first_action_qp_to_applied_linf",
    "residual_mpc_first_action_pre_correction_l2",
    "residual_mpc_first_action_pre_correction_linf",
    "residual_mpc_first_action_raw_to_projected_l2",
    "residual_mpc_first_action_raw_to_projected_linf",
    "residual_mpc_first_action_projected_to_final_l2",
    "residual_mpc_first_action_projected_to_final_linf",
    "residual_mpc_first_action_final_to_reprojected_l2",
    "residual_mpc_first_action_final_to_reprojected_linf",
    "residual_mpc_first_action_fixed_point_linf",
    "residual_mpc_first_action_fixed_point_iterations",
    "residual_mpc_first_action_minmax_clip_count",
    "residual_mpc_first_action_rate_clip_count",
    "residual_mpc_first_action_low_rel_blend_count",
    "residual_mpc_first_action_infeasible_blend_count",
    "residual_mpc_first_action_modal_coupling_bound_miss_count",
    "residual_mpc_first_action_fixed_point_out_of_support",
    "residual_mpc_first_action_support_comparison_evaluable",
    "residual_mpc_first_action_support_comparison_error",
    "residual_mpc_first_action_active_mode_order",
    "residual_mpc_first_action_fixed_point_active_target",
    "residual_mpc_first_action_existing_lower",
    "residual_mpc_first_action_existing_upper",
    "residual_mpc_first_action_lower_signed_margins",
    "residual_mpc_first_action_upper_signed_margins",
    "residual_mpc_first_action_support_violation_count",
    "residual_mpc_first_action_support_violation_linf",
    "residual_mpc_first_action_support_worst_mode",
    "residual_mpc_first_action_support_worst_index",
    "residual_mpc_first_action_support_worst_side",
    "residual_mpc_first_action_support_worst_target",
    "residual_mpc_first_action_support_worst_lower",
    "residual_mpc_first_action_support_worst_upper",
    "residual_mpc_first_action_support_worst_signed_gap",
    "residual_mpc_first_action_support_worst_absolute_gap",
    "residual_mpc_first_action_support_worst_normalized_gap",
    "residual_mpc_first_action_support_binding_constraint_group",
    "residual_mpc_first_action_support_conversion_provenance",
    "residual_mpc_first_action_support_comparison_tolerance",
    "residual_mpc_first_action_corrective_qp_constructed",
    "residual_mpc_first_action_corrective_qp_rejected",
    "residual_mpc_debug_qp_snapshot_key",
    "residual_mpc_debug_qp_snapshot_candidate_ordinal",
    "residual_mpc_bounds_guard_active",
    "residual_mpc_projection_active",
    "residual_mpc_final_guard_active",
    "residual_mpc_phase_a_commit_match",
    "residual_mpc_measured_dt_s",
    "residual_mpc_planning_age_frames",
    "residual_mpc_planning_age_s",
    "residual_mpc_preview_validity_reason",
    "residual_mpc_strict_step07_shadow_validation",
    "residual_mpc_spring_scale_invariant",
) + tuple(
    "%s_%s" % (prefix, mode)
    for prefix in (
        "residual_mpc_phase_a_modal",
        "residual_mpc_requested_residual_modal",
        "residual_mpc_final_residual_modal",
    )
    for mode in MODAL_BASIS_NAMES
) + tuple(
    "%s_%s" % (prefix, label)
    for prefix in (
        "residual_mpc_phase_a_shadow_damper",
        "residual_mpc_raw_qp_first_damper",
        "residual_mpc_qp_first_damper",
        "residual_mpc_semi_active_first_damper",
        "residual_mpc_safety_first_damper",
        "residual_mpc_reprojected_first_damper",
        "residual_mpc_fixed_point_first_damper",
        "residual_mpc_requested_residual",
        "residual_mpc_projected_residual",
        "residual_mpc_final_residual",
        "residual_mpc_final_damper",
    )
    for label in _WHEEL_LABELS
)


@dataclass(frozen=True)
class QpConstraintGroup:
    """Provenance for a set of box-bound constraints."""

    name: str
    kind: str
    indices: Tuple[int, ...]
    lower: Tuple[float, ...]
    upper: Tuple[float, ...]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class DenseQpProblem:
    """Dense convex QP with box constraints.

    Minimize ``0.5 x'Hx + f'x`` subject to ``lower <= x <= upper``.
    """

    hessian: Tuple[Tuple[float, ...], ...]
    linear: Tuple[float, ...]
    lower: Tuple[float, ...]
    upper: Tuple[float, ...]
    metadata: Mapping[str, Any]
    constraint_groups: Tuple[QpConstraintGroup, ...] = ()


@dataclass(frozen=True)
class SolverResult:
    status: str
    primal_solution: Tuple[float, ...] = ()
    objective: float = 0.0
    iterations: int = 0
    setup_ms: float = 0.0
    solve_ms: float = 0.0
    total_ms: float = 0.0
    primal_residual: float = 0.0
    dual_residual: float = 0.0
    primal_residual_available: bool = False
    dual_residual_available: bool = False
    primal_residual_provenance: str = "unavailable"
    dual_residual_provenance: str = "unavailable"
    error: str = ""
    osqp_status: str = ""
    osqp_status_val: Optional[int] = None
    osqp_iter: Optional[int] = None
    osqp_prim_res: Optional[float] = None
    osqp_dual_res: Optional[float] = None
    osqp_obj_val: Optional[float] = None
    osqp_rho_updates: Optional[int] = None
    osqp_setup_time_ms: Optional[float] = None
    osqp_solve_time_ms: Optional[float] = None
    osqp_update_time_ms: Optional[float] = None


@dataclass(frozen=True)
class ZeroFeasibilityResult:
    feasible: bool
    max_violation: float
    worst_group: str
    worst_lower: float
    worst_value: float
    worst_upper: float


@dataclass(frozen=True)
class ResidualLifecycleTrace:
    raw_model_residual: Tuple[float, ...]
    qp_solution_residual: Tuple[float, ...]
    budget_accepted_residual: Tuple[float, ...]
    final_applied_residual: Tuple[float, ...]
    qp_solution_finite: bool
    would_propose_without_timing_gate: bool
    timing_gate_pass: bool
    safety_projection_pass: bool
    initial_solver_result: SolverResult
    first_action_resolve_status: str
    controller_solver_outcome: str
    initial_zero_feasibility: ZeroFeasibilityResult
    resolve_zero_feasibility: ZeroFeasibilityResult


@dataclass(frozen=True)
class DebugQpStageRecord:
    stage: str
    solve_ordinal: int
    problem: DenseQpProblem
    zero_feasibility: ZeroFeasibilityResult
    solver_result: SolverResult
    warm_start_input: Tuple[float, ...]
    warm_start_provenance: str


@dataclass(frozen=True)
class DebugQpSnapshotRecord:
    key: str
    frame: int
    step: int
    event_window: bool
    controller_solver_outcome: str
    solver_name: str
    solver_max_iterations: int
    solver_tolerance: float
    solver_setup_arguments: Mapping[str, Any]
    runtime_provenance: Mapping[str, Any]
    stages: Tuple[DebugQpStageRecord, ...]
    first_action_support_comparison: Optional[
        "FixedPointSupportComparison"] = None


@dataclass(frozen=True)
class SolverPreflightResult:
    ok: bool
    solver_name: str
    status: str
    package_versions: Mapping[str, str]
    package_paths: Mapping[str, str]
    solution: Tuple[float, ...] = ()
    objective: float = 0.0
    setup_ms: float = 0.0
    solve_ms: float = 0.0
    total_ms: float = 0.0
    iterations: int = 0
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "solver_name": self.solver_name,
            "status": self.status,
            "package_versions": dict(self.package_versions),
            "package_paths": dict(self.package_paths),
            "solution": list(self.solution),
            "objective": self.objective,
            "setup_ms": self.setup_ms,
            "solve_ms": self.solve_ms,
            "total_ms": self.total_ms,
            "iterations": self.iterations,
            "error": self.error,
        }


@dataclass(frozen=True)
class FixedPointSupportViolation:
    active_index: int
    mode: str
    side: str
    target: float
    lower: float
    upper: float
    signed_gap: float
    absolute_gap: float
    normalized_gap: float
    normalization_span: float
    binding_constraint_group: str
    binding_constraint_group_candidates: Tuple[str, ...]


@dataclass(frozen=True)
class FixedPointSupportComparison:
    evaluable: bool
    error: str
    active_mode_order: Tuple[str, ...]
    wheel_residual: Tuple[float, ...]
    full_modal_target: Tuple[float, ...]
    active_target: Tuple[float, ...]
    existing_lower: Tuple[float, ...]
    existing_upper: Tuple[float, ...]
    lower_signed_margins: Tuple[float, ...]
    upper_signed_margins: Tuple[float, ...]
    violations: Tuple[FixedPointSupportViolation, ...]
    violation_count: int
    violation_linf: float
    worst_mode: str
    worst_index: int
    worst_side: str
    worst_target: float
    worst_lower: float
    worst_upper: float
    worst_signed_gap: float
    worst_absolute_gap: float
    worst_normalized_gap: float
    binding_constraint_group: str
    conversion_provenance: str
    comparison_tolerance: float
    corrective_qp_constructed: bool
    corrective_qp_rejected: bool


@dataclass(frozen=True)
class FirstActionProjectionTrace:
    raw_dampers: Tuple[float, ...]
    semi_active_dampers: Tuple[float, ...]
    safety_dampers: Tuple[float, ...]
    reprojected_dampers: Tuple[float, ...]
    fixed_point_dampers: Tuple[float, ...]
    pre_correction_l2: float
    pre_correction_linf: float
    raw_to_projected_l2: float
    raw_to_projected_linf: float
    projected_to_final_l2: float
    projected_to_final_linf: float
    final_to_reprojected_l2: float
    final_to_reprojected_linf: float
    fixed_point_linf: float
    fixed_point_iterations: int
    minmax_clip_count: int
    rate_clip_count: int
    low_rel_blend_count: int
    infeasible_blend_count: int
    modal_coupling_bound_miss_count: int
    fixed_point_out_of_support: int = 0
    fixed_point_support_comparison: Optional[
        FixedPointSupportComparison] = None


class ResidualQpSolverAdapter:
    """Common solver adapter contract used by the online controller."""

    name = "unavailable"

    def solve(
        self,
        problem_data: DenseQpProblem,
        warm_start: Optional[Sequence[float]] = None,
    ) -> SolverResult:
        raise NotImplementedError


class UnavailableSolverAdapter(ResidualQpSolverAdapter):
    name = "unavailable"

    def __init__(self, reason: str):
        self.reason = reason

    def solve(
        self,
        problem_data: DenseQpProblem,
        warm_start: Optional[Sequence[float]] = None,
    ) -> SolverResult:
        return SolverResult(status="solver_unavailable", error=self.reason)


class ProjectedGradientQpSolverAdapter(ResidualQpSolverAdapter):
    """Dependency-free deterministic QP adapter for CARLA-free validation."""

    name = "box_projected_gradient"

    def __init__(
        self,
        *,
        max_iterations: int = 80,
        tolerance: float = 1.0e-8,
    ):
        self.max_iterations = max(1, int(max_iterations))
        self.tolerance = max(float(tolerance), 1.0e-12)

    def solve(
        self,
        problem_data: DenseQpProblem,
        warm_start: Optional[Sequence[float]] = None,
    ) -> SolverResult:
        start_ns = time.perf_counter_ns()
        lower = tuple(float(value) for value in problem_data.lower)
        upper = tuple(float(value) for value in problem_data.upper)
        dim = len(problem_data.linear)
        if dim == 0:
            return SolverResult(status="solved", total_ms=0.0)
        if len(lower) != dim or len(upper) != dim:
            return SolverResult(status="solver_error", error="bound_dim")
        for lo, hi in zip(lower, upper):
            if lo > hi or not math.isfinite(lo) or not math.isfinite(hi):
                return SolverResult(status="solver_infeasible", error="bad_bounds")
        hessian = tuple(tuple(float(value) for value in row)
                        for row in problem_data.hessian)
        if len(hessian) != dim or any(len(row) != dim for row in hessian):
            return SolverResult(status="solver_error", error="hessian_dim")
        linear = tuple(float(value) for value in problem_data.linear)
        if warm_start and len(warm_start) == dim:
            x = [
                clamp(_safe_float(warm_start[index], 0.0), lower[index], upper[index])
                for index in range(dim)
            ]
        else:
            x = [clamp(0.0, lower[index], upper[index]) for index in range(dim)]
        lipschitz = max(
            sum(abs(hessian[row][col]) for col in range(dim))
            for row in range(dim))
        step = 1.0 / max(lipschitz, 1.0e-9)
        iterations = 0
        residual = 0.0
        for iterations in range(1, self.max_iterations + 1):
            grad = [
                linear[row] +
                sum(hessian[row][col] * x[col] for col in range(dim))
                for row in range(dim)
            ]
            candidate = [
                clamp(x[index] - step * grad[index], lower[index], upper[index])
                for index in range(dim)
            ]
            residual = math.sqrt(sum(
                (candidate[index] - x[index]) ** 2
                for index in range(dim)))
            x = candidate
            if residual <= self.tolerance:
                break
        total_ms = max(0.0, (time.perf_counter_ns() - start_ns) / 1.0e6)
        objective = _qp_objective(hessian, linear, x)
        return SolverResult(
            status="solved",
            primal_solution=tuple(x),
            objective=objective,
            iterations=iterations,
            solve_ms=total_ms,
            total_ms=total_ms,
            primal_residual=residual,
            dual_residual=0.0,
            primal_residual_available=True,
            dual_residual_available=False,
            primal_residual_provenance=(
                "box_projected_gradient_iterate_delta_l2"),
            dual_residual_provenance=(
                "unavailable_box_projected_gradient"))


class OsqpSolverAdapter(ResidualQpSolverAdapter):
    """Thin OSQP adapter.

    The repository does not vendor OSQP. If OSQP/scipy are absent, the adapter
    reports ``solver_unavailable`` and the controller falls back to Phase A.
    """

    name = "osqp"

    def __init__(
        self,
        *,
        max_iterations: int = 80,
        tolerance: float = 1.0e-8,
        numpy_module: Any = None,
        osqp_module: Any = None,
        sparse_module: Any = None,
        import_module: Callable[[str], Any] = importlib.import_module,
    ):
        self.max_iterations = max(1, int(max_iterations))
        self.tolerance = max(float(tolerance), 1.0e-12)
        if (
                numpy_module is not None and
                osqp_module is not None and
                sparse_module is not None):
            self._numpy = numpy_module
            self._osqp = osqp_module
            self._sparse = sparse_module
            self._error = ""
            return
        try:
            numpy = import_module("numpy")
            osqp = import_module("osqp")
            sparse = import_module("scipy.sparse")
        except Exception as exc:  # noqa: BLE001 - availability probe.
            self._numpy = None
            self._osqp = None
            self._sparse = None
            self._error = "%s:%s" % (exc.__class__.__name__, exc)
        else:
            self._numpy = numpy
            self._osqp = osqp
            self._sparse = sparse
            self._error = ""

    def solve(
        self,
        problem_data: DenseQpProblem,
        warm_start: Optional[Sequence[float]] = None,
    ) -> SolverResult:
        if self._numpy is None or self._osqp is None or self._sparse is None:
            return SolverResult(status="solver_unavailable", error=self._error)
        start_ns = time.perf_counter_ns()
        try:
            hessian = self._sparse.csc_matrix(
                [list(row) for row in problem_data.hessian])
            identity = self._sparse.eye(len(problem_data.linear), format="csc")
            solver = self._osqp.OSQP()
            setup_start = time.perf_counter_ns()
            solver.setup(
                P=hessian,
                q=self._numpy.asarray(problem_data.linear, dtype=float),
                A=identity,
                l=self._numpy.asarray(problem_data.lower, dtype=float),
                u=self._numpy.asarray(problem_data.upper, dtype=float),
                verbose=False,
                polish=True,
                warm_start=True,
                max_iter=self.max_iterations,
                eps_abs=self.tolerance,
                eps_rel=self.tolerance)
            setup_ms = max(
                0.0,
                (time.perf_counter_ns() - setup_start) / 1.0e6)
            if warm_start:
                solver.warm_start(
                    x=self._numpy.asarray(tuple(warm_start), dtype=float))
            solve_start = time.perf_counter_ns()
            result = solver.solve()
            solve_ms = max(0.0, (time.perf_counter_ns() - solve_start) / 1.0e6)
        except Exception as exc:  # noqa: BLE001 - runtime solver failure.
            total_ms = max(0.0, (time.perf_counter_ns() - start_ns) / 1.0e6)
            return SolverResult(
                status="solver_error",
                total_ms=total_ms,
                error="%s:%s" % (exc.__class__.__name__, exc))
        total_ms = max(0.0, (time.perf_counter_ns() - start_ns) / 1.0e6)
        info = getattr(result, "info", None)
        status = str(getattr(info, "status", ""))
        mapped_status = "solved" if status.lower().startswith("solved") else status
        raw_solution = getattr(result, "x", None)
        solution = tuple(float(value) for value in (
            raw_solution if raw_solution is not None else ()))
        primal_residual, primal_residual_name = _optional_info_first_float(
            info, ("prim_res", "pri_res"))
        dual_residual, dual_residual_name = _optional_info_first_float(
            info, ("dual_res", "dua_res"))
        return SolverResult(
            status=mapped_status,
            primal_solution=solution,
            objective=_safe_float(getattr(info, "obj_val", 0.0), 0.0),
            iterations=int(_safe_float(getattr(info, "iter", 0), 0)),
            setup_ms=setup_ms,
            solve_ms=solve_ms,
            total_ms=total_ms,
            # Keep the historic numeric fields stable at 0.0 when native
            # fields are absent, but make absence explicit below.  A consumer
            # must never treat that compatibility value as measured residual
            # evidence.
            primal_residual=(
                primal_residual if primal_residual is not None else 0.0),
            dual_residual=(
                dual_residual if dual_residual is not None else 0.0),
            primal_residual_available=primal_residual is not None,
            dual_residual_available=dual_residual is not None,
            primal_residual_provenance=(
                "osqp_info_%s" % primal_residual_name
                if primal_residual_name else
                "unavailable_osqp_info_prim_res_or_pri_res"),
            dual_residual_provenance=(
                "osqp_info_%s" % dual_residual_name
                if dual_residual_name else
                "unavailable_osqp_info_dual_res_or_dua_res"),
            osqp_status=status,
            osqp_status_val=_optional_info_int(info, "status_val"),
            osqp_iter=_optional_info_int(info, "iter"),
            osqp_prim_res=primal_residual,
            osqp_dual_res=dual_residual,
            osqp_obj_val=_optional_info_float(info, "obj_val"),
            osqp_rho_updates=_optional_info_int(info, "rho_updates"),
            osqp_setup_time_ms=_optional_info_ms(info, "setup_time"),
            osqp_solve_time_ms=_optional_info_ms(info, "solve_time"),
            osqp_update_time_ms=_optional_info_ms(info, "update_time"))


@dataclass(frozen=True)
class ArxOrders:
    output_lags: int
    input_lags: int


@dataclass(frozen=True)
class OnlineArxModel:
    model_family: str
    output_channels: Tuple[str, ...]
    input_modes: Tuple[str, ...]
    phase_a_modes: Tuple[str, ...]
    exogenous_channels: Tuple[str, ...]
    orders: ArxOrders
    input_delay_ticks: int
    coefficients: Tuple[Tuple[float, ...], ...]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "OnlineArxModel":
        orders = dict(payload.get("orders") or {})
        return cls(
            model_family=str(payload["model_family"]),
            output_channels=tuple(str(item) for item in payload["output_channels"]),
            input_modes=tuple(str(item) for item in payload["input_modes"]),
            phase_a_modes=tuple(
                str(item) for item in payload.get("phase_a_modes", MODAL_BASIS_NAMES)),
            exogenous_channels=tuple(
                str(item) for item in payload.get("exogenous_channels", ())),
            orders=ArxOrders(
                output_lags=int(orders.get("output_lags", 1)),
                input_lags=int(orders.get("input_lags", 1))),
            input_delay_ticks=int(payload.get("input_delay_ticks", 0)),
            coefficients=tuple(
                tuple(float(value) for value in row)
                for row in payload["coefficients"]))

    def predict_next(
        self,
        y_history: Sequence[Sequence[float]],
        u_history: Sequence[Sequence[float]],
        phase_history: Sequence[Sequence[float]],
        q_current: Sequence[float],
    ) -> Tuple[float, ...]:
        features = [1.0]
        for lag in range(self.orders.output_lags):
            features.extend(float(value) for value in y_history[-1 - lag])
        for lag in range(self.orders.input_lags):
            features.extend(float(value) for value in
                            u_history[-1 - self.input_delay_ticks - lag])
        for lag in range(self.orders.input_lags):
            features.extend(float(value) for value in
                            phase_history[-1 - self.input_delay_ticks - lag])
        features.extend(float(value) for value in q_current)
        if len(features) != len(self.coefficients):
            raise ValueError(
                "feature dimension mismatch: expected %d got %d" %
                (len(self.coefficients), len(features)))
        output_dim = len(self.output_channels)
        return tuple(
            sum(features[row] * self.coefficients[row][col]
                for row in range(len(features)))
            for col in range(output_dim))


@dataclass(frozen=True)
class ResidualMpcArtifact:
    root: Path
    manifest: Mapping[str, Any]
    preprocessing: Mapping[str, Any]
    model: OnlineArxModel

    @property
    def model_id(self) -> str:
        return str(self.manifest.get("model_id", ""))

    @property
    def validation_status(self) -> str:
        return str(self.manifest.get("validation_status", ""))

    @property
    def active_modes(self) -> Tuple[str, ...]:
        return tuple(str(item) for item in self.manifest.get("active_modes", ()))

    @classmethod
    def load(
        cls,
        artifact_dir: str,
        *,
        require_validated_live: bool = True,
        allow_dry_model: bool = False,
    ) -> "ResidualMpcArtifact":
        if not artifact_dir:
            raise FileNotFoundError("empty model_artifact_dir")
        root = Path(os.path.expanduser(artifact_dir))
        manifest_path = root / "manifest.json"
        preprocessing_path = root / "preprocessing.json"
        model_path = root / "model.npz"
        if not manifest_path.is_file():
            raise FileNotFoundError(str(manifest_path))
        if not preprocessing_path.is_file():
            raise FileNotFoundError(str(preprocessing_path))
        if not model_path.is_file():
            raise FileNotFoundError(str(model_path))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validation_status = str(manifest.get("validation_status", ""))
        if (
                require_validated_live and
                validation_status != "validated_live" and
                not allow_dry_model):
            raise ValueError("validation_status_not_validated_live:%s" %
                             validation_status)
        hashes = dict(manifest.get("hashes") or {})
        expected_hash = str(hashes.get("model_npz_sha256", ""))
        if expected_hash:
            actual_hash = _sha256_file(model_path)
            if actual_hash != expected_hash:
                raise ValueError("model_hash_mismatch")
        with zipfile.ZipFile(model_path, "r") as archive:
            model_payload = json.loads(archive.read("model.json").decode("utf-8"))
        model = OnlineArxModel.from_payload(model_payload)
        preprocessing = json.loads(preprocessing_path.read_text(encoding="utf-8"))
        if not isinstance(preprocessing, dict):
            raise ValueError("preprocessing_schema_mismatch:not_an_object")
        preprocessing_schema = str(preprocessing.get("schema_version", ""))
        if preprocessing_schema != RESIDUAL_MPC_PREPROCESSING_SCHEMA_VERSION:
            raise ValueError(
                "preprocessing_schema_mismatch:%s" %
                (preprocessing_schema or "<missing>"))
        _validate_manifest_model_consistency(manifest, model)
        return cls(root=root, manifest=manifest, preprocessing=preprocessing, model=model)


@dataclass(frozen=True)
class PlanningAwareV5ResidualQpMpcConfig(PlanningAwareV4MpcPhaseAConfig):
    controller_version: str = PLANNING_AWARE_V5_RESIDUAL_QP_MPC_VERSION
    telemetry_schema_version: str = PLANNING_AWARE_V5_RESIDUAL_QP_MPC_SCHEMA_VERSION
    config_path: str = ""
    model_artifact_dir: str = ""
    require_validated_live: bool = True
    allow_dry_model: bool = False
    shadow_mode: bool = False
    strict_step07_shadow_validation: bool = False
    solver_name: str = "auto"
    solver_max_iterations: int = 80
    solver_tolerance: float = 1.0e-8
    max_compute_ms: float = 40.0
    timing_budget_metric: str = RESIDUAL_MPC_LEGACY_TIMING_BUDGET_METRIC
    residual_incremental_budget_ms: Optional[float] = None
    # These flags opt into the new detailed stage timings. Existing solver
    # diagnostic fields remain schema-present for backward compatibility.
    record_stage_timing: bool = False
    record_solver_diagnostics: bool = False
    debug_qp_snapshot_enabled: bool = False
    debug_qp_snapshot_every_n: int = 0
    debug_qp_snapshot_event_window_only: bool = True
    debug_qp_snapshot_max_rows: int = 200
    first_action_mismatch_tolerance: float = 1.0e-5
    output_tracking_weight: float = 1.0
    body_motion_weight: float = 0.25
    residual_effort_weight: float = 0.10
    residual_slew_weight: float = 1.0
    terminal_output_weight: float = 0.25
    trust_support_scale: float = 1.0
    max_control_blocks: int = 8
    min_support_bound: float = 0.0
    require_valid_planning: bool = False
    require_suspension_state_for_projection: bool = True

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> "PlanningAwareV5ResidualQpMpcConfig":
        allowed = {item.name for item in fields(cls)}
        kwargs = {
            key: _coerce_config_value(value)
            for key, value in dict(values or {}).items()
            if key in allowed
        }
        return cls(**kwargs)


class PlanningAwareV5ResidualQpMpcController(
        PlanningAwareV3MpcPrimaryController):
    """Step 05 applied-residual QP-MPC controller."""

    name = "planning_aware_v5_residual_qp_mpc"
    requires_suspension_state = True

    def __init__(
        self,
        config: Optional[PlanningAwareV5ResidualQpMpcConfig] = None,
    ):
        self.config = config or PlanningAwareV5ResidualQpMpcConfig()
        super().__init__(self.config)
        self.phase_a_controller = PlanningAwareV4MpcPhaseAController(
            PlanningAwareV4MpcPhaseAConfig.from_mapping(
                _phase_a_config_values(self.config)))
        self._artifact: Optional[ResidualMpcArtifact] = None
        self._artifact_error = ""
        self._artifact_loaded = False
        self._filtered_outputs: Dict[str, _LowPassState] = {}
        self._y_history: List[Tuple[float, ...]] = []
        self._u_history: List[Tuple[float, ...]] = []
        self._phase_history: List[Tuple[float, ...]] = []
        self._warm_start: Tuple[float, ...] = ()
        self._solver_adapter_cached: Optional[ResidualQpSolverAdapter] = None
        self._debug_qp_snapshot_candidates = 0
        self._debug_qp_snapshot_captures = 0
        self._debug_qp_snapshot_queue: List[DebugQpSnapshotRecord] = []

    def reset(self, native_suspension: Any = None) -> None:
        super().reset(native_suspension)
        self.phase_a_controller.reset(native_suspension)
        self._filtered_outputs = {}
        self._y_history = []
        self._u_history = []
        self._phase_history = []
        self._warm_start = ()
        self._debug_qp_snapshot_candidates = 0
        self._debug_qp_snapshot_captures = 0
        self._debug_qp_snapshot_queue = []

    def compute(self, context: ControllerContext) -> ControllerOutput:
        start_ns = time.perf_counter_ns()
        cfg = self.config
        record_stage_timing = bool(cfg.record_stage_timing)
        record_solver_diagnostics = bool(cfg.record_solver_diagnostics)
        # The two Phase A timers are mandatory because they define incremental
        # timing. Detailed residual/solver stage timers honor the opt-in flags.
        # Some timers are deliberately nested: model prediction is contained
        # by QP matrix build, solver setup by solver-call wall time, and safety
        # projection (plus any corrective re-solve) by QP post-processing.
        # These diagnostics describe scopes and must not be summed as a flat
        # exclusive-stage decomposition.
        stage_timing_ms = {
            "phase_a_shadow": 0.0,
            "phase_a_commit": 0.0,
            "feature_build": 0.0,
            "model_predict": 0.0,
            "qp_matrix_build": 0.0,
            "qp_setup_or_update": 0.0,
            "qp_solve_wall": 0.0,
            "qp_postprocess": 0.0,
            "safety_projection": 0.0,
        }

        def solve_with_stage_timing(
            problem_data: DenseQpProblem,
            warm_start: Optional[Sequence[float]],
        ) -> SolverResult:
            solve_start_ns = (
                time.perf_counter_ns() if record_solver_diagnostics else 0)
            try:
                result = self._solver_adapter().solve(
                    problem_data,
                    warm_start)
            finally:
                if record_solver_diagnostics:
                    stage_timing_ms["qp_solve_wall"] += max(
                        0.0,
                        (time.perf_counter_ns() - solve_start_ns) / 1.0e6)
            # Every built-in adapter measures setup_ms with perf_counter_ns().
            # Accumulation keeps initial and corrective re-solve setup visible.
            if record_solver_diagnostics:
                stage_timing_ms["qp_setup_or_update"] += max(
                    0.0,
                    _safe_float(result.setup_ms, 0.0))
            return result

        wheel_count = self._wheel_count(context)
        self._ensure_previous_count(wheel_count)
        phase_a_shadow_start_ns = time.perf_counter_ns()
        phase_a_shadow = self.phase_a_controller.compute_shadow(context)
        stage_timing_ms["phase_a_shadow"] = max(
            0.0,
            (time.perf_counter_ns() - phase_a_shadow_start_ns) / 1.0e6)
        feature_build_start_ns = (
            time.perf_counter_ns() if record_stage_timing else 0)
        phase_a_dampers = self._command_dampers(
            phase_a_shadow.command,
            wheel_count,
            default=1.0)
        phase_a_dampers = self._clamped_finite_dampers(
            phase_a_dampers,
            wheel_count)
        phase_a_modal_all = wheel_residual_to_modal(tuple(
            value - 1.0 for value in _match_length(phase_a_dampers, 4, 1.0)))
        artifact = self._load_artifact_once()
        fallback_reason = ""
        solver_result = SolverResult(status="not_run")
        initial_solver_result = SolverResult(status="not_run")
        first_action_resolve_result = SolverResult(status="not_run")
        # These are captured before the corresponding solver calls.  Do not
        # infer them from ``self._warm_start`` later: success/fallback updates
        # that controller state after the solve and would invert the evidence.
        initial_warm_start_input: Tuple[float, ...] = ()
        initial_warm_start_provenance = "none"
        first_action_resolve_warm_start_input: Tuple[float, ...] = ()
        first_action_resolve_warm_start_provenance = "none"
        problem: Optional[DenseQpProblem] = None
        first_action_resolve_problem: Optional[DenseQpProblem] = None
        initial_zero_feasibility = _zero_feasibility_not_checked(
            "initial_qp_not_constructed")
        resolve_zero_feasibility = _zero_feasibility_not_checked(
            "first_action_resolve_not_constructed")
        first_action_resolve_status = "not_run"
        controller_solver_outcome = "not_run"
        support_bounds: Tuple[Tuple[float, float], ...] = ()
        re_solve_count = 0
        re_solve_attempted = 0
        re_solve_succeeded = 0
        re_solve_failed = 0
        first_action_correction_active = 0
        first_modal = tuple(0.0 for _ in MODAL_BASIS_NAMES)
        raw_model_residual = tuple(0.0 for _ in range(wheel_count))
        qp_solution_residual = tuple(0.0 for _ in range(wheel_count))
        pre_budget_residual = tuple(0.0 for _ in range(wheel_count))
        qp_solution_finite = False
        safety_projection_pass = False
        requested_residual = tuple(0.0 for _ in range(wheel_count))
        projected_residual = tuple(0.0 for _ in range(wheel_count))
        final_residual = tuple(0.0 for _ in range(wheel_count))
        final_dampers = phase_a_dampers
        qp_first_dampers = phase_a_dampers
        raw_qp_first_dampers = phase_a_dampers
        first_action_trace = _empty_first_action_trace(phase_a_dampers)
        bounds_guard_active = False
        projection_active = False
        final_guard_active = False
        support_distance = 0.0
        support_scale = 0.0
        cost_info: Dict[str, float] = {}
        history_ready = 0
        mode = "phase_a_fallback"

        y_current: Tuple[float, ...] = ()
        q_current: Tuple[float, ...] = ()
        if artifact is None:
            fallback_reason = "model_unavailable"
        else:
            try:
                y_current = self._current_output_vector(context, artifact)
                q_current = self._current_exogenous_vector(context, artifact)
            except ValueError:
                fallback_reason = "telemetry_invalid"
            else:
                phase_vector = _phase_vector_for_model(
                    artifact.model.phase_a_modes,
                    phase_a_modal_all)
                self._append_current_histories(artifact, y_current, phase_vector)
                history_ready = int(self._history_ready(artifact))
                if not history_ready:
                    fallback_reason = "history_warmup"
                else:
                    preview_reason = self._preview_invalid_reason(context)
                    if preview_reason:
                        fallback_reason = preview_reason

        if artifact is not None and not fallback_reason:
            support_bounds, support_distance, support_scale = (
                self._active_bounds_from_support(
                    artifact,
                    phase_a_dampers,
                    context))
            if not support_bounds:
                fallback_reason = "out_of_support"
        if record_stage_timing:
            stage_timing_ms["feature_build"] = max(
                0.0,
                (time.perf_counter_ns() - feature_build_start_ns) / 1.0e6)

        if (
                artifact is not None and
                support_bounds and
                not fallback_reason):
            model_predict_elapsed_ns = [0] if record_stage_timing else None
            try:
                matrix_build_start_ns = (
                    time.perf_counter_ns() if record_stage_timing else 0)
                try:
                    problem, cost_info = self._build_qp_problem(
                        artifact=artifact,
                        context=context,
                        q_current=q_current,
                        support_bounds=support_bounds,
                        phase_a_dampers=phase_a_dampers,
                        model_predict_elapsed_ns=model_predict_elapsed_ns)
                finally:
                    if record_stage_timing:
                        stage_timing_ms["qp_matrix_build"] += max(
                            0.0,
                            (time.perf_counter_ns() -
                             matrix_build_start_ns) / 1.0e6)
                        stage_timing_ms["model_predict"] += max(
                            0.0,
                            model_predict_elapsed_ns[0] / 1.0e6)
                initial_zero_feasibility = check_zero_vector_feasibility(problem)
                initial_warm_start_input = tuple(self._warm_start)
                initial_warm_start_provenance = (
                    "previous_controller_solution"
                    if initial_warm_start_input else "none")
                initial_solver_result = solve_with_stage_timing(
                    problem,
                    initial_warm_start_input or None)
                solver_result = initial_solver_result
            except Exception as exc:  # noqa: BLE001 - online fallback.
                solver_result = SolverResult(
                    status="solver_error",
                    error="%s:%s" % (exc.__class__.__name__, exc))
                if problem is not None:
                    initial_solver_result = solver_result
                    controller_solver_outcome = "initial_qp_exception"
                else:
                    controller_solver_outcome = "qp_build_exception"
                fallback_reason = "solver_error"
            if not fallback_reason:
                fallback_reason = _fallback_from_solver(solver_result)
                controller_solver_outcome = (
                    "initial_qp_solved"
                    if not fallback_reason else "initial_qp_non_solved")

        if artifact is not None and problem is not None and not fallback_reason:
            qp_postprocess_start_ns = (
                time.perf_counter_ns() if record_stage_timing else 0)
            solution = tuple(solver_result.primal_solution)
            if not solution or not _all_finite(solution):
                fallback_reason = "nonfinite_solution"
                controller_solver_outcome = "initial_qp_nonfinite_solution"
            else:
                first_active = solution[:len(artifact.active_modes)]
                first_modal = _active_to_candidate_modal(
                    artifact,
                    first_active,
                    MODAL_BASIS_NAMES)
                wheel_residual = _match_length(
                    modal_to_wheel_residual(first_modal),
                    wheel_count,
                    0.0)
                raw_model_residual = tuple(wheel_residual)
                qp_solution_residual = tuple(wheel_residual)
                qp_solution_finite = True
                requested_residual = tuple(wheel_residual)
                qp_first_dampers = tuple(
                    phase_a_dampers[index] + requested_residual[index]
                    for index in range(wheel_count))
                raw_qp_first_dampers = qp_first_dampers
                safety_projection_start_ns = (
                    time.perf_counter_ns() if record_stage_timing else 0)
                projected, projection_active = self._semi_active_project_dampers(
                    qp_first_dampers,
                    context)
                final_dampers, final_guard_active = self._final_safety_dampers(
                    projected,
                    context)
                bounded, bounds_guard_active = (
                    self._candidate_bounds_and_rate_dampers(
                        qp_first_dampers,
                        self.previous_final_damper_scales,
                        context))
                bounds_guard_active = bounds_guard_active or self._l2_distance(
                    bounded,
                    qp_first_dampers) > 1.0e-12
                mismatch_l2, mismatch_linf = _mismatch(qp_first_dampers, final_dampers)
                first_action_trace = self._first_action_projection_trace(
                    qp_first_dampers,
                    context,
                    first_action_tolerance=cfg.first_action_mismatch_tolerance)
                if record_stage_timing:
                    stage_timing_ms["safety_projection"] += max(
                        0.0,
                        (time.perf_counter_ns() -
                         safety_projection_start_ns) / 1.0e6)
                if mismatch_linf > cfg.first_action_mismatch_tolerance:
                    first_action_correction_active = 1
                    re_solve_attempted = 1
                    re_solve_count = 1
                    qp_solution_residual = tuple(
                        0.0 for _ in range(wheel_count))
                    qp_solution_finite = False
                    fixed_point_support_comparison = (
                        compare_first_action_fixed_point_support(
                            problem,
                            artifact.active_modes,
                            phase_a_dampers,
                            first_action_trace.fixed_point_dampers,
                            tolerance=1.0e-10))
                    first_action_resolve_problem = (
                        _tightened_problem_for_first_action(
                            problem,
                            artifact,
                            phase_a_dampers,
                            first_action_trace.fixed_point_dampers,
                            require_within_existing_bounds=True,
                            comparison=fixed_point_support_comparison))
                    fixed_point_support_comparison = replace(
                        fixed_point_support_comparison,
                        corrective_qp_constructed=(
                            first_action_resolve_problem is not None),
                        corrective_qp_rejected=(
                            first_action_resolve_problem is None))
                    first_action_trace = _replace_first_action_trace(
                        first_action_trace,
                        fixed_point_support_comparison=(
                            fixed_point_support_comparison))
                    if first_action_resolve_problem is None:
                        re_solve_failed = 1
                        first_action_resolve_status = (
                            "not_run_fixed_point_out_of_support")
                        controller_solver_outcome = (
                            "first_action_fixed_point_out_of_support")
                        fallback_reason = "solver_infeasible"
                        solver_result = SolverResult(
                            status="solver_infeasible",
                            error="first_action_fixed_point_out_of_support")
                        first_action_trace = _replace_first_action_trace(
                            first_action_trace,
                            fixed_point_out_of_support=1)
                    else:
                        resolve_zero_feasibility = (
                            check_zero_vector_feasibility(
                                first_action_resolve_problem))
                        first_action_resolve_warm_start_input = tuple(solution)
                        first_action_resolve_warm_start_provenance = (
                            "initial_solution_for_corrective")
                        first_action_resolve_result = solve_with_stage_timing(
                            first_action_resolve_problem,
                            first_action_resolve_warm_start_input)
                        solver_result = first_action_resolve_result
                        first_action_resolve_status = solver_result.status
                        if (
                                solver_result.status != "solved" or
                                not solver_result.primal_solution):
                            re_solve_failed = 1
                            controller_solver_outcome = (
                                "first_action_resolve_non_solved")
                            fallback_reason = _fallback_from_solver(solver_result)
                            if not fallback_reason:
                                fallback_reason = "solver_error"
                        elif not _all_finite(solver_result.primal_solution):
                            re_solve_failed = 1
                            controller_solver_outcome = (
                                "first_action_resolve_nonfinite_solution")
                            fallback_reason = "nonfinite_solution"
                    if (
                            not fallback_reason and
                            solver_result.status == "solved" and
                            solver_result.primal_solution):
                        re_solve_succeeded = 1
                        controller_solver_outcome = (
                            "first_action_resolve_solved")
                        solution = tuple(solver_result.primal_solution)
                        first_active = solution[:len(artifact.active_modes)]
                        first_modal = _active_to_candidate_modal(
                            artifact,
                            first_active,
                            MODAL_BASIS_NAMES)
                        requested_residual = _match_length(
                            modal_to_wheel_residual(first_modal),
                            wheel_count,
                            0.0)
                        qp_solution_residual = tuple(requested_residual)
                        qp_solution_finite = bool(
                            requested_residual and
                            _all_finite(requested_residual))
                        qp_first_dampers = tuple(
                            phase_a_dampers[index] + requested_residual[index]
                            for index in range(wheel_count))
                        safety_projection_start_ns = (
                            time.perf_counter_ns()
                            if record_stage_timing else 0)
                        projected, projection_active = (
                            self._semi_active_project_dampers(
                                qp_first_dampers,
                                context))
                        final_dampers, final_guard_active = (
                            self._final_safety_dampers(projected, context))
                        mismatch_l2, mismatch_linf = _mismatch(
                            qp_first_dampers,
                            final_dampers)
                        if record_stage_timing:
                            stage_timing_ms["safety_projection"] += max(
                                0.0,
                                (time.perf_counter_ns() -
                                 safety_projection_start_ns) / 1.0e6)
                    if (
                            not fallback_reason and
                            mismatch_linf > cfg.first_action_mismatch_tolerance):
                        re_solve_failed = 1
                        controller_solver_outcome = "post_projection_mismatch"
                        fallback_reason = "first_action_mismatch"
                if not fallback_reason:
                    projected_residual = tuple(
                        projected[index] - phase_a_dampers[index]
                        for index in range(wheel_count))
                    pre_budget_residual = tuple(
                        final_dampers[index] - phase_a_dampers[index]
                        for index in range(wheel_count))
                    final_residual = pre_budget_residual
                    safety_projection_pass = True
                    if bool(cfg.shadow_mode):
                        mode = "shadow_qp_mpc"
                        final_dampers = phase_a_dampers
                        final_residual = tuple(0.0 for _ in range(wheel_count))
                    else:
                        mode = "residual_qp_mpc"
                    self._warm_start = tuple(solution)
            if record_stage_timing:
                stage_timing_ms["qp_postprocess"] += max(
                    0.0,
                    (time.perf_counter_ns() -
                     qp_postprocess_start_ns) / 1.0e6)

        phase_a_commit_start_ns = time.perf_counter_ns()
        phase_a_commit = self.phase_a_controller.compute(context)
        stage_timing_ms["phase_a_commit"] = max(
            0.0,
            (time.perf_counter_ns() - phase_a_commit_start_ns) / 1.0e6)
        phase_a_commit_match = int(
            phase_a_commit.command.is_close(phase_a_shadow.command))
        if fallback_reason:
            final_dampers = phase_a_dampers
            final_residual = tuple(0.0 for _ in range(wheel_count))
            projected_residual = tuple(0.0 for _ in range(wheel_count))
            requested_residual = tuple(0.0 for _ in range(wheel_count))
            first_modal = tuple(0.0 for _ in MODAL_BASIS_NAMES)
            qp_first_dampers = phase_a_dampers
            self._warm_start = ()
            mode = "phase_a_fallback"
        final_modal = wheel_residual_to_modal(_match_length(
            final_residual,
            4,
            0.0))
        if artifact is not None:
            self._append_applied_residual(artifact, final_modal)
        command = self._command_from_dampers(final_dampers)
        # Preserve the historical legacy boundary: command ready, before
        # diagnostics construction and all suite CSV serialization/write I/O.
        compute_ms = max(0.0, (time.perf_counter_ns() - start_ns) / 1.0e6)
        budget_metric_name, budget_ms = _timing_budget_selection(cfg)
        budget_metric_ms = (
            compute_ms
            if budget_metric_name == RESIDUAL_MPC_LEGACY_TIMING_BUDGET_METRIC
            else 0.0)
        compute_budget_exceeded = (
            budget_metric_ms > budget_ms
            if budget_metric_name == RESIDUAL_MPC_LEGACY_TIMING_BUDGET_METRIC
            else False)
        if compute_budget_exceeded and not fallback_reason:
            if controller_solver_outcome != "not_run":
                controller_solver_outcome += "__timing_rejected"
            fallback_reason = "compute_budget_exceeded"
            final_dampers = phase_a_dampers
            final_residual = tuple(0.0 for _ in range(wheel_count))
            projected_residual = tuple(0.0 for _ in range(wheel_count))
            requested_residual = tuple(0.0 for _ in range(wheel_count))
            first_modal = tuple(0.0 for _ in MODAL_BASIS_NAMES)
            final_modal = tuple(0.0 for _ in MODAL_BASIS_NAMES)
            qp_first_dampers = phase_a_dampers
            command = self._command_from_dampers(final_dampers)
            mode = "phase_a_fallback"

        def current_lifecycle(
            selected_metric_exceeded: bool,
        ) -> ResidualLifecycleTrace:
            budget_accepted_residual = (
                pre_budget_residual
                if safety_projection_pass and not selected_metric_exceeded else
                tuple(0.0 for _ in range(wheel_count)))
            return ResidualLifecycleTrace(
                raw_model_residual=tuple(raw_model_residual),
                qp_solution_residual=tuple(qp_solution_residual),
                budget_accepted_residual=tuple(budget_accepted_residual),
                final_applied_residual=tuple(final_residual),
                qp_solution_finite=bool(qp_solution_finite),
                would_propose_without_timing_gate=bool(
                    safety_projection_pass and
                    _residual_nonzero(pre_budget_residual)),
                timing_gate_pass=not selected_metric_exceeded,
                safety_projection_pass=bool(safety_projection_pass),
                initial_solver_result=initial_solver_result,
                first_action_resolve_status=first_action_resolve_status,
                controller_solver_outcome=controller_solver_outcome,
                initial_zero_feasibility=initial_zero_feasibility,
                resolve_zero_feasibility=resolve_zero_feasibility)

        def build_current_diagnostics(
            controller_end_to_end_ms: float,
            incremental_compute_ms: float,
            selected_metric_ms: float,
            selected_metric_exceeded: bool,
        ) -> Dict[str, Any]:
            return self._diagnostics(
                context=context,
                artifact=artifact,
                phase_a_shadow_diagnostics=phase_a_shadow.diagnostics,
                phase_a_dampers=phase_a_dampers,
                phase_a_modal=phase_a_modal_all,
                requested_modal=first_modal,
                final_modal=final_modal,
                requested_residual=requested_residual,
                projected_residual=projected_residual,
                final_residual=final_residual,
                qp_first_dampers=qp_first_dampers,
                raw_qp_first_dampers=raw_qp_first_dampers,
                first_action_trace=first_action_trace,
                final_dampers=final_dampers,
                mode=mode,
                fallback_reason=fallback_reason,
                solver_result=solver_result,
                re_solve_count=re_solve_count,
                re_solve_attempted=re_solve_attempted,
                re_solve_succeeded=re_solve_succeeded,
                re_solve_failed=re_solve_failed,
                first_action_correction_active=first_action_correction_active,
                support_distance=support_distance,
                support_scale=support_scale,
                history_ready=history_ready,
                cost_info=cost_info,
                bounds_guard_active=bounds_guard_active,
                projection_active=projection_active,
                final_guard_active=final_guard_active,
                phase_a_commit_match=phase_a_commit_match,
                compute_ms=compute_ms,
                controller_end_to_end_ms=controller_end_to_end_ms,
                incremental_compute_ms=incremental_compute_ms,
                budget_metric_name=budget_metric_name,
                budget_metric_ms=selected_metric_ms,
                budget_ms=budget_ms,
                compute_budget_exceeded=selected_metric_exceeded,
                stage_timing_ms=stage_timing_ms,
                initial_warm_start_input=initial_warm_start_input,
                initial_warm_start_provenance=initial_warm_start_provenance,
                first_action_resolve_warm_start_input=(
                    first_action_resolve_warm_start_input),
                first_action_resolve_warm_start_provenance=(
                    first_action_resolve_warm_start_provenance),
                lifecycle=current_lifecycle(selected_metric_exceeded))

        diagnostics = build_current_diagnostics(
            0.0,
            0.0,
            budget_metric_ms,
            compute_budget_exceeded)
        controller_end_to_end_ms = max(
            0.0,
            (time.perf_counter_ns() - start_ns) / 1.0e6)
        incremental_compute_ms = max(
            0.0,
            controller_end_to_end_ms -
            stage_timing_ms["phase_a_shadow"] -
            stage_timing_ms["phase_a_commit"])
        budget_metric_values = {
            RESIDUAL_MPC_LEGACY_TIMING_BUDGET_METRIC: compute_ms,
            RESIDUAL_MPC_END_TO_END_TIMING_BUDGET_METRIC: (
                controller_end_to_end_ms),
            RESIDUAL_MPC_INCREMENTAL_TIMING_BUDGET_METRIC: (
                incremental_compute_ms),
        }
        budget_metric_ms = budget_metric_values[budget_metric_name]
        compute_budget_exceeded = budget_metric_ms > budget_ms

        if (
                budget_metric_name !=
                RESIDUAL_MPC_LEGACY_TIMING_BUDGET_METRIC and
                compute_budget_exceeded and
                not fallback_reason):
            if controller_solver_outcome != "not_run":
                controller_solver_outcome += "__timing_rejected"
            fallback_reason = "compute_budget_exceeded"
            final_dampers = phase_a_dampers
            final_residual = tuple(0.0 for _ in range(wheel_count))
            projected_residual = tuple(0.0 for _ in range(wheel_count))
            requested_residual = tuple(0.0 for _ in range(wheel_count))
            first_modal = tuple(0.0 for _ in MODAL_BASIS_NAMES)
            final_modal = tuple(0.0 for _ in MODAL_BASIS_NAMES)
            qp_first_dampers = phase_a_dampers
            command = self._command_from_dampers(final_dampers)
            mode = "phase_a_fallback"
            diagnostics = build_current_diagnostics(
                controller_end_to_end_ms,
                incremental_compute_ms,
                budget_metric_ms,
                True)
            controller_end_to_end_ms = max(
                0.0,
                (time.perf_counter_ns() - start_ns) / 1.0e6)
            incremental_compute_ms = max(
                0.0,
                controller_end_to_end_ms -
                stage_timing_ms["phase_a_shadow"] -
                stage_timing_ms["phase_a_commit"])
            budget_metric_values[
                RESIDUAL_MPC_END_TO_END_TIMING_BUDGET_METRIC
            ] = controller_end_to_end_ms
            budget_metric_values[
                RESIDUAL_MPC_INCREMENTAL_TIMING_BUDGET_METRIC
            ] = incremental_compute_ms
            budget_metric_ms = budget_metric_values[budget_metric_name]
            compute_budget_exceeded = budget_metric_ms > budget_ms

        diagnostics.update({
            "residual_mpc_controller_end_to_end_ms": controller_end_to_end_ms,
            "residual_mpc_incremental_compute_ms": incremental_compute_ms,
            "residual_mpc_budget_metric_name": budget_metric_name,
            "residual_mpc_budget_metric_ms": budget_metric_ms,
            "residual_mpc_budget_ms": budget_ms,
            "residual_mpc_compute_budget_exceeded": int(
                compute_budget_exceeded),
        })
        diagnostics.update(_residual_lifecycle_diagnostics(
            current_lifecycle(compute_budget_exceeded)))
        debug_snapshot_key = ""
        debug_snapshot_candidate_ordinal = 0
        if problem is not None:
            snapshot_stages = [DebugQpStageRecord(
                stage="initial",
                solve_ordinal=1,
                problem=problem,
                zero_feasibility=initial_zero_feasibility,
                solver_result=initial_solver_result,
                warm_start_input=initial_warm_start_input,
                warm_start_provenance=initial_warm_start_provenance)]
            if first_action_resolve_problem is not None:
                snapshot_stages.append(DebugQpStageRecord(
                    stage="first_action_resolve",
                    solve_ordinal=2,
                    problem=first_action_resolve_problem,
                    zero_feasibility=resolve_zero_feasibility,
                    solver_result=first_action_resolve_result,
                    warm_start_input=first_action_resolve_warm_start_input,
                    warm_start_provenance=(
                        first_action_resolve_warm_start_provenance)))
            (
                debug_snapshot_key,
                debug_snapshot_candidate_ordinal,
            ) = self._queue_debug_qp_snapshot(
                context,
                controller_solver_outcome,
                tuple(snapshot_stages),
                first_action_trace.fixed_point_support_comparison)
        diagnostics["residual_mpc_debug_qp_snapshot_key"] = (
            debug_snapshot_key)
        diagnostics["residual_mpc_debug_qp_snapshot_candidate_ordinal"] = (
            debug_snapshot_candidate_ordinal)
        self.previous_final_damper_scales = list(final_dampers)
        return ControllerOutput(command=command, diagnostics=diagnostics)

    def _solver_adapter(self) -> ResidualQpSolverAdapter:
        if self._solver_adapter_cached is None:
            self._solver_adapter_cached = _make_solver(self.config)
        return self._solver_adapter_cached

    def _solver_name_for_diagnostics(self) -> str:
        if self._solver_adapter_cached is not None:
            return self._solver_adapter_cached.name
        return _solver_name(self.config)

    def _config_exists_for_diagnostics(self) -> int:
        path = str(getattr(self.config, "config_path", "") or "")
        return int(bool(path) and os.path.isfile(os.path.expanduser(path)))

    def _model_artifact_dir_exists_for_diagnostics(self) -> int:
        path = str(getattr(self.config, "model_artifact_dir", "") or "")
        return int(bool(path) and os.path.isdir(os.path.expanduser(path)))

    def _queue_debug_qp_snapshot(
        self,
        context: ControllerContext,
        controller_solver_outcome: str,
        stages: Tuple[DebugQpStageRecord, ...],
        first_action_support_comparison: Optional[
            FixedPointSupportComparison],
    ) -> Tuple[str, int]:
        cfg = self.config
        if not bool(cfg.debug_qp_snapshot_enabled) or not stages:
            return "", 0
        self._debug_qp_snapshot_candidates += 1
        every_n = max(0, int(_safe_float(
            cfg.debug_qp_snapshot_every_n,
            0.0)))
        max_rows = max(0, int(_safe_float(
            cfg.debug_qp_snapshot_max_rows,
            200.0)))
        if (
                every_n <= 0 or
                self._debug_qp_snapshot_candidates % every_n != 0 or
                self._debug_qp_snapshot_captures >= max_rows):
            return "", 0
        event_window = _debug_qp_snapshot_event_window(context)
        if bool(cfg.debug_qp_snapshot_event_window_only) and not event_window:
            return "", 0
        state = getattr(context, "state", None)
        frame = int(_safe_float(getattr(state, "frame", 0), 0.0))
        step = int(_safe_float(
            getattr(context, "step", getattr(state, "step", 0)),
            0.0))
        key = "%d:%d:%d" % (
            frame,
            step,
            self._debug_qp_snapshot_candidates)
        solver_adapter = self._solver_adapter()
        self._debug_qp_snapshot_queue.append(DebugQpSnapshotRecord(
            key=key,
            frame=frame,
            step=step,
            event_window=event_window,
            controller_solver_outcome=controller_solver_outcome,
            solver_name=self._solver_name_for_diagnostics(),
            solver_max_iterations=int(cfg.solver_max_iterations),
            solver_tolerance=float(cfg.solver_tolerance),
            solver_setup_arguments=_debug_solver_setup_arguments(
                solver_adapter,
                cfg),
            runtime_provenance=_debug_runtime_provenance(),
            stages=stages,
            first_action_support_comparison=(
                first_action_support_comparison)))
        self._debug_qp_snapshot_captures += 1
        return key, self._debug_qp_snapshot_candidates

    def drain_debug_qp_snapshots(self) -> Tuple[Mapping[str, Any], ...]:
        """Serialize and drain queued rows after ``compute()`` returns."""

        queued = tuple(self._debug_qp_snapshot_queue)
        self._debug_qp_snapshot_queue = []
        return tuple(_debug_qp_snapshot_to_dict(row) for row in queued)

    def _first_action_projection_trace(
        self,
        raw_dampers: Sequence[float],
        context: ControllerContext,
        *,
        first_action_tolerance: float,
    ) -> FirstActionProjectionTrace:
        raw = tuple(_safe_float(value, 1.0) for value in raw_dampers)
        projected, _ = self._semi_active_project_dampers(raw, context)
        safety, _ = self._final_safety_dampers(projected, context)
        reprojected, _ = self._semi_active_project_dampers(safety, context)
        refinal, _ = self._final_safety_dampers(reprojected, context)

        fixed = self._first_action_direct_fixed_point(raw, context)
        fixed_iterations = 1
        fixed_projected, _ = self._semi_active_project_dampers(fixed, context)
        fixed_final, _ = self._final_safety_dampers(fixed_projected, context)
        fixed_linf = _mismatch(fixed, fixed_final)[1]
        if fixed_linf > first_action_tolerance:
            current = fixed
            for iteration in range(2, 12):
                iter_projected, _ = self._semi_active_project_dampers(
                    current,
                    context)
                iter_final, _ = self._final_safety_dampers(
                    iter_projected,
                    context)
                fixed_iterations = iteration
                fixed_linf = _mismatch(current, iter_final)[1]
                fixed = iter_final
                current = iter_final
                if fixed_linf <= first_action_tolerance:
                    break

        minmax_count, rate_count, wheel_bound_miss = (
            self._first_action_clip_counts(raw, context))
        low_rel_count, infeasible_count = (
            self._semi_active_blend_counts(raw, context))
        raw_to_projected = _mismatch(raw, projected)
        projected_to_final = _mismatch(projected, safety)
        final_to_reprojected = _mismatch(safety, refinal)
        pre_correction = _mismatch(raw, safety)
        return FirstActionProjectionTrace(
            raw_dampers=raw,
            semi_active_dampers=projected,
            safety_dampers=safety,
            reprojected_dampers=refinal,
            fixed_point_dampers=fixed,
            pre_correction_l2=pre_correction[0],
            pre_correction_linf=pre_correction[1],
            raw_to_projected_l2=raw_to_projected[0],
            raw_to_projected_linf=raw_to_projected[1],
            projected_to_final_l2=projected_to_final[0],
            projected_to_final_linf=projected_to_final[1],
            final_to_reprojected_l2=final_to_reprojected[0],
            final_to_reprojected_linf=final_to_reprojected[1],
            fixed_point_linf=fixed_linf,
            fixed_point_iterations=fixed_iterations,
            minmax_clip_count=minmax_count,
            rate_clip_count=rate_count,
            low_rel_blend_count=low_rel_count,
            infeasible_blend_count=infeasible_count,
            modal_coupling_bound_miss_count=wheel_bound_miss)

    def _first_action_direct_fixed_point(
        self,
        raw_dampers: Sequence[float],
        context: ControllerContext,
    ) -> Tuple[float, ...]:
        cfg = self.config
        raw = tuple(_safe_float(value, 1.0) for value in raw_dampers)
        wheel_count = len(raw)
        low = _safe_float(cfg.min_damper_scale, 0.75)
        high = _safe_float(cfg.max_damper_scale, 1.25)
        dt = _context_dt(context, cfg.default_dt)
        max_up = max(0.0, _safe_float(cfg.max_rate_up_scale_per_s, 1.2)) * dt
        max_down = (
            max(0.0, _safe_float(cfg.max_rate_down_scale_per_s, 1.2)) * dt)
        previous = _match_length(
            self.previous_final_damper_scales,
            wheel_count,
            1.0)
        projection_config = self._projection_config()
        native_dampers = self._native_dampers(context, wheel_count)
        rel_velocities = self._relative_extension_velocities(context, wheel_count)
        corner_velocities = self._corner_vertical_velocities(context.state)
        corner_velocities = _match_length(corner_velocities, wheel_count, 0.0)
        neutral = _safe_float(cfg.neutral_damper_scale, 1.0)
        low_rel = max(_safe_float(cfg.rel_velocity_deadband, 0.015),
                      _safe_float(cfg.projection_eps, 1.0e-6))
        fixed: List[float] = []
        for index, value in enumerate(raw):
            wheel_low = max(low, previous[index] - max_down)
            wheel_high = min(high, previous[index] + max_up)
            native = max(_safe_float(native_dampers[index], 4500.0), 1.0e-9)
            v_eff = _safe_float(corner_velocities[index], 0.0)
            v_rel = _safe_float(rel_velocities[index], 0.0)
            c_sky = native * max(abs(value - neutral), 0.05)
            result = canonical_skyhook_v3_projection(
                v_eff_i=v_eff,
                v_rel_extension_i=v_rel,
                C_native_i=native,
                C_sky_i=c_sky,
                config=projection_config)
            if (
                    abs(v_rel) <= low_rel and
                    clamp(cfg.projection_blend_low_rel, 0.0, 1.0) > 0.0):
                target = neutral
            elif (
                    int(result.get("semi_active_feasible", 0) or 0) != 1 and
                    clamp(cfg.projection_blend_infeasible, 0.0, 1.0) > 0.0):
                target = _safe_float(result["final_target_damper"], neutral)
            else:
                target = value
            fixed.append(clamp(target, wheel_low, wheel_high))
        return tuple(fixed)

    def _first_action_clip_counts(
        self,
        raw_dampers: Sequence[float],
        context: ControllerContext,
    ) -> Tuple[int, int, int]:
        cfg = self.config
        raw = tuple(_safe_float(value, 1.0) for value in raw_dampers)
        wheel_count = len(raw)
        low = _safe_float(cfg.min_damper_scale, 0.75)
        high = _safe_float(cfg.max_damper_scale, 1.25)
        dt = _context_dt(context, cfg.default_dt)
        max_up = max(0.0, _safe_float(cfg.max_rate_up_scale_per_s, 1.2)) * dt
        max_down = (
            max(0.0, _safe_float(cfg.max_rate_down_scale_per_s, 1.2)) * dt)
        previous = _match_length(
            self.previous_final_damper_scales,
            wheel_count,
            1.0)
        minmax_count = 0
        rate_count = 0
        wheel_bound_miss = 0
        for index, value in enumerate(raw):
            if value < low - 1.0e-12 or value > high + 1.0e-12:
                minmax_count += 1
            wheel_low = max(low, previous[index] - max_down)
            wheel_high = min(high, previous[index] + max_up)
            if value < wheel_low - 1.0e-12 or value > wheel_high + 1.0e-12:
                rate_count += 1
                wheel_bound_miss += 1
        return minmax_count, rate_count, wheel_bound_miss

    def _semi_active_blend_counts(
        self,
        desired: Sequence[float],
        context: ControllerContext,
    ) -> Tuple[int, int]:
        cfg = self.config
        wheel_count = len(desired)
        projection_config = self._projection_config()
        native_dampers = self._native_dampers(context, wheel_count)
        rel_velocities = self._relative_extension_velocities(context, wheel_count)
        corner_velocities = self._corner_vertical_velocities(context.state)
        corner_velocities = _match_length(corner_velocities, wheel_count, 0.0)
        neutral = _safe_float(cfg.neutral_damper_scale, 1.0)
        low_rel = max(_safe_float(cfg.rel_velocity_deadband, 0.015),
                      _safe_float(cfg.projection_eps, 1.0e-6))
        low_rel_count = 0
        infeasible_count = 0
        for index, desired_value in enumerate(desired):
            value = _safe_float(desired_value, neutral)
            native = max(_safe_float(native_dampers[index], 4500.0), 1.0e-9)
            v_eff = _safe_float(corner_velocities[index], 0.0)
            v_rel = _safe_float(rel_velocities[index], 0.0)
            c_sky = native * max(abs(value - neutral), 0.05)
            result = canonical_skyhook_v3_projection(
                v_eff_i=v_eff,
                v_rel_extension_i=v_rel,
                C_native_i=native,
                C_sky_i=c_sky,
                config=projection_config)
            if (
                    abs(v_rel) <= low_rel and
                    abs(value - neutral) > 1.0e-12 and
                    clamp(cfg.projection_blend_low_rel, 0.0, 1.0) > 0.0):
                low_rel_count += 1
            elif (
                    int(result.get("semi_active_feasible", 0) or 0) != 1 and
                    clamp(cfg.projection_blend_infeasible, 0.0, 1.0) > 0.0):
                target = _safe_float(result["final_target_damper"], neutral)
                if abs(value - target) > 1.0e-12:
                    infeasible_count += 1
        return low_rel_count, infeasible_count

    def _load_artifact_once(self) -> Optional[ResidualMpcArtifact]:
        if not self._artifact_loaded:
            self._artifact_loaded = True
            try:
                self._artifact = ResidualMpcArtifact.load(
                    self.config.model_artifact_dir,
                    require_validated_live=bool(
                        self.config.require_validated_live),
                    allow_dry_model=bool(self.config.allow_dry_model))
                self._artifact_error = ""
                self._prime_histories(self._artifact)
            except Exception as exc:  # noqa: BLE001 - fallback diagnostics.
                self._artifact = None
                self._artifact_error = "%s:%s" % (exc.__class__.__name__, exc)
                if bool(getattr(
                        self.config,
                        "strict_step07_shadow_validation",
                        False)):
                    raise RuntimeError(
                        "strict Step07 residual QP-MPC shadow model artifact "
                        "load failed: %s" % self._artifact_error) from exc
        return self._artifact

    def _prime_histories(self, artifact: ResidualMpcArtifact) -> None:
        u_dim = len(artifact.model.input_modes)
        phase_dim = len(artifact.model.phase_a_modes)
        minimum = max(
            artifact.model.input_delay_ticks + artifact.model.orders.input_lags + 2,
            1)
        self._u_history = [tuple(0.0 for _ in range(u_dim)) for _ in range(minimum)]
        self._phase_history = [
            tuple(0.0 for _ in range(phase_dim)) for _ in range(minimum)
        ]

    def _append_current_histories(
        self,
        artifact: ResidualMpcArtifact,
        y_current: Sequence[float],
        phase_vector: Sequence[float],
    ) -> None:
        self._y_history.append(tuple(float(value) for value in y_current))
        self._phase_history.append(tuple(float(value) for value in phase_vector))
        max_len = max(
            artifact.model.orders.output_lags + artifact.model.input_delay_ticks + 8,
            artifact.model.orders.input_lags + artifact.model.input_delay_ticks + 8,
            artifact.manifest.get("recommended_horizon_steps", 10) + 16,
        )
        self._y_history = self._y_history[-max_len:]
        self._phase_history = self._phase_history[-max_len:]
        self._u_history = self._u_history[-max_len:]

    def _append_applied_residual(
        self,
        artifact: ResidualMpcArtifact,
        final_modal_all: Sequence[float],
    ) -> None:
        values = _phase_vector_for_model(artifact.model.input_modes, final_modal_all)
        self._u_history.append(tuple(values))

    def _history_ready(self, artifact: ResidualMpcArtifact) -> bool:
        model = artifact.model
        return (
            len(self._y_history) >= model.orders.output_lags and
            len(self._u_history) >= model.input_delay_ticks + model.orders.input_lags + 1 and
            len(self._phase_history) >= model.input_delay_ticks + model.orders.input_lags + 1)

    def _current_output_vector(
        self,
        context: ControllerContext,
        artifact: ResidualMpcArtifact,
    ) -> Tuple[float, ...]:
        dt = _context_dt(context, self.config.default_dt)
        tau = _safe_float(
            artifact.preprocessing.get("low_pass_tau_s", 0.15),
            0.15)
        values = []
        for channel in artifact.model.output_channels:
            value = self._channel_value(channel, context, dt, tau)
            if not math.isfinite(value):
                raise ValueError("nonfinite_output:%s" % channel)
            values.append(value)
        return tuple(values)

    def _current_exogenous_vector(
        self,
        context: ControllerContext,
        artifact: ResidualMpcArtifact,
    ) -> Tuple[float, ...]:
        values = []
        for channel in artifact.model.exogenous_channels:
            value = _exogenous_channel_value(channel, context)
            if not math.isfinite(value):
                raise ValueError("nonfinite_exogenous:%s" % channel)
            values.append(value)
        return tuple(values)

    def _channel_value(
        self,
        channel: str,
        context: ControllerContext,
        dt: float,
        tau: float,
    ) -> float:
        state = getattr(context, "state", VehicleState())
        raw_map = {
            "residual_mpc_roll_deg": getattr(state, "roll", 0.0),
            "residual_mpc_roll_rate_degps": getattr(state, "roll_rate", 0.0),
            "residual_mpc_pitch_deg": getattr(state, "pitch", 0.0),
            "residual_mpc_pitch_rate_degps": getattr(state, "pitch_rate", 0.0),
            "residual_mpc_yaw_rate_degps": getattr(state, "yaw_rate", 0.0),
            "residual_mpc_ax_body_mps2": getattr(state, "local_ax", 0.0),
            "residual_mpc_ay_body_mps2": getattr(state, "local_ay", 0.0),
            "residual_mpc_ax_world_mps2": getattr(state, "ax", 0.0),
            "residual_mpc_ay_world_mps2": getattr(state, "ay", 0.0),
        }
        if channel.endswith("_raw"):
            base = channel[:-4]
            return _safe_float(raw_map.get(base, 0.0), 0.0)
        if channel.endswith("_filt"):
            base = channel[:-5]
            raw = _safe_float(raw_map.get(base, 0.0), 0.0)
            state_obj = self._filtered_outputs.get(channel)
            if state_obj is None:
                state_obj = _LowPassState(tau_s=tau)
                self._filtered_outputs[channel] = state_obj
            return state_obj.update(raw, dt)
        if channel in raw_map:
            return _safe_float(raw_map[channel], 0.0)
        raise ValueError("unsupported_output_channel:%s" % channel)

    def _preview_invalid_reason(self, context: ControllerContext) -> str:
        planning = getattr(context, "planning", None)
        if bool(self.config.require_valid_planning):
            if planning is None or not bool(getattr(planning, "available", False)):
                return "preview_invalid"
            age = _planning_age_frames(context)
            if (
                    age is not None and
                    age > int(_safe_float(self.config.max_planning_age_frames, 5))):
                return "preview_stale"
        if (
                bool(self.config.require_suspension_state_for_projection) and
                not bool(getattr(context, "suspension_state_valid", False))):
            return "telemetry_invalid"
        return ""

    def _active_bounds_from_support(
        self,
        artifact: ResidualMpcArtifact,
        phase_a_dampers: Sequence[float],
        context: ControllerContext,
    ) -> Tuple[Tuple[Tuple[float, float], ...], float, float]:
        active_bounds = self._support_trust_region_bounds(artifact)
        scale = clamp(_safe_float(self.config.trust_support_scale, 1.0), 0.0, 1.0)
        if not active_bounds:
            return (), 1.0, 0.0
        active_bounds = self._tighten_active_bounds_for_first_wheel_command(
            artifact,
            active_bounds,
            phase_a_dampers,
            context)
        if not active_bounds:
            return (), 1.0, 0.0
        return tuple(active_bounds), 0.0, scale

    def _support_trust_region_bounds(
        self,
        artifact: ResidualMpcArtifact,
    ) -> Tuple[Tuple[float, float], ...]:
        support = dict(artifact.manifest.get("training_support") or {})
        scale = clamp(_safe_float(self.config.trust_support_scale, 1.0), 0.0, 1.0)
        active_bounds: List[Tuple[float, float]] = []
        for mode in artifact.active_modes:
            row = dict(support.get(mode) or {})
            lo = _safe_float(row.get("min", 0.0), 0.0) * scale
            hi = _safe_float(row.get("max", 0.0), 0.0) * scale
            if abs(lo) < self.config.min_support_bound and abs(hi) < self.config.min_support_bound:
                return ()
            active_bounds.append((min(lo, hi), max(lo, hi)))
        return tuple(active_bounds)

    def _tighten_active_bounds_for_first_wheel_command(
        self,
        artifact: ResidualMpcArtifact,
        bounds: Sequence[Tuple[float, float]],
        phase_a_dampers: Sequence[float],
        context: ControllerContext,
    ) -> Tuple[Tuple[float, float], ...]:
        low = _safe_float(self.config.min_damper_scale, 0.75)
        high = _safe_float(self.config.max_damper_scale, 1.25)
        dt = _context_dt(context, self.config.default_dt)
        max_up = max(0.0, _safe_float(self.config.max_rate_up_scale_per_s, 1.2)) * dt
        max_down = max(0.0, _safe_float(self.config.max_rate_down_scale_per_s, 1.2)) * dt
        previous = _match_length(self.previous_final_damper_scales, len(phase_a_dampers), 1.0)
        wheel_low = tuple(max(low, previous[index] - max_down)
                          for index in range(len(phase_a_dampers)))
        wheel_high = tuple(min(high, previous[index] + max_up)
                           for index in range(len(phase_a_dampers)))
        tightened = []
        for active_index, mode in enumerate(artifact.active_modes):
            column = _active_mode_wheel_column(mode)
            lo, hi = bounds[active_index]
            for wheel_index, coeff in enumerate(column[:len(phase_a_dampers)]):
                if abs(coeff) <= 1.0e-12:
                    continue
                residual_low = wheel_low[wheel_index] - phase_a_dampers[wheel_index]
                residual_high = wheel_high[wheel_index] - phase_a_dampers[wheel_index]
                c_lo = residual_low / coeff
                c_hi = residual_high / coeff
                lo = max(lo, min(c_lo, c_hi))
                hi = min(hi, max(c_lo, c_hi))
            if lo > hi:
                return ()
            tightened.append((lo, hi))
        return tuple(tightened)

    def _build_qp_problem(
        self,
        *,
        artifact: ResidualMpcArtifact,
        context: ControllerContext,
        q_current: Sequence[float],
        support_bounds: Sequence[Tuple[float, float]],
        phase_a_dampers: Sequence[float],
        model_predict_elapsed_ns: Optional[List[int]] = None,
    ) -> Tuple[DenseQpProblem, Dict[str, float]]:
        active_dim = len(artifact.active_modes)
        horizon = int(_safe_float(
            artifact.manifest.get("recommended_horizon_steps", 10),
            10))
        horizon = max(1, min(48, horizon))
        block_steps = int(_safe_float(
            artifact.manifest.get("recommended_control_block_steps", 2),
            2))
        block_steps = max(1, block_steps)
        control_blocks = max(1, int(math.ceil(float(horizon) / block_steps)))
        control_blocks = min(control_blocks, max(1, int(self.config.max_control_blocks)))
        dim = active_dim * control_blocks
        lower: List[float] = []
        upper: List[float] = []
        for _ in range(control_blocks):
            for lo, hi in support_bounds:
                lower.append(lo)
                upper.append(hi)
        support_only_bounds = self._support_trust_region_bounds(artifact)
        support_only_lower: List[float] = []
        support_only_upper: List[float] = []
        for _ in range(control_blocks):
            for lo, hi in support_only_bounds:
                support_only_lower.append(lo)
                support_only_upper.append(hi)
        decision_indices = tuple(range(dim))
        constraint_groups = (
            QpConstraintGroup(
                name="support_trust_region",
                kind="support_trust_region",
                indices=decision_indices,
                lower=tuple(support_only_lower),
                upper=tuple(support_only_upper),
                metadata={
                    "active_modes": tuple(artifact.active_modes),
                    "control_blocks": control_blocks,
                    "trust_support_scale": _safe_float(
                        self.config.trust_support_scale,
                        1.0),
                }),
            QpConstraintGroup(
                name="first_action_rate_absolute_damper_tightening",
                kind="first_action_rate_absolute_damper",
                indices=decision_indices,
                lower=tuple(lower),
                upper=tuple(upper),
                metadata={
                    "active_modes": tuple(artifact.active_modes),
                    "control_blocks": control_blocks,
                    "derived_from_first_action": True,
                    "repeated_across_control_blocks": True,
                }),
        )
        base = self._simulate_flat_outputs(
            artifact,
            horizon=horizon,
            block_steps=block_steps,
            decision=tuple(0.0 for _ in range(dim)),
            q_current=q_current,
            model_predict_elapsed_ns=model_predict_elapsed_ns)
        epsilon = 1.0e-4
        columns: List[Tuple[float, ...]] = []
        for col in range(dim):
            perturb = [0.0 for _ in range(dim)]
            perturb[col] = epsilon
            shifted = self._simulate_flat_outputs(
                artifact,
                horizon=horizon,
                block_steps=block_steps,
                decision=tuple(perturb),
                q_current=q_current,
                model_predict_elapsed_ns=model_predict_elapsed_ns)
            columns.append(tuple(
                (shifted[row] - base[row]) / epsilon
                for row in range(len(base))))
        weights = _prediction_weights(
            artifact.model.output_channels,
            horizon,
            output_weight=self.config.output_tracking_weight,
            body_weight=self.config.body_motion_weight,
            terminal_weight=self.config.terminal_output_weight)
        hessian = [[0.0 for _ in range(dim)] for _ in range(dim)]
        linear = [0.0 for _ in range(dim)]
        for row, base_value in enumerate(base):
            weight = weights[row]
            for col in range(dim):
                linear[col] += 2.0 * weight * columns[col][row] * base_value
                for other in range(dim):
                    hessian[col][other] += (
                        2.0 * weight * columns[col][row] * columns[other][row])
        for col in range(dim):
            hessian[col][col] += 2.0 * max(self.config.residual_effort_weight, 0.0)
        for block in range(control_blocks):
            for active in range(active_dim):
                col = block * active_dim + active
                hessian[col][col] += 2.0 * max(self.config.residual_slew_weight, 0.0)
                if block > 0:
                    prev = (block - 1) * active_dim + active
                    weight = 2.0 * max(self.config.residual_slew_weight, 0.0)
                    hessian[col][prev] -= weight
                    hessian[prev][col] -= weight
                    hessian[prev][prev] += weight
        for col in range(dim):
            hessian[col][col] += 1.0e-9
        cost_info = {
            "baseline_predicted_comfort_cost": _weighted_norm(base, weights),
            "baseline_predicted_body_cost": _weighted_norm(base, weights),
            "baseline_predicted_safety_cost": 0.0,
            "horizon_steps": float(horizon),
            "control_block_steps": float(block_steps),
            "control_blocks": float(control_blocks),
        }
        return (
            DenseQpProblem(
                hessian=tuple(tuple(row) for row in hessian),
                linear=tuple(linear),
                lower=tuple(lower),
                upper=tuple(upper),
                metadata={
                    "horizon_steps": horizon,
                    "control_block_steps": block_steps,
                    "control_blocks": control_blocks,
                    "active_dim": active_dim,
                    "constraint_group_names": tuple(
                        group.name for group in constraint_groups),
                },
                constraint_groups=constraint_groups),
            cost_info)

    def _simulate_flat_outputs(
        self,
        artifact: ResidualMpcArtifact,
        *,
        horizon: int,
        block_steps: int,
        decision: Sequence[float],
        q_current: Sequence[float],
        model_predict_elapsed_ns: Optional[List[int]] = None,
    ) -> Tuple[float, ...]:
        predict_start_ns = time.perf_counter_ns()
        try:
            model = artifact.model
            active_dim = len(artifact.active_modes)
            y_history = [tuple(row) for row in self._y_history]
            u_history = [tuple(row) for row in self._u_history]
            phase_history = [tuple(row) for row in self._phase_history]
            flat: List[float] = []
            for step in range(horizon):
                block = min(step // max(block_steps, 1),
                            max((len(decision) // max(active_dim, 1)) - 1, 0))
                active_values = decision[
                    block * active_dim:(block + 1) * active_dim]
                full_modal = _active_to_candidate_modal(
                    artifact,
                    active_values,
                    model.input_modes)
                u_history.append(tuple(full_modal))
                phase_history.append(phase_history[-1])
                prediction = model.predict_next(
                    y_history,
                    u_history,
                    phase_history,
                    q_current)
                if not _all_finite(prediction):
                    raise ValueError("nonfinite_prediction")
                y_history.append(prediction)
                flat.extend(prediction)
            return tuple(flat)
        finally:
            if model_predict_elapsed_ns is not None:
                model_predict_elapsed_ns[0] += max(
                    0,
                    time.perf_counter_ns() - predict_start_ns)

    def _diagnostics(
        self,
        *,
        context: ControllerContext,
        artifact: Optional[ResidualMpcArtifact],
        phase_a_shadow_diagnostics: Mapping[str, Any],
        phase_a_dampers: Sequence[float],
        phase_a_modal: Sequence[float],
        requested_modal: Sequence[float],
        final_modal: Sequence[float],
        requested_residual: Sequence[float],
        projected_residual: Sequence[float],
        final_residual: Sequence[float],
        qp_first_dampers: Sequence[float],
        raw_qp_first_dampers: Sequence[float],
        first_action_trace: FirstActionProjectionTrace,
        final_dampers: Sequence[float],
        mode: str,
        fallback_reason: str,
        solver_result: SolverResult,
        re_solve_count: int,
        re_solve_attempted: int,
        re_solve_succeeded: int,
        re_solve_failed: int,
        first_action_correction_active: int,
        support_distance: float,
        support_scale: float,
        history_ready: int,
        cost_info: Mapping[str, float],
        bounds_guard_active: bool,
        projection_active: bool,
        final_guard_active: bool,
        phase_a_commit_match: int,
        compute_ms: float,
        controller_end_to_end_ms: float,
        incremental_compute_ms: float,
        budget_metric_name: str,
        budget_metric_ms: float,
        budget_ms: float,
        compute_budget_exceeded: bool,
        stage_timing_ms: Mapping[str, float],
        initial_warm_start_input: Sequence[float],
        initial_warm_start_provenance: str,
        first_action_resolve_warm_start_input: Sequence[float],
        first_action_resolve_warm_start_provenance: str,
        lifecycle: ResidualLifecycleTrace,
    ) -> Dict[str, Any]:
        planning = getattr(context, "planning", None)
        planning_age_frames = _planning_age_frames(context)
        planning_age_s = (
            planning_age_frames * _safe_float(
                getattr(planning, "horizon_dt", 0.1),
                0.1)
            if planning is not None and planning_age_frames is not None else "")
        model = artifact.model if artifact is not None else None
        manifest = artifact.manifest if artifact is not None else {}
        optimized_cost = _safe_float(solver_result.objective, 0.0)
        baseline_cost = _safe_float(
            cost_info.get("baseline_predicted_comfort_cost", 0.0),
            0.0)
        gain_abs = baseline_cost - optimized_cost
        gain_rel = gain_abs / max(abs(baseline_cost), 1.0e-12)
        diagnostics: Dict[str, Any] = {
            "controller": self.name,
            "controller_name": self.name,
            "controller_version": self.config.controller_version,
            "residual_qp_mpc_shadow_config_path": str(
                getattr(self.config, "config_path", "") or ""),
            "residual_qp_mpc_shadow_config_exists": (
                self._config_exists_for_diagnostics()),
            "residual_mpc_schema_version": self.config.telemetry_schema_version,
            "residual_mpc_controller_version": self.config.controller_version,
            "residual_mpc_enabled": 1,
            "residual_mpc_shadow_mode": int(bool(self.config.shadow_mode)),
            "residual_mpc_mode": mode,
            "residual_mpc_fallback_active": int(bool(fallback_reason)),
            "residual_mpc_fallback_reason": fallback_reason or "none",
            "residual_mpc_model_artifact_dir": str(
                getattr(self.config, "model_artifact_dir", "") or ""),
            "residual_mpc_model_artifact_dir_exists": (
                self._model_artifact_dir_exists_for_diagnostics()),
            "residual_mpc_model_artifact_loaded": int(artifact is not None),
            "residual_mpc_model_id": str(manifest.get("model_id", "")),
            "residual_mpc_model_family": str(manifest.get("model_family", "")),
            "residual_mpc_model_validation_status": str(
                manifest.get("validation_status", "")),
            "residual_mpc_model_hash_status": (
                "ok" if artifact is not None else self._artifact_error),
            "residual_mpc_active_modes": ",".join(artifact.active_modes)
            if artifact is not None else "",
            "residual_mpc_output_channels": ",".join(
                model.output_channels if model is not None else ()),
            "residual_mpc_exogenous_channels": ",".join(
                model.exogenous_channels if model is not None else ()),
            "residual_mpc_input_delay_ticks": (
                model.input_delay_ticks if model is not None else 0),
            "residual_mpc_output_lags": (
                model.orders.output_lags if model is not None else 0),
            "residual_mpc_input_lags": (
                model.orders.input_lags if model is not None else 0),
            "residual_mpc_horizon_steps": _safe_float(
                cost_info.get("horizon_steps", manifest.get(
                    "recommended_horizon_steps", 0)),
                0.0),
            "residual_mpc_control_block_steps": _safe_float(
                cost_info.get("control_block_steps", manifest.get(
                    "recommended_control_block_steps", 0)),
                0.0),
            "residual_mpc_control_blocks": _safe_float(
                cost_info.get("control_blocks", 0.0),
                0.0),
            "residual_mpc_history_ready": int(history_ready),
            "residual_mpc_history_samples": len(self._y_history),
            "residual_mpc_support_distance": support_distance,
            "residual_mpc_support_scale": support_scale,
            "residual_mpc_operating_mode": _operating_mode(context),
            "residual_mpc_weight_calm": 1.0,
            "residual_mpc_weight_lateral": 0.0,
            "residual_mpc_weight_braking": 0.0,
            "residual_mpc_solver_backend": self._solver_name_for_diagnostics(),
            "residual_mpc_solver_name": self._solver_name_for_diagnostics(),
            "residual_mpc_solver_status": solver_result.status,
            "residual_mpc_solver_iterations": solver_result.iterations,
            "residual_mpc_solver_setup_ms": solver_result.setup_ms,
            "residual_mpc_solver_solve_ms": solver_result.solve_ms,
            "residual_mpc_solver_total_ms": solver_result.total_ms,
            "residual_mpc_solver_primal_residual": solver_result.primal_residual,
            "residual_mpc_solver_dual_residual": solver_result.dual_residual,
            "residual_mpc_solver_primal_residual_available": int(
                solver_result.primal_residual_available),
            "residual_mpc_solver_dual_residual_available": int(
                solver_result.dual_residual_available),
            "residual_mpc_solver_primal_residual_provenance": (
                solver_result.primal_residual_provenance),
            "residual_mpc_solver_dual_residual_provenance": (
                solver_result.dual_residual_provenance),
            # Kept for CSV compatibility.  This intentionally means the
            # initial QP *input* only; one boolean cannot express a later
            # corrective solve, for which the stage-specific fields below are
            # authoritative.  In particular this never reads controller state
            # that may have been updated or cleared after a solver outcome.
            "residual_mpc_solver_warm_start_used": int(
                bool(initial_warm_start_input)),
            "residual_mpc_initial_qp_warm_start_used": int(
                bool(initial_warm_start_input)),
            "residual_mpc_initial_qp_warm_start_dimension": len(
                tuple(initial_warm_start_input)),
            "residual_mpc_initial_qp_warm_start_provenance": (
                initial_warm_start_provenance),
            "residual_mpc_first_action_resolve_warm_start_used": int(
                bool(first_action_resolve_warm_start_input)),
            "residual_mpc_first_action_resolve_warm_start_dimension": len(
                tuple(first_action_resolve_warm_start_input)),
            "residual_mpc_first_action_resolve_warm_start_provenance": (
                first_action_resolve_warm_start_provenance),
            "residual_mpc_qp_objective": solver_result.objective,
            "residual_mpc_re_solve_count": re_solve_count,
            "residual_mpc_re_solve_attempted": re_solve_attempted,
            "residual_mpc_re_solve_succeeded": re_solve_succeeded,
            "residual_mpc_re_solve_failed": re_solve_failed,
            "residual_mpc_first_action_correction_active": (
                first_action_correction_active),
            "residual_mpc_compute_ms": compute_ms,
            "residual_mpc_controller_end_to_end_ms": controller_end_to_end_ms,
            "residual_mpc_incremental_compute_ms": incremental_compute_ms,
            "residual_mpc_budget_metric_name": budget_metric_name,
            "residual_mpc_budget_metric_ms": budget_metric_ms,
            "residual_mpc_budget_ms": budget_ms,
            "residual_mpc_stage_phase_a_shadow_ms": _stage_timing_value(
                stage_timing_ms,
                "phase_a_shadow"),
            "residual_mpc_stage_phase_a_commit_ms": _stage_timing_value(
                stage_timing_ms,
                "phase_a_commit"),
            "residual_mpc_stage_feature_build_ms": _stage_timing_value(
                stage_timing_ms,
                "feature_build"),
            "residual_mpc_stage_model_predict_ms": _stage_timing_value(
                stage_timing_ms,
                "model_predict"),
            "residual_mpc_stage_qp_matrix_build_ms": _stage_timing_value(
                stage_timing_ms,
                "qp_matrix_build"),
            "residual_mpc_stage_qp_setup_or_update_ms": _stage_timing_value(
                stage_timing_ms,
                "qp_setup_or_update"),
            "residual_mpc_stage_qp_solve_wall_ms": _stage_timing_value(
                stage_timing_ms,
                "qp_solve_wall"),
            "residual_mpc_stage_qp_postprocess_ms": _stage_timing_value(
                stage_timing_ms,
                "qp_postprocess"),
            "residual_mpc_stage_safety_projection_ms": _stage_timing_value(
                stage_timing_ms,
                "safety_projection"),
            "residual_mpc_compute_budget_exceeded": int(
                compute_budget_exceeded),
            "residual_mpc_baseline_predicted_comfort_cost": baseline_cost,
            "residual_mpc_optimized_predicted_comfort_cost": optimized_cost,
            "residual_mpc_predicted_cost_gain_abs": gain_abs,
            "residual_mpc_predicted_cost_gain_rel": gain_rel,
            "residual_mpc_baseline_predicted_body_cost": _safe_float(
                cost_info.get("baseline_predicted_body_cost", 0.0),
                0.0),
            "residual_mpc_optimized_predicted_body_cost": optimized_cost,
            "residual_mpc_baseline_predicted_safety_cost": _safe_float(
                cost_info.get("baseline_predicted_safety_cost", 0.0),
                0.0),
            "residual_mpc_optimized_predicted_safety_cost": 0.0,
            "residual_mpc_max_slack": 0.0,
            "residual_mpc_first_action_qp_to_applied_l2": _mismatch(
                qp_first_dampers,
                final_dampers)[0],
            "residual_mpc_first_action_qp_to_applied_linf": _mismatch(
                qp_first_dampers,
                final_dampers)[1],
            "residual_mpc_first_action_pre_correction_l2": (
                first_action_trace.pre_correction_l2),
            "residual_mpc_first_action_pre_correction_linf": (
                first_action_trace.pre_correction_linf),
            "residual_mpc_first_action_raw_to_projected_l2": (
                first_action_trace.raw_to_projected_l2),
            "residual_mpc_first_action_raw_to_projected_linf": (
                first_action_trace.raw_to_projected_linf),
            "residual_mpc_first_action_projected_to_final_l2": (
                first_action_trace.projected_to_final_l2),
            "residual_mpc_first_action_projected_to_final_linf": (
                first_action_trace.projected_to_final_linf),
            "residual_mpc_first_action_final_to_reprojected_l2": (
                first_action_trace.final_to_reprojected_l2),
            "residual_mpc_first_action_final_to_reprojected_linf": (
                first_action_trace.final_to_reprojected_linf),
            "residual_mpc_first_action_fixed_point_linf": (
                first_action_trace.fixed_point_linf),
            "residual_mpc_first_action_fixed_point_iterations": (
                first_action_trace.fixed_point_iterations),
            "residual_mpc_first_action_minmax_clip_count": (
                first_action_trace.minmax_clip_count),
            "residual_mpc_first_action_rate_clip_count": (
                first_action_trace.rate_clip_count),
            "residual_mpc_first_action_low_rel_blend_count": (
                first_action_trace.low_rel_blend_count),
            "residual_mpc_first_action_infeasible_blend_count": (
                first_action_trace.infeasible_blend_count),
            "residual_mpc_first_action_modal_coupling_bound_miss_count": (
                first_action_trace.modal_coupling_bound_miss_count),
            "residual_mpc_first_action_fixed_point_out_of_support": (
                first_action_trace.fixed_point_out_of_support),
            "residual_mpc_bounds_guard_active": int(bounds_guard_active),
            "residual_mpc_projection_active": int(projection_active),
            "residual_mpc_final_guard_active": int(final_guard_active),
            "residual_mpc_phase_a_commit_match": phase_a_commit_match,
            "residual_mpc_measured_dt_s": _context_dt(
                context,
                self.config.default_dt),
            "residual_mpc_planning_age_frames": (
                planning_age_frames if planning_age_frames is not None else ""),
            "residual_mpc_planning_age_s": planning_age_s,
            "residual_mpc_preview_validity_reason": _preview_validity_reason(context),
            "residual_mpc_strict_step07_shadow_validation": int(
                bool(getattr(
                    self.config,
                    "strict_step07_shadow_validation",
                    False))),
            "residual_mpc_spring_scale_invariant": 1.0,
        }
        diagnostics.update(_fixed_point_support_comparison_diagnostics(
            first_action_trace.fixed_point_support_comparison))
        diagnostics.update(_residual_lifecycle_diagnostics(lifecycle))
        diagnostics.update(dict(phase_a_shadow_diagnostics or {}))
        _modal_diagnostics(diagnostics, "residual_mpc_phase_a_modal", phase_a_modal)
        _modal_diagnostics(
            diagnostics,
            "residual_mpc_requested_residual_modal",
            requested_modal)
        _modal_diagnostics(
            diagnostics,
            "residual_mpc_final_residual_modal",
            final_modal)
        _wheel_diagnostics(
            diagnostics,
            "residual_mpc_phase_a_shadow_damper",
            phase_a_dampers)
        _wheel_diagnostics(
            diagnostics,
            "residual_mpc_raw_qp_first_damper",
            raw_qp_first_dampers)
        _wheel_diagnostics(
            diagnostics,
            "residual_mpc_qp_first_damper",
            qp_first_dampers)
        _wheel_diagnostics(
            diagnostics,
            "residual_mpc_semi_active_first_damper",
            first_action_trace.semi_active_dampers)
        _wheel_diagnostics(
            diagnostics,
            "residual_mpc_safety_first_damper",
            first_action_trace.safety_dampers)
        _wheel_diagnostics(
            diagnostics,
            "residual_mpc_reprojected_first_damper",
            first_action_trace.reprojected_dampers)
        _wheel_diagnostics(
            diagnostics,
            "residual_mpc_fixed_point_first_damper",
            first_action_trace.fixed_point_dampers)
        _wheel_diagnostics(
            diagnostics,
            "residual_mpc_requested_residual",
            requested_residual)
        _wheel_diagnostics(
            diagnostics,
            "residual_mpc_projected_residual",
            projected_residual)
        _wheel_diagnostics(
            diagnostics,
            "residual_mpc_final_residual",
            final_residual)
        _wheel_diagnostics(
            diagnostics,
            "residual_mpc_final_damper",
            final_dampers)
        _ensure_residual_qp_diagnostics(diagnostics)
        return diagnostics


class PlanningAwareV5ResidualQpMpcShadowController(
        PlanningAwareV5ResidualQpMpcController):
    name = "planning_aware_v5_residual_qp_mpc_shadow"

    def __init__(
        self,
        config: Optional[PlanningAwareV5ResidualQpMpcConfig] = None,
    ):
        cfg = config or PlanningAwareV5ResidualQpMpcConfig(shadow_mode=True)
        if not bool(cfg.shadow_mode):
            values = {field.name: getattr(cfg, field.name) for field in fields(cfg)}
            values["shadow_mode"] = True
            cfg = PlanningAwareV5ResidualQpMpcConfig(**values)
        super().__init__(cfg)


@dataclass
class _LowPassState:
    tau_s: float
    value: Optional[float] = None

    def update(self, raw_value: float, dt_s: float) -> float:
        raw = float(raw_value)
        if self.value is None:
            self.value = raw
            return raw
        tau = max(float(self.tau_s), 0.0)
        alpha = 1.0 if tau <= 1.0e-12 else clamp(
            float(dt_s) / (tau + float(dt_s)),
            0.0,
            1.0)
        self.value = self.value + alpha * (raw - self.value)
        return self.value


def _make_solver(
    config: PlanningAwareV5ResidualQpMpcConfig,
) -> ResidualQpSolverAdapter:
    name = _solver_name(config)
    if name == "box_projected_gradient":
        return ProjectedGradientQpSolverAdapter(
            max_iterations=config.solver_max_iterations,
            tolerance=config.solver_tolerance)
    if name == "osqp":
        return OsqpSolverAdapter(
            max_iterations=config.solver_max_iterations,
            tolerance=config.solver_tolerance)
    if name == "auto":
        osqp_adapter = OsqpSolverAdapter(
            max_iterations=config.solver_max_iterations,
            tolerance=config.solver_tolerance)
        probe = osqp_adapter.solve(_tiny_qp_problem())
        if _solver_preflight_solve_ok(probe):
            return osqp_adapter
        detail = probe.error or probe.status or "unknown"
        return UnavailableSolverAdapter("osqp_unavailable:%s" % detail)
    return UnavailableSolverAdapter("unknown_solver:%s" % name)


def _solver_name(config: PlanningAwareV5ResidualQpMpcConfig) -> str:
    return str(config.solver_name or "auto")


def run_residual_qp_solver_preflight(
    *,
    solver_name: str = "osqp",
    import_module: Callable[[str], Any] = importlib.import_module,
    adapter_factory: Optional[
        Callable[[Any, Any], ResidualQpSolverAdapter]] = None,
) -> SolverPreflightResult:
    start_ns = time.perf_counter_ns()
    package_versions: Dict[str, str] = {}
    package_paths: Dict[str, str] = {}

    def elapsed_ms() -> float:
        return max(0.0, (time.perf_counter_ns() - start_ns) / 1.0e6)

    if solver_name != "osqp":
        return SolverPreflightResult(
            ok=False,
            solver_name=solver_name,
            status="unsupported_solver",
            package_versions=package_versions,
            package_paths=package_paths,
            total_ms=elapsed_ms(),
            error="unsupported_solver:%s" % solver_name)

    modules: Dict[str, Any] = {}
    try:
        for package_name in ("numpy", "scipy", "osqp"):
            module = import_module(package_name)
            modules[package_name] = module
            package_versions[package_name] = str(
                getattr(module, "__version__", ""))
            package_paths[package_name] = str(getattr(module, "__file__", ""))
        sparse = import_module("scipy.sparse")
    except Exception as exc:  # noqa: BLE001 - preflight must report import cause.
        return SolverPreflightResult(
            ok=False,
            solver_name="osqp",
            status="import_error",
            package_versions=package_versions,
            package_paths=package_paths,
            total_ms=elapsed_ms(),
            error="%s:%s" % (exc.__class__.__name__, exc))

    try:
        if adapter_factory is not None:
            adapter = adapter_factory(modules["osqp"], sparse)
        else:
            adapter = OsqpSolverAdapter(
                numpy_module=modules["numpy"],
                osqp_module=modules["osqp"],
                sparse_module=sparse)
        solver_result = adapter.solve(_tiny_qp_problem())
    except Exception as exc:  # noqa: BLE001 - preflight must hard-fail cleanly.
        return SolverPreflightResult(
            ok=False,
            solver_name="osqp",
            status="solver_error",
            package_versions=package_versions,
            package_paths=package_paths,
            total_ms=elapsed_ms(),
            error="%s:%s" % (exc.__class__.__name__, exc))

    solution = tuple(float(value) for value in solver_result.primal_solution)
    objective = _safe_float(solver_result.objective, math.nan)
    finite_solution = bool(solution) and _all_finite(solution)
    finite_objective = math.isfinite(objective)
    ok = (
        _solver_preflight_solve_ok(solver_result) and
        finite_solution and
        finite_objective)
    error = ""
    if not _solver_preflight_solve_ok(solver_result):
        error = solver_result.error or "tiny_qp_status:%s" % solver_result.status
    elif not finite_solution:
        error = "tiny_qp_nonfinite_solution"
    elif not finite_objective:
        error = "tiny_qp_nonfinite_objective"
    return SolverPreflightResult(
        ok=ok,
        solver_name="osqp",
        status=solver_result.status,
        package_versions=package_versions,
        package_paths=package_paths,
        solution=solution,
        objective=objective,
        setup_ms=max(0.0, _safe_float(solver_result.setup_ms, 0.0)),
        solve_ms=max(0.0, _safe_float(solver_result.solve_ms, 0.0)),
        total_ms=max(0.0, _safe_float(solver_result.total_ms, elapsed_ms())),
        iterations=solver_result.iterations,
        error=error)


def require_residual_qp_solver_preflight(
    *,
    solver_name: str = "osqp",
) -> SolverPreflightResult:
    result = run_residual_qp_solver_preflight(solver_name=solver_name)
    if not result.ok:
        raise RuntimeError(
            "residual QP solver preflight failed: %s status=%s error=%s" % (
                result.solver_name,
                result.status,
                result.error))
    return result


def _tiny_qp_problem() -> DenseQpProblem:
    return DenseQpProblem(
        hessian=((2.0,),),
        linear=(-2.0,),
        lower=(0.0,),
        upper=(2.0,),
        metadata={"purpose": "residual_qp_solver_preflight_tiny_qp"},
        constraint_groups=(QpConstraintGroup(
            name="solver_preflight_box",
            kind="support_trust_region",
            indices=(0,),
            lower=(0.0,),
            upper=(2.0,),
            metadata={"purpose": "solver_preflight"}),))


def _solver_preflight_solve_ok(result: SolverResult) -> bool:
    status = str(result.status or "").strip().lower().replace("_", " ")
    return status == "solved" or status == "solved inaccurate"


def _qp_objective(
    hessian: Sequence[Sequence[float]],
    linear: Sequence[float],
    x: Sequence[float],
) -> float:
    quad = 0.0
    for row in range(len(x)):
        for col in range(len(x)):
            quad += 0.5 * x[row] * hessian[row][col] * x[col]
    return quad + sum(linear[index] * x[index] for index in range(len(x)))


def check_zero_vector_feasibility(
    problem: DenseQpProblem,
) -> ZeroFeasibilityResult:
    """Pure zero-vector check for the current identity-constrained box QP."""

    dim = len(problem.linear)
    if len(problem.lower) != dim or len(problem.upper) != dim:
        return ZeroFeasibilityResult(
            feasible=False,
            max_violation=1.0,
            worst_group="problem_dimension",
            worst_lower=float(len(problem.lower)),
            worst_value=float(dim),
            worst_upper=float(len(problem.upper)))

    max_violation = 0.0
    worst_group = "none"
    worst_lower = 0.0
    worst_upper = 0.0
    invalid = False

    def consider(group_name: str, lower: float, upper: float) -> None:
        nonlocal max_violation, worst_group, worst_lower, worst_upper, invalid
        lo = _optional_finite_float(lower)
        hi = _optional_finite_float(upper)
        if lo is None or hi is None:
            invalid = True
            if max_violation < 1.0:
                max_violation = 1.0
                worst_group = group_name
                worst_lower = 0.0 if lo is None else lo
                worst_upper = 0.0 if hi is None else hi
            return
        violation = max(0.0, lo, -hi)
        if violation > max_violation + 1.0e-15:
            max_violation = violation
            worst_group = group_name
            worst_lower = lo
            worst_upper = hi

    for group in problem.constraint_groups:
        if not (
                len(group.indices) == len(group.lower) == len(group.upper)):
            invalid = True
            consider("%s:invalid_group_dimension" % group.name, 1.0, -1.0)
            continue
        for index, lo, hi in zip(group.indices, group.lower, group.upper):
            if index < 0 or index >= dim:
                invalid = True
                consider("%s:invalid_index" % group.name, 1.0, -1.0)
                continue
            consider(group.name, lo, hi)

    for lo, hi in zip(problem.lower, problem.upper):
        consider("box_bounds_unattributed", lo, hi)

    return ZeroFeasibilityResult(
        feasible=not invalid and max_violation <= 1.0e-12,
        max_violation=max_violation,
        worst_group=worst_group,
        worst_lower=worst_lower,
        worst_value=0.0,
        worst_upper=worst_upper)


def _zero_feasibility_not_checked(reason: str) -> ZeroFeasibilityResult:
    return ZeroFeasibilityResult(
        feasible=False,
        max_violation=0.0,
        worst_group=reason,
        worst_lower=0.0,
        worst_value=0.0,
        worst_upper=0.0)


def _residual_linf(values: Sequence[float]) -> float:
    finite = tuple(abs(_safe_float(value, 0.0)) for value in values)
    return max(finite) if finite else 0.0


def _residual_nonzero(values: Sequence[float], tolerance: float = 1.0e-12) -> int:
    return int(_residual_linf(values) > max(0.0, tolerance))


def _optional_info_float(info: Any, name: str) -> Optional[float]:
    if info is None or not hasattr(info, name):
        return None
    return _optional_finite_float(getattr(info, name))


def _optional_info_first_float(
    info: Any,
    names: Sequence[str],
) -> Tuple[Optional[float], str]:
    """Return the first finite native OSQP field and its exact alias.

    OSQP 1.x exposes ``prim_res``/``dual_res`` while OSQP 0.6.x exposes
    ``pri_res``/``dua_res``.  The alias is retained so CSV/snapshot consumers
    can distinguish a measured zero from a compatibility default.
    """
    for name in names:
        value = _optional_info_float(info, name)
        if value is not None:
            return value, name
    return None, ""


def _optional_info_int(info: Any, name: str) -> Optional[int]:
    value = _optional_info_float(info, name)
    return None if value is None else int(value)


def _optional_info_ms(info: Any, name: str) -> Optional[float]:
    value = _optional_info_float(info, name)
    return None if value is None else max(0.0, value * 1000.0)


def _optional_diagnostic(value: Any) -> Any:
    if value is None:
        return ""
    return value


def _compact_float_vector(values: Sequence[float]) -> str:
    return ",".join("%.17g" % float(value) for value in values)


def _fixed_point_support_comparison_diagnostics(
    comparison: Optional[FixedPointSupportComparison],
) -> Dict[str, Any]:
    if comparison is None:
        return {
            "residual_mpc_first_action_support_comparison_evaluable": 0,
            "residual_mpc_first_action_support_comparison_error": "not_run",
            "residual_mpc_first_action_active_mode_order": "",
            "residual_mpc_first_action_fixed_point_active_target": "",
            "residual_mpc_first_action_existing_lower": "",
            "residual_mpc_first_action_existing_upper": "",
            "residual_mpc_first_action_lower_signed_margins": "",
            "residual_mpc_first_action_upper_signed_margins": "",
            "residual_mpc_first_action_support_violation_count": 0,
            "residual_mpc_first_action_support_violation_linf": 0.0,
            "residual_mpc_first_action_support_worst_mode": "",
            "residual_mpc_first_action_support_worst_index": -1,
            "residual_mpc_first_action_support_worst_side": "",
            "residual_mpc_first_action_support_worst_target": 0.0,
            "residual_mpc_first_action_support_worst_lower": 0.0,
            "residual_mpc_first_action_support_worst_upper": 0.0,
            "residual_mpc_first_action_support_worst_signed_gap": 0.0,
            "residual_mpc_first_action_support_worst_absolute_gap": 0.0,
            "residual_mpc_first_action_support_worst_normalized_gap": 0.0,
            "residual_mpc_first_action_support_binding_constraint_group": "",
            "residual_mpc_first_action_support_conversion_provenance": "",
            "residual_mpc_first_action_support_comparison_tolerance": 0.0,
            "residual_mpc_first_action_corrective_qp_constructed": 0,
            "residual_mpc_first_action_corrective_qp_rejected": 0,
        }
    return {
        "residual_mpc_first_action_support_comparison_evaluable": int(
            comparison.evaluable),
        "residual_mpc_first_action_support_comparison_error": comparison.error,
        "residual_mpc_first_action_active_mode_order": ",".join(
            comparison.active_mode_order),
        "residual_mpc_first_action_fixed_point_active_target": (
            _compact_float_vector(comparison.active_target)),
        "residual_mpc_first_action_existing_lower": (
            _compact_float_vector(comparison.existing_lower)),
        "residual_mpc_first_action_existing_upper": (
            _compact_float_vector(comparison.existing_upper)),
        "residual_mpc_first_action_lower_signed_margins": (
            _compact_float_vector(comparison.lower_signed_margins)),
        "residual_mpc_first_action_upper_signed_margins": (
            _compact_float_vector(comparison.upper_signed_margins)),
        "residual_mpc_first_action_support_violation_count": (
            comparison.violation_count),
        "residual_mpc_first_action_support_violation_linf": (
            comparison.violation_linf),
        "residual_mpc_first_action_support_worst_mode": comparison.worst_mode,
        "residual_mpc_first_action_support_worst_index": comparison.worst_index,
        "residual_mpc_first_action_support_worst_side": comparison.worst_side,
        "residual_mpc_first_action_support_worst_target": comparison.worst_target,
        "residual_mpc_first_action_support_worst_lower": comparison.worst_lower,
        "residual_mpc_first_action_support_worst_upper": comparison.worst_upper,
        "residual_mpc_first_action_support_worst_signed_gap": (
            comparison.worst_signed_gap),
        "residual_mpc_first_action_support_worst_absolute_gap": (
            comparison.worst_absolute_gap),
        "residual_mpc_first_action_support_worst_normalized_gap": (
            comparison.worst_normalized_gap),
        "residual_mpc_first_action_support_binding_constraint_group": (
            comparison.binding_constraint_group),
        "residual_mpc_first_action_support_conversion_provenance": (
            comparison.conversion_provenance),
        "residual_mpc_first_action_support_comparison_tolerance": (
            comparison.comparison_tolerance),
        "residual_mpc_first_action_corrective_qp_constructed": int(
            comparison.corrective_qp_constructed),
        "residual_mpc_first_action_corrective_qp_rejected": int(
            comparison.corrective_qp_rejected),
    }


def _fixed_point_support_violation_to_dict(
    violation: FixedPointSupportViolation,
) -> Dict[str, Any]:
    return {
        "active_index": violation.active_index,
        "mode": violation.mode,
        "side": violation.side,
        "target": violation.target,
        "lower": violation.lower,
        "upper": violation.upper,
        "signed_gap": violation.signed_gap,
        "absolute_gap": violation.absolute_gap,
        "normalized_gap": violation.normalized_gap,
        "normalization_span": violation.normalization_span,
        "binding_constraint_group": violation.binding_constraint_group,
        "binding_constraint_group_candidates": list(
            violation.binding_constraint_group_candidates),
    }


def fixed_point_support_comparison_to_dict(
    comparison: Optional[FixedPointSupportComparison],
) -> Optional[Dict[str, Any]]:
    """Stable public serializer for bounded debug and offline diagnostics."""
    if comparison is None:
        return None
    return {
        "evaluable": comparison.evaluable,
        "error": comparison.error,
        "active_mode_order": list(comparison.active_mode_order),
        "wheel_residual": list(comparison.wheel_residual),
        "full_modal_target": list(comparison.full_modal_target),
        "active_target": list(comparison.active_target),
        "existing_lower": list(comparison.existing_lower),
        "existing_upper": list(comparison.existing_upper),
        "lower_signed_margins": list(comparison.lower_signed_margins),
        "upper_signed_margins": list(comparison.upper_signed_margins),
        "violations": [
            _fixed_point_support_violation_to_dict(row)
            for row in comparison.violations
        ],
        "violation_count": comparison.violation_count,
        "violation_linf": comparison.violation_linf,
        "worst_mode": comparison.worst_mode,
        "worst_index": comparison.worst_index,
        "worst_side": comparison.worst_side,
        "worst_target": comparison.worst_target,
        "worst_lower": comparison.worst_lower,
        "worst_upper": comparison.worst_upper,
        "worst_signed_gap": comparison.worst_signed_gap,
        "worst_absolute_gap": comparison.worst_absolute_gap,
        "worst_normalized_gap": comparison.worst_normalized_gap,
        "binding_constraint_group": comparison.binding_constraint_group,
        "conversion_provenance": comparison.conversion_provenance,
        "comparison_tolerance": comparison.comparison_tolerance,
        "corrective_qp_constructed": comparison.corrective_qp_constructed,
        "corrective_qp_rejected": comparison.corrective_qp_rejected,
    }


def _zero_feasibility_to_dict(
    result: ZeroFeasibilityResult,
) -> Dict[str, Any]:
    return {
        "feasible": bool(result.feasible),
        "max_violation": result.max_violation,
        "worst_group": result.worst_group,
        "worst_lower": result.worst_lower,
        "worst_value": result.worst_value,
        "worst_upper": result.worst_upper,
    }


def _constraint_group_to_dict(group: QpConstraintGroup) -> Dict[str, Any]:
    start = min(group.indices) if group.indices else 0
    stop = max(group.indices) + 1 if group.indices else 0
    return {
        "name": group.name,
        "kind": group.kind,
        "start": start,
        "stop": stop,
        "indices": list(group.indices),
        "lower": list(group.lower),
        "upper": list(group.upper),
        "metadata": dict(group.metadata),
    }


def _solver_result_snapshot(result: SolverResult) -> Dict[str, Any]:
    primal = tuple(result.primal_solution)
    primal_finite = bool(primal and _all_finite(primal))
    return {
        "status": result.status,
        "iterations": result.iterations,
        "objective": result.objective,
        "primal_residual": result.primal_residual,
        "dual_residual": result.dual_residual,
        "primal_residual_available": bool(result.primal_residual_available),
        "dual_residual_available": bool(result.dual_residual_available),
        "primal_residual_provenance": result.primal_residual_provenance,
        "dual_residual_provenance": result.dual_residual_provenance,
        "setup_ms": result.setup_ms,
        "solve_ms": result.solve_ms,
        "total_ms": result.total_ms,
        "primal_dimension": len(primal),
        "primal_finite": primal_finite,
        # Snapshot v1 retained only this summary.  Snapshot v2 preserves the
        # full immutable returned vector when it is finite, allowing a future
        # matching-runtime stage parity comparison without serializing broader
        # controller history/state.
        "primal_solution": list(primal) if primal_finite else None,
        "primal_linf": _residual_linf(primal),
        "primal_nonzero": bool(_residual_nonzero(primal)),
        "error": str(result.error or "")[:512],
        "osqp": {
            "status": result.osqp_status or None,
            "status_val": result.osqp_status_val,
            "iter": result.osqp_iter,
            "prim_res": result.osqp_prim_res,
            "dual_res": result.osqp_dual_res,
            "obj_val": result.osqp_obj_val,
            "rho_updates": result.osqp_rho_updates,
            "setup_time_ms": result.osqp_setup_time_ms,
            "solve_time_ms": result.osqp_solve_time_ms,
            "update_time_ms": result.osqp_update_time_ms,
        },
    }


def _debug_qp_stage_to_dict(stage: DebugQpStageRecord) -> Dict[str, Any]:
    problem = stage.problem
    hessian_rows: List[int] = []
    hessian_cols: List[int] = []
    hessian_data: List[float] = []
    for row, values in enumerate(problem.hessian):
        for col, value in enumerate(values):
            finite = _safe_float(value, 0.0)
            if abs(finite) <= 0.0:
                continue
            hessian_rows.append(row)
            hessian_cols.append(col)
            hessian_data.append(finite)
    return {
        "stage": stage.stage,
        "solve_ordinal": int(stage.solve_ordinal),
        "warm_start_input": {
            "provenance": stage.warm_start_provenance,
            "supplied": bool(stage.warm_start_input),
            "dimension": len(stage.warm_start_input),
            "finite": bool(
                not stage.warm_start_input or
                _all_finite(stage.warm_start_input)),
            "values": list(stage.warm_start_input),
        },
        "dimensions": {
            "variables": len(problem.linear),
            "hessian_rows": len(problem.hessian),
            "hessian_cols": max(
                (len(row) for row in problem.hessian),
                default=0),
        },
        "qp": {
            "hessian_coo": {
                "rows": hessian_rows,
                "cols": hessian_cols,
                "data": hessian_data,
            },
            "linear": list(problem.linear),
            "lower": list(problem.lower),
            "upper": list(problem.upper),
            "identity_box_constraints": True,
        },
        "constraint_groups": [
            _constraint_group_to_dict(group)
            for group in problem.constraint_groups
        ],
        "metadata": dict(problem.metadata),
        "zero_feasibility": _zero_feasibility_to_dict(
            stage.zero_feasibility),
        "solver_result": _solver_result_snapshot(stage.solver_result),
    }


def _debug_qp_snapshot_to_dict(
    record: DebugQpSnapshotRecord,
) -> Dict[str, Any]:
    return {
        "schema_version": "residual_mpc_debug_qp_snapshot_v2",
        "key": record.key,
        "frame": record.frame,
        "step": record.step,
        "event_window": record.event_window,
        "controller_solver_outcome": record.controller_solver_outcome,
        "solver_settings": {
            "solver_name": record.solver_name,
            "max_iterations": record.solver_max_iterations,
            "tolerance": record.solver_tolerance,
        },
        "solver_setup_arguments": dict(record.solver_setup_arguments),
        "runtime_provenance": dict(record.runtime_provenance),
        "first_action_support_comparison": (
            fixed_point_support_comparison_to_dict(
                record.first_action_support_comparison)),
        "stages": [_debug_qp_stage_to_dict(stage) for stage in record.stages],
    }


def _debug_solver_setup_arguments(
    solver_adapter: ResidualQpSolverAdapter,
    config: PlanningAwareV5ResidualQpMpcConfig,
) -> Dict[str, Any]:
    """Return only the setup contract actually supplied by repository code.

    This debug-only serializer must not make an OSQP default explicit merely
    for easier replay.  Fields absent from ``OsqpSolverAdapter.solve`` are
    explicitly identified as inherited/unknown instead.
    """
    if isinstance(solver_adapter, OsqpSolverAdapter):
        return {
            "adapter": "OsqpSolverAdapter",
            "arguments": {
                "verbose": False,
                "polish": True,
                "warm_start": True,
                "max_iter": solver_adapter.max_iterations,
                "eps_abs": solver_adapter.tolerance,
                "eps_rel": solver_adapter.tolerance,
            },
            "inherited_defaults": {
                "scaling": "inherited_osqp_default_not_explicit",
                "adaptive_rho": "inherited_osqp_default_not_explicit",
            },
            "identity_box_constraints": True,
        }
    if isinstance(solver_adapter, ProjectedGradientQpSolverAdapter):
        return {
            "adapter": "ProjectedGradientQpSolverAdapter",
            "arguments": {
                "max_iterations": solver_adapter.max_iterations,
                "tolerance": solver_adapter.tolerance,
            },
            "inherited_defaults": {},
            "identity_box_constraints": True,
        }
    return {
        "adapter": solver_adapter.__class__.__name__,
        "arguments": {
            "solver_name": _solver_name(config),
        },
        "inherited_defaults": {
            "all": "unknown_unavailable_adapter_no_setup_call",
        },
        "identity_box_constraints": True,
    }


def _debug_runtime_provenance() -> Dict[str, Any]:
    """Capture cheap package/interpreter identity for bounded debug rows."""
    packages: Dict[str, Dict[str, Any]] = {}
    for package_name in ("osqp", "numpy", "scipy"):
        try:
            module = importlib.import_module(package_name)
        except Exception as exc:  # noqa: BLE001 - debug provenance only.
            packages[package_name] = {
                "available": False,
                "version": "",
                "path": "",
                "error": "%s:%s" % (exc.__class__.__name__, exc),
            }
        else:
            packages[package_name] = {
                "available": True,
                "version": str(getattr(module, "__version__", "")),
                "path": str(getattr(module, "__file__", "")),
                "error": "",
            }
    return {
        "interpreter": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "packages": packages,
    }


def _debug_qp_snapshot_event_window(context: ControllerContext) -> bool:
    state = getattr(context, "state", VehicleState())
    speed = _safe_float(getattr(state, "speed", 0.0), 0.0)
    planning = getattr(context, "planning", None)
    summary = planning.preview_summary() if planning is not None else {}
    lateral = _safe_float(summary.get("preview_lateral_acc_max_abs", 0.0), 0.0)
    curvature = _safe_float(summary.get("preview_curvature_max_abs", 0.0), 0.0)
    mode = _operating_mode(context)
    return bool(
        speed >= 5.0 and
        (
            lateral >= 0.5 or
            curvature >= 0.005 or
            mode in ("lateral", "braking_combined")
        ))


def _residual_lifecycle_diagnostics(
    lifecycle: ResidualLifecycleTrace,
) -> Dict[str, Any]:
    initial = lifecycle.initial_solver_result
    zero = lifecycle.initial_zero_feasibility
    resolve_zero = lifecycle.resolve_zero_feasibility
    return {
        "residual_mpc_raw_model_linf": _residual_linf(
            lifecycle.raw_model_residual),
        "residual_mpc_raw_model_nonzero": _residual_nonzero(
            lifecycle.raw_model_residual),
        "residual_mpc_qp_solution_linf": _residual_linf(
            lifecycle.qp_solution_residual),
        "residual_mpc_qp_solution_nonzero": _residual_nonzero(
            lifecycle.qp_solution_residual),
        "residual_mpc_budget_accepted_linf": _residual_linf(
            lifecycle.budget_accepted_residual),
        "residual_mpc_budget_accepted_nonzero": _residual_nonzero(
            lifecycle.budget_accepted_residual),
        "residual_mpc_final_applied_linf": _residual_linf(
            lifecycle.final_applied_residual),
        "residual_mpc_final_applied_nonzero": _residual_nonzero(
            lifecycle.final_applied_residual),
        "residual_mpc_qp_solution_finite": int(
            lifecycle.qp_solution_finite),
        "residual_mpc_would_propose_without_timing_gate": int(
            lifecycle.would_propose_without_timing_gate),
        "residual_mpc_timing_gate_pass": int(lifecycle.timing_gate_pass),
        "residual_mpc_safety_projection_pass": int(
            lifecycle.safety_projection_pass),
        "residual_mpc_osqp_status": initial.osqp_status,
        "residual_mpc_osqp_status_val": _optional_diagnostic(
            initial.osqp_status_val),
        "residual_mpc_osqp_iter": _optional_diagnostic(initial.osqp_iter),
        "residual_mpc_osqp_prim_res": _optional_diagnostic(
            initial.osqp_prim_res),
        "residual_mpc_osqp_dual_res": _optional_diagnostic(
            initial.osqp_dual_res),
        "residual_mpc_osqp_prim_res_provenance": (
            initial.primal_residual_provenance),
        "residual_mpc_osqp_dual_res_provenance": (
            initial.dual_residual_provenance),
        "residual_mpc_osqp_obj_val": _optional_diagnostic(
            initial.osqp_obj_val),
        "residual_mpc_osqp_rho_updates": _optional_diagnostic(
            initial.osqp_rho_updates),
        "residual_mpc_osqp_setup_time_ms": _optional_diagnostic(
            initial.osqp_setup_time_ms),
        "residual_mpc_osqp_solve_time_ms": _optional_diagnostic(
            initial.osqp_solve_time_ms),
        "residual_mpc_osqp_update_time_ms": _optional_diagnostic(
            initial.osqp_update_time_ms),
        "residual_mpc_initial_qp_status": initial.status,
        "residual_mpc_first_action_resolve_status": (
            lifecycle.first_action_resolve_status),
        "residual_mpc_controller_solver_outcome": (
            lifecycle.controller_solver_outcome),
        "residual_mpc_zero_feasible": int(zero.feasible),
        "residual_mpc_zero_feas_max_violation": zero.max_violation,
        "residual_mpc_zero_feas_worst_group": zero.worst_group,
        "residual_mpc_zero_feas_worst_lower": zero.worst_lower,
        "residual_mpc_zero_feas_worst_value": zero.worst_value,
        "residual_mpc_zero_feas_worst_upper": zero.worst_upper,
        "residual_mpc_first_action_resolve_zero_feasible": int(
            resolve_zero.feasible),
        "residual_mpc_first_action_resolve_zero_feas_max_violation": (
            resolve_zero.max_violation),
        "residual_mpc_first_action_resolve_zero_feas_worst_group": (
            resolve_zero.worst_group),
        "residual_mpc_first_action_resolve_zero_feas_worst_lower": (
            resolve_zero.worst_lower),
        "residual_mpc_first_action_resolve_zero_feas_worst_value": (
            resolve_zero.worst_value),
        "residual_mpc_first_action_resolve_zero_feas_worst_upper": (
            resolve_zero.worst_upper),
    }


def _phase_a_config_values(
    config: PlanningAwareV5ResidualQpMpcConfig,
) -> Dict[str, Any]:
    phase_fields = {field.name for field in fields(PlanningAwareV4MpcPhaseAConfig)}
    values = {
        key: getattr(config, key)
        for key in phase_fields
        if hasattr(config, key)
    }
    values["controller_version"] = PLANNING_AWARE_V4_MPC_PHASEA_VERSION
    return values


def _timing_budget_selection(
    config: PlanningAwareV5ResidualQpMpcConfig,
) -> Tuple[str, float]:
    """Resolve the explicit gate while retaining the legacy default.

    Existing configurations select ``residual_mpc_compute_ms`` and continue
    to use ``max_compute_ms``.  An incremental selection uses its dedicated
    budget when finite, and otherwise safely maps back to ``max_compute_ms``.
    """

    metric_name = str(
        getattr(
            config,
            "timing_budget_metric",
            RESIDUAL_MPC_LEGACY_TIMING_BUDGET_METRIC) or "").strip()
    if metric_name not in RESIDUAL_MPC_TIMING_BUDGET_METRICS:
        metric_name = RESIDUAL_MPC_LEGACY_TIMING_BUDGET_METRIC
    legacy_budget_ms = max(
        0.0,
        _safe_float(getattr(config, "max_compute_ms", 40.0), 40.0))
    if metric_name != RESIDUAL_MPC_INCREMENTAL_TIMING_BUDGET_METRIC:
        return metric_name, legacy_budget_ms
    incremental_budget_ms = _optional_finite_float(getattr(
        config,
        "residual_incremental_budget_ms",
        None))
    return (
        metric_name,
        max(
            0.0,
            legacy_budget_ms
            if incremental_budget_ms is None else incremental_budget_ms))


def _stage_timing_value(
    stage_timing_ms: Mapping[str, float],
    name: str,
) -> float:
    return max(0.0, _safe_float(stage_timing_ms.get(name, 0.0), 0.0))


def _coerce_config_value(value: Any) -> Any:
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in {"true", "false"}:
            return lower == "true"
        try:
            if any(char in lower for char in (".", "e")):
                return float(value)
            return int(value)
        except ValueError:
            return value
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_manifest_model_consistency(
    manifest: Mapping[str, Any],
    model: OnlineArxModel,
) -> None:
    output_channels = tuple(str(item) for item in manifest.get("output_channels", ()))
    exogenous_channels = tuple(str(item) for item in manifest.get("exogenous_channels", ()))
    if output_channels != model.output_channels:
        raise ValueError("manifest_model_output_channels_mismatch")
    if exogenous_channels != model.exogenous_channels:
        raise ValueError("manifest_model_exogenous_channels_mismatch")
    active_modes = tuple(str(item) for item in manifest.get("active_modes", ()))
    for mode in active_modes:
        if mode not in model.input_modes:
            raise ValueError("active_mode_not_in_model:%s" % mode)
        if mode not in MODAL_BASIS_NAMES:
            raise ValueError("active_mode_not_in_contract:%s" % mode)


def _phase_vector_for_model(
    mode_names: Sequence[str],
    modal_all: Sequence[float],
) -> Tuple[float, ...]:
    by_mode = {
        mode: _match_length(modal_all, len(MODAL_BASIS_NAMES), 0.0)[index]
        for index, mode in enumerate(MODAL_BASIS_NAMES)
    }
    return tuple(_safe_float(by_mode.get(mode, 0.0), 0.0) for mode in mode_names)


def _active_to_candidate_modal(
    artifact: ResidualMpcArtifact,
    active_values: Sequence[float],
    target_modes: Sequence[str],
) -> Tuple[float, ...]:
    by_mode = {mode: 0.0 for mode in target_modes}
    for mode, value in zip(artifact.active_modes, active_values):
        if mode in by_mode:
            by_mode[mode] = _safe_float(value, 0.0)
    return tuple(by_mode[mode] for mode in target_modes)


def _active_mode_wheel_column(mode: str) -> Tuple[float, float, float, float]:
    values = [0.0, 0.0, 0.0, 0.0]
    if mode in MODAL_BASIS_NAMES:
        values[MODAL_BASIS_NAMES.index(mode)] = 1.0
    return tuple(modal_to_wheel_residual(tuple(values)))  # type: ignore[return-value]


def _prediction_weights(
    output_channels: Sequence[str],
    horizon: int,
    *,
    output_weight: float,
    body_weight: float,
    terminal_weight: float,
) -> Tuple[float, ...]:
    weights: List[float] = []
    for step in range(horizon):
        for channel in output_channels:
            lower = channel.lower()
            weight = output_weight
            if any(token in lower for token in ("roll", "pitch")):
                weight = body_weight
            if step == horizon - 1:
                weight += terminal_weight
            weights.append(max(float(weight), 0.0))
    return tuple(weights)


def _weighted_norm(values: Sequence[float], weights: Sequence[float]) -> float:
    return sum(
        weights[index] * values[index] * values[index]
        for index in range(min(len(values), len(weights))))


def _mismatch(
    left: Sequence[float],
    right: Sequence[float],
) -> Tuple[float, float]:
    length = max(len(left), len(right))
    l_values = _match_length(left, length, 0.0)
    r_values = _match_length(right, length, 0.0)
    deltas = [abs(l_values[index] - r_values[index]) for index in range(length)]
    return math.sqrt(sum(value * value for value in deltas)), max(deltas or [0.0])


def _empty_first_action_trace(
    dampers: Sequence[float],
) -> FirstActionProjectionTrace:
    values = tuple(_safe_float(value, 1.0) for value in dampers)
    return FirstActionProjectionTrace(
        raw_dampers=values,
        semi_active_dampers=values,
        safety_dampers=values,
        reprojected_dampers=values,
        fixed_point_dampers=values,
        pre_correction_l2=0.0,
        pre_correction_linf=0.0,
        raw_to_projected_l2=0.0,
        raw_to_projected_linf=0.0,
        projected_to_final_l2=0.0,
        projected_to_final_linf=0.0,
        final_to_reprojected_l2=0.0,
        final_to_reprojected_linf=0.0,
        fixed_point_linf=0.0,
        fixed_point_iterations=0,
        minmax_clip_count=0,
        rate_clip_count=0,
        low_rel_blend_count=0,
        infeasible_blend_count=0,
        modal_coupling_bound_miss_count=0)


def _replace_first_action_trace(
    trace: FirstActionProjectionTrace,
    **changes: Any,
) -> FirstActionProjectionTrace:
    return replace(trace, **changes)


def _invalid_fixed_point_support_comparison(
    error: str,
    active_modes: Sequence[str],
    tolerance: float,
) -> FixedPointSupportComparison:
    return FixedPointSupportComparison(
        evaluable=False,
        error=error,
        active_mode_order=tuple(str(mode) for mode in active_modes),
        wheel_residual=(),
        full_modal_target=(),
        active_target=(),
        existing_lower=(),
        existing_upper=(),
        lower_signed_margins=(),
        upper_signed_margins=(),
        violations=(),
        violation_count=0,
        violation_linf=0.0,
        worst_mode="",
        worst_index=-1,
        worst_side="",
        worst_target=0.0,
        worst_lower=0.0,
        worst_upper=0.0,
        worst_signed_gap=0.0,
        worst_absolute_gap=0.0,
        worst_normalized_gap=0.0,
        binding_constraint_group="",
        conversion_provenance=(
            "fixed_point_wheel_minus_phase_a_then_"
            "wheel_residual_to_modal_then_configured_active_order"),
        comparison_tolerance=tolerance,
        corrective_qp_constructed=False,
        corrective_qp_rejected=True)


def _binding_constraint_group_for_bound(
    problem: DenseQpProblem,
    index: int,
    side: str,
    bound: float,
    tolerance: float,
) -> Tuple[str, Tuple[str, ...]]:
    matches: List[Tuple[str, str]] = []
    support_matches: List[str] = []
    for group in problem.constraint_groups:
        if not (
                len(group.indices) == len(group.lower) == len(group.upper)):
            continue
        for position, group_index in enumerate(group.indices):
            if group_index != index:
                continue
            raw_value = (
                group.lower[position] if side == "lower"
                else group.upper[position])
            value = _optional_finite_float(raw_value)
            if value is None or abs(value - bound) > tolerance:
                continue
            matches.append((group.name, group.kind))
            if group.kind == "support_trust_region":
                support_matches.append(group.name)

    names = tuple(dict.fromkeys(name for name, _ in matches))
    if support_matches:
        # The rate/absolute group serializes the already intersected bound and
        # can therefore numerically tie the support group.  When support itself
        # equals the effective bound, it is the originating constraint.
        return support_matches[0], names
    for name, kind in matches:
        if kind == "first_action_rate_absolute_damper":
            return name, names
    if matches:
        return matches[0][0], names
    return "box_bounds_unattributed", ("box_bounds_unattributed",)


def compare_first_action_fixed_point_support(
    problem: DenseQpProblem,
    active_modes: Sequence[str],
    phase_a_dampers: Sequence[float],
    fixed_point_dampers: Sequence[float],
    *,
    tolerance: float = 1.0e-10,
) -> FixedPointSupportComparison:
    """Pure wheel-to-modal comparison used by online construction and reports.

    A negative lower-side signed gap means ``target < lower``.  A positive
    upper-side signed gap means ``target > upper``.  Absolute gaps are
    normalized by the existing coordinate span ``max(upper-lower, tolerance)``.
    Invalid dimensions, modes, or nonfinite values fail closed.
    """
    comparison_tolerance = _optional_finite_float(tolerance)
    if comparison_tolerance is None or comparison_tolerance < 0.0:
        comparison_tolerance = 1.0e-10
    modes = tuple(str(mode) for mode in active_modes)
    if (
            not modes or
            len(set(modes)) != len(modes) or
            any(mode not in MODAL_BASIS_NAMES for mode in modes)):
        return _invalid_fixed_point_support_comparison(
            "invalid_or_duplicate_active_mode_order",
            modes,
            comparison_tolerance)
    phase_values = tuple(
        value for value in (
            _optional_finite_float(item) for item in phase_a_dampers)
        if value is not None)
    fixed_values = tuple(
        value for value in (
            _optional_finite_float(item) for item in fixed_point_dampers)
        if value is not None)
    if (
            len(phase_values) != len(MODAL_BASIS_NAMES) or
            len(fixed_values) != len(MODAL_BASIS_NAMES) or
            len(phase_values) != len(tuple(phase_a_dampers)) or
            len(fixed_values) != len(tuple(fixed_point_dampers))):
        return _invalid_fixed_point_support_comparison(
            "wheel_vectors_must_be_finite_length_4",
            modes,
            comparison_tolerance)
    if len(problem.lower) < len(modes) or len(problem.upper) < len(modes):
        return _invalid_fixed_point_support_comparison(
            "problem_bounds_shorter_than_active_mode_order",
            modes,
            comparison_tolerance)
    lower_values = tuple(
        value for value in (
            _optional_finite_float(problem.lower[index])
            for index in range(len(modes)))
        if value is not None)
    upper_values = tuple(
        value for value in (
            _optional_finite_float(problem.upper[index])
            for index in range(len(modes)))
        if value is not None)
    if len(lower_values) != len(modes) or len(upper_values) != len(modes):
        return _invalid_fixed_point_support_comparison(
            "active_problem_bounds_contain_nonfinite_value",
            modes,
            comparison_tolerance)
    if any(lower > upper for lower, upper in zip(lower_values, upper_values)):
        return _invalid_fixed_point_support_comparison(
            "active_problem_lower_exceeds_upper",
            modes,
            comparison_tolerance)

    wheel_residual = tuple(
        fixed_values[index] - phase_values[index]
        for index in range(len(MODAL_BASIS_NAMES)))
    full_modal = wheel_residual_to_modal(wheel_residual)
    by_mode = dict(zip(MODAL_BASIS_NAMES, full_modal))
    active_target = tuple(by_mode[mode] for mode in modes)
    lower_margins = tuple(
        target - lower
        for target, lower in zip(active_target, lower_values))
    upper_margins = tuple(
        upper - target
        for target, upper in zip(active_target, upper_values))
    violations: List[FixedPointSupportViolation] = []
    for index, (mode, target, lower, upper) in enumerate(zip(
            modes, active_target, lower_values, upper_values)):
        span = max(upper - lower, comparison_tolerance, 1.0e-12)
        if target < lower - comparison_tolerance:
            signed_gap = target - lower
            group, candidates = _binding_constraint_group_for_bound(
                problem, index, "lower", lower, comparison_tolerance)
            violations.append(FixedPointSupportViolation(
                active_index=index,
                mode=mode,
                side="lower",
                target=target,
                lower=lower,
                upper=upper,
                signed_gap=signed_gap,
                absolute_gap=abs(signed_gap),
                normalized_gap=abs(signed_gap) / span,
                normalization_span=span,
                binding_constraint_group=group,
                binding_constraint_group_candidates=candidates))
        if target > upper + comparison_tolerance:
            signed_gap = target - upper
            group, candidates = _binding_constraint_group_for_bound(
                problem, index, "upper", upper, comparison_tolerance)
            violations.append(FixedPointSupportViolation(
                active_index=index,
                mode=mode,
                side="upper",
                target=target,
                lower=lower,
                upper=upper,
                signed_gap=signed_gap,
                absolute_gap=abs(signed_gap),
                normalized_gap=abs(signed_gap) / span,
                normalization_span=span,
                binding_constraint_group=group,
                binding_constraint_group_candidates=candidates))
    worst = max(
        violations,
        key=lambda row: (
            row.absolute_gap,
            row.normalized_gap,
            -row.active_index,
            row.side == "upper"),
        default=None)
    within = not violations
    return FixedPointSupportComparison(
        evaluable=True,
        error="",
        active_mode_order=modes,
        wheel_residual=wheel_residual,
        full_modal_target=tuple(full_modal),
        active_target=active_target,
        existing_lower=lower_values,
        existing_upper=upper_values,
        lower_signed_margins=lower_margins,
        upper_signed_margins=upper_margins,
        violations=tuple(violations),
        violation_count=len(violations),
        violation_linf=max(
            (row.absolute_gap for row in violations),
            default=0.0),
        worst_mode=worst.mode if worst is not None else "",
        worst_index=worst.active_index if worst is not None else -1,
        worst_side=worst.side if worst is not None else "",
        worst_target=worst.target if worst is not None else 0.0,
        worst_lower=worst.lower if worst is not None else 0.0,
        worst_upper=worst.upper if worst is not None else 0.0,
        worst_signed_gap=worst.signed_gap if worst is not None else 0.0,
        worst_absolute_gap=worst.absolute_gap if worst is not None else 0.0,
        worst_normalized_gap=(
            worst.normalized_gap if worst is not None else 0.0),
        binding_constraint_group=(
            worst.binding_constraint_group if worst is not None else "none"),
        conversion_provenance=(
            "fixed_point_wheel_minus_phase_a_then_"
            "wheel_residual_to_modal_then_configured_active_order"),
        comparison_tolerance=comparison_tolerance,
        corrective_qp_constructed=within,
        corrective_qp_rejected=not within)


def _tightened_problem_for_first_action(
    problem: DenseQpProblem,
    artifact: ResidualMpcArtifact,
    phase_a_dampers: Sequence[float],
    final_dampers: Sequence[float],
    *,
    require_within_existing_bounds: bool = False,
    comparison: Optional[FixedPointSupportComparison] = None,
) -> Optional[DenseQpProblem]:
    support_comparison = comparison or compare_first_action_fixed_point_support(
        problem,
        artifact.active_modes,
        phase_a_dampers,
        final_dampers,
        tolerance=1.0e-10)
    if not support_comparison.evaluable:
        return None
    active = support_comparison.active_target
    lower = list(problem.lower)
    upper = list(problem.upper)
    for index, value in enumerate(active):
        if (
                require_within_existing_bounds and
                support_comparison.violation_count > 0):
            return None
        lower[index] = value
        upper[index] = value
    fixed_point_group = QpConstraintGroup(
        name="first_action_fixed_point_resolve",
        kind="first_action_rate_absolute_damper",
        indices=tuple(range(len(active))),
        lower=tuple(active),
        upper=tuple(active),
        metadata={
            "constraint_role": "fixed_point_equality",
            "hard_safety_constraint": False,
        })
    metadata = dict(problem.metadata)
    metadata.update({
        "qp_stage": "first_action_resolve",
        "constraint_group_names": tuple(
            group.name
            for group in problem.constraint_groups + (fixed_point_group,)),
    })
    return DenseQpProblem(
        hessian=problem.hessian,
        linear=problem.linear,
        lower=tuple(lower),
        upper=tuple(upper),
        metadata=metadata,
        constraint_groups=problem.constraint_groups + (fixed_point_group,))


def _fallback_from_solver(result: SolverResult) -> str:
    status = str(result.status or "")
    if status == "solved":
        return ""
    if status in RESIDUAL_QP_MPC_FALLBACK_REASONS:
        return status
    if "infeasible" in status:
        return "solver_infeasible"
    if "time" in status:
        return "solver_timeout"
    if "unavailable" in status:
        return "solver_unavailable"
    return "solver_error"


def _exogenous_channel_value(channel: str, context: ControllerContext) -> float:
    state = getattr(context, "state", VehicleState())
    planning = getattr(context, "planning", None)
    summary = planning.preview_summary() if planning is not None else {}
    mapping = {
        "residual_mpc_preview_curvature_max_abs": summary.get(
            "preview_curvature_max_abs",
            0.0),
        "residual_mpc_preview_lateral_acc_max_abs": summary.get(
            "preview_lateral_acc_max_abs",
            0.0),
        "residual_mpc_preview_target_speed_now": summary.get(
            "preview_target_speed_now",
            0.0),
        "speed": getattr(state, "speed", 0.0),
        "vehicle_speed_mps": getattr(state, "speed", 0.0),
    }
    if channel not in mapping:
        raise ValueError("unsupported_exogenous_channel:%s" % channel)
    return _safe_float(mapping[channel], 0.0)


def _planning_age_frames(context: ControllerContext) -> Optional[int]:
    planning = getattr(context, "planning", None)
    state = getattr(context, "state", None)
    planning_frame = _optional_finite_float(getattr(planning, "frame", None))
    state_frame = _optional_finite_float(getattr(state, "frame", None))
    if planning_frame is None or state_frame is None or planning_frame < 0:
        return None
    return int(state_frame) - int(planning_frame)


def _preview_validity_reason(context: ControllerContext) -> str:
    planning = getattr(context, "planning", None)
    metadata = dict(getattr(planning, "metadata", {}) or {})
    return str(metadata.get("validity_reason", ""))


def _operating_mode(context: ControllerContext) -> str:
    planning = getattr(context, "planning", None)
    summary = planning.preview_summary() if planning is not None else {}
    lat = abs(_safe_float(summary.get("preview_lateral_acc_max_abs", 0.0), 0.0))
    ax = abs(_safe_float(summary.get("preview_longitudinal_acc_max_abs", 0.0), 0.0))
    if lat >= 3.0 and ax >= 2.0:
        return "braking_combined"
    if lat >= 3.0:
        return "lateral"
    if ax >= 2.0:
        return "braking_combined"
    return "calm"


def _modal_diagnostics(
    diagnostics: Dict[str, Any],
    prefix: str,
    values: Sequence[float],
) -> None:
    modal_values = _match_length(values, len(MODAL_BASIS_NAMES), 0.0)
    for mode_name, value in zip(MODAL_BASIS_NAMES, modal_values):
        diagnostics["%s_%s" % (prefix, mode_name)] = _safe_float(value, 0.0)


def _wheel_diagnostics(
    diagnostics: Dict[str, Any],
    prefix: str,
    values: Sequence[float],
) -> None:
    wheel_values = _match_length(values, len(_WHEEL_LABELS), 0.0)
    for label, value in zip(_WHEEL_LABELS, wheel_values):
        diagnostics["%s_%s" % (prefix, label)] = _safe_float(value, 0.0)


def _ensure_residual_qp_diagnostics(diagnostics: Dict[str, Any]) -> None:
    for key in RESIDUAL_QP_MPC_DIAGNOSTIC_FIELDS:
        if key in diagnostics:
            continue
        if (
                key.endswith("_version") or
                key.endswith("_mode") or
                key.endswith("_reason") or
                key.endswith("_id") or
                key.endswith("_family") or
                key.endswith("_status") or
                key.endswith("_modes") or
                key.endswith("_channels") or
                key.endswith("_path") or
                key.endswith("_dir") or
                key.endswith("_backend") or
                key.endswith("_name") or
                key.endswith("_group") or
                key.endswith("_outcome") or
                key.endswith("_provenance")):
            diagnostics[key] = ""
        else:
            diagnostics[key] = 0.0
