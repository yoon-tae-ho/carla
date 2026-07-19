"""CARLA-free checks for planning-aware v5 residual QP-MPC."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock
import zipfile
from pathlib import Path

from suspension_control.controllers import planning_aware_residual_qp_mpc as residual_qp
from suspension_control.controllers.base import (
    ControllerContext,
    ControllerOutput,
    PlanningInfo,
    VehicleState,
)
from suspension_control.controllers.planning_aware_residual_qp_mpc import (
    DenseQpProblem,
    OsqpSolverAdapter,
    QpConstraintGroup,
    ResidualQpSolverAdapter,
    PLANNING_AWARE_V5_RESIDUAL_QP_MPC_SCHEMA_VERSION,
    PLANNING_AWARE_V5_RESIDUAL_QP_MPC_VERSION,
    RESIDUAL_QP_MPC_DIAGNOSTIC_FIELDS,
    PlanningAwareV5ResidualQpMpcConfig,
    PlanningAwareV5ResidualQpMpcController,
    PlanningAwareV5ResidualQpMpcShadowController,
    ResidualMpcArtifact,
    SolverResult,
    check_zero_vector_feasibility,
    compare_first_action_fixed_point_support,
    require_residual_qp_solver_preflight,
    run_residual_qp_solver_preflight,
)


_LABELS = ("fl", "fr", "rl", "rr")


def _fixed_point_support_problem(
    *,
    lower=(-1.0, -1.0, -1.0, -1.0),
    upper=(1.0, 1.0, 1.0, 1.0),
    support_lower=(-1.0, -1.0, -1.0, -1.0),
    support_upper=(1.0, 1.0, 1.0, 1.0),
):
    indices = (0, 1, 2, 3)
    return DenseQpProblem(
        hessian=tuple(
            tuple(1.0 if row == col else 0.0 for col in indices)
            for row in indices),
        linear=(0.0, 0.0, 0.0, 0.0),
        lower=tuple(lower),
        upper=tuple(upper),
        metadata={"active_dim": 4},
        constraint_groups=(
            QpConstraintGroup(
                name="support_trust_region",
                kind="support_trust_region",
                indices=indices,
                lower=tuple(support_lower),
                upper=tuple(support_upper),
                metadata={}),
            QpConstraintGroup(
                name="first_action_rate_absolute_damper_tightening",
                kind="first_action_rate_absolute_damper",
                indices=indices,
                lower=tuple(lower),
                upper=tuple(upper),
                metadata={}),
        ))


def _fixed_dampers_from_modal(modal):
    wheel = residual_qp.modal_to_wheel_residual(modal)
    return tuple(1.0 + value for value in wheel)

_STEP01_TIMING_NUMERIC_FIELDS = (
    "residual_mpc_compute_ms",
    "residual_mpc_controller_end_to_end_ms",
    "residual_mpc_incremental_compute_ms",
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
)

_STEP02_LIFECYCLE_FIELDS = (
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
)

_STEP02_OSQP_FIELDS = (
    "residual_mpc_osqp_status",
    "residual_mpc_osqp_status_val",
    "residual_mpc_osqp_iter",
    "residual_mpc_osqp_prim_res",
    "residual_mpc_osqp_dual_res",
    "residual_mpc_osqp_obj_val",
    "residual_mpc_osqp_rho_updates",
    "residual_mpc_osqp_setup_time_ms",
    "residual_mpc_osqp_solve_time_ms",
    "residual_mpc_osqp_update_time_ms",
)


class _FakeWheel:

    def __init__(self, index, velocity):
        self.wheel_index_raw = index
        self.wheel_name_canonical = ("FL", "FR", "RL", "RR")[index]
        self.raw_suspension_offset_m = 0.01
        self.suspension_compression_m = 0.01
        self.suspension_travel_m = 0.01
        self.suspension_velocity_mps = velocity
        self.normalized_travel = 0.5
        self.contact_valid = True
        self.wheel_in_air = False
        self.field_valid = True
        self.velocity_valid = True


class _FakeSuspensionState:

    state_valid = True
    velocity_valid = True
    compression_convention_validated = True
    state_source = "fake"
    failure_reason = ""

    def __init__(self, velocities=(-0.5, 0.5, -0.5, 0.5)):
        self.wheels = tuple(
            _FakeWheel(index, velocity)
            for index, velocity in enumerate(velocities))


class _FakeNativeWheel:
    spring_strength = 35000.0
    spring_damper_rate = 4500.0


class _FakeNativeSuspension:
    wheels = (_FakeNativeWheel(),) * 4


class _ManualPerfCounterNs:
    """Deterministic monotonic clock for controller timing tests."""

    def __init__(self):
        self.now_ns = 0

    def __call__(self):
        return self.now_ns

    def advance_ms(self, elapsed_ms):
        self.now_ns += int(round(float(elapsed_ms) * 1.0e6))


class _TimedPhaseAController:
    """Return one Phase A output while charging each real evaluation."""

    def __init__(
        self,
        clock,
        output,
        *,
        shadow_ms,
        commit_ms,
    ):
        self.clock = clock
        self.output = output
        self.shadow_ms = float(shadow_ms)
        self.commit_ms = float(commit_ms)

    def compute_shadow(self, _context):
        self.clock.advance_ms(self.shadow_ms)
        return self.output

    def compute(self, _context):
        self.clock.advance_ms(self.commit_ms)
        return self.output


def _sim_root() -> str:
    return os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "..",
        ".."))


def _state(frame=100, elapsed_seconds=5.0, roll=1.0):
    return VehicleState(
        frame=frame,
        elapsed_seconds=elapsed_seconds,
        dt=0.05,
        speed=12.0,
        roll=roll,
        pitch=0.1,
        roll_rate=0.2,
        pitch_rate=0.1,
        yaw_rate=0.1,
        local_ax=0.1,
        local_ay=0.2,
        ax=0.1,
        ay=0.2)


def _planning(frame=100):
    return PlanningInfo(
        available=True,
        source="lead_tfv6",
        frame=frame,
        horizon_dt=0.1,
        target_speed=(12.0, 12.0, 12.0, 12.0),
        curvature=(0.0, 0.01, 0.0, -0.01),
        predicted_ax=(0.0, 0.0, 0.0, 0.0),
        predicted_ay=(0.0, 1.44, 0.0, -1.44),
        metadata={
            "validity_reason": "valid_prediction_with_preview",
            "speed_semantics": (
                "pred_target_speed_scalar_flattened_not_time_sequence"),
            "control_semantics": "current_final_control_single_sample",
        })


def _context(roll=1.0, suspension_velocities=(-0.5, 0.5, -0.5, 0.5)):
    return ControllerContext(
        state=_state(roll=roll),
        planning=_planning(),
        native_suspension=_FakeNativeSuspension(),
        current_suspension=_FakeNativeSuspension(),
        suspension_state=_FakeSuspensionState(suspension_velocities),
        suspension_state_valid=True,
        native_spring_strength_by_wheel=(35000.0,) * 4,
        native_damper_rate_by_wheel=(4500.0,) * 4,
        dt=0.05)


def _dampers(output):
    return tuple(wheel.damper_scale for wheel in output.command.wheels)


def _springs(output):
    return tuple(wheel.spring_scale for wheel in output.command.wheels)


def _write_artifact(root: Path, *, validation_status="validated_live") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    model_payload = {
        "schema_version": "residual_mpc_arx_model_v1",
        "model_family": "global_arx",
        "output_channels": ["residual_mpc_roll_deg_filt"],
        "input_modes": ["roll_front"],
        "phase_a_modes": ["mean", "roll_front", "roll_rear", "pitch"],
        "exogenous_channels": [],
        "orders": {"output_lags": 1, "input_lags": 1},
        "input_delay_ticks": 0,
        "ridge_alpha": 1.0e-5,
        "feature_names": [
            "intercept",
            "y_lag0::residual_mpc_roll_deg_filt",
            "u_applied_lag0::roll_front",
            "u_phaseA_lag0::mean",
            "u_phaseA_lag0::roll_front",
            "u_phaseA_lag0::roll_rear",
            "u_phaseA_lag0::pitch",
        ],
        "coefficients": [
            [0.0],
            [1.0],
            [-4.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
        ],
        "validation_status": validation_status,
        "regime_models": {},
    }
    model_path = root / "model.npz"
    with zipfile.ZipFile(model_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "model.json",
            json.dumps(model_payload, indent=2, sort_keys=True) + "\n")
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "residual_mpc_model_artifact_v1",
        "pipeline_schema_version": "residual_mpc_id_pipeline_v1",
        "model_id": "unit_validated_live_model",
        "model_family": "global_arx",
        "nominal_dt_s": 0.05,
        "input_delay_ticks": 0,
        "output_channels": ["residual_mpc_roll_deg_filt"],
        "exogenous_channels": [],
        "candidate_modes": ["mean", "roll_front", "roll_rear", "pitch"],
        "active_modes": ["roll_front"],
        "modal_matrix": [[0.0, 1.0, 0.0, 0.0]],
        "orders": {
            "output_lags": 1,
            "input_lags": 1,
            "ridge_alpha": 1.0e-5,
        },
        "normalization": {"feature_normalization": "none"},
        "training_support": {
            "roll_front": {"min": -0.05, "max": 0.05, "rms": 0.05},
        },
        "recommended_horizon_steps": 4,
        "recommended_control_block_steps": 2,
        "hashes": {"model_npz_sha256": digest},
        "validation_status": validation_status,
    }
    preprocessing = {
        "schema_version": "residual_mpc_preprocessing_v1",
        "causal_filter": "first_order_low_pass",
        "low_pass_tau_s": 0.15,
        "nominal_dt_s": 0.05,
        "zero_phase_filtering": False,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    (root / "preprocessing.json").write_text(
        json.dumps(preprocessing, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    (root / "validation_metrics.json").write_text("{}\n", encoding="utf-8")
    (root / "training_run_manifest.json").write_text("{}\n", encoding="utf-8")
    return root


def _active_config(artifact_dir, **overrides):
    values = {
        "model_artifact_dir": str(artifact_dir),
        "solver_name": "box_projected_gradient",
        "allow_dry_model": False,
        "require_validated_live": True,
        "max_rate_up_scale_per_s": 10.0,
        "max_rate_down_scale_per_s": 10.0,
        "projection_blend_infeasible": 0.0,
        "projection_blend_low_rel": 0.0,
        "residual_slew_weight": 0.0,
        "residual_effort_weight": 0.01,
        "max_compute_ms": 5000.0,
        "require_suspension_state_for_projection": False,
    }
    values.update(overrides)
    return PlanningAwareV5ResidualQpMpcConfig(**values)


def _fake_preflight_module(name):
    return types.SimpleNamespace(
        __version__="1.2.3",
        __file__="/fake/%s.py" % name.replace(".", "/"))


def _fake_preflight_import_module(name):
    return _fake_preflight_module(name)


class _PreflightOkAdapter(ResidualQpSolverAdapter):
    name = "osqp"

    def solve(self, problem_data, warm_start=None):
        return SolverResult(
            status="solved",
            primal_solution=(1.0,),
            objective=-1.0,
            iterations=4,
            setup_ms=0.125,
            solve_ms=0.250,
            total_ms=0.375,
            primal_residual=0.0,
            dual_residual=0.0)


class _PreflightFailAdapter(ResidualQpSolverAdapter):
    name = "osqp"

    def solve(self, problem_data, warm_start=None):
        return SolverResult(
            status="solver_error",
            error="forced_tiny_qp_failure")


class _MaximumIterationsAdapter(ResidualQpSolverAdapter):
    name = "osqp"

    def solve(self, problem_data, warm_start=None):
        return SolverResult(
            status="maximum iterations reached",
            primal_solution=tuple(0.04 for _ in problem_data.linear),
            objective=-0.5,
            iterations=80,
            osqp_status="maximum iterations reached",
            osqp_status_val=7,
            osqp_iter=80,
            osqp_prim_res=1.0e-3,
            osqp_dual_res=2.0e-3,
            osqp_obj_val=-0.5,
            osqp_rho_updates=1,
            osqp_setup_time_ms=0.2,
            osqp_solve_time_ms=0.4,
            osqp_update_time_ms=0.0)


class _InitialSolvedThenMaxIterationsAdapter(ResidualQpSolverAdapter):
    name = "osqp"

    def __init__(self):
        self.calls = 0

    def solve(self, problem_data, warm_start=None):
        self.calls += 1
        if self.calls == 1:
            return SolverResult(
                status="solved",
                primal_solution=tuple(0.04 for _ in problem_data.linear),
                objective=-1.0,
                iterations=12,
                osqp_status="solved",
                osqp_status_val=1,
                osqp_iter=12,
                osqp_prim_res=1.0e-9,
                osqp_dual_res=2.0e-9,
                osqp_obj_val=-1.0,
                osqp_rho_updates=0,
                osqp_setup_time_ms=0.2,
                osqp_solve_time_ms=0.3,
                osqp_update_time_ms=0.0)
        return SolverResult(
            status="maximum iterations reached",
            primal_solution=tuple(0.02 for _ in problem_data.linear),
            objective=-0.75,
            iterations=80,
            osqp_status="maximum iterations reached",
            osqp_status_val=7,
            osqp_iter=80,
            osqp_prim_res=1.0e-3,
            osqp_dual_res=2.0e-3,
            osqp_obj_val=-0.75,
            osqp_rho_updates=1,
            osqp_setup_time_ms=0.2,
            osqp_solve_time_ms=0.4,
            osqp_update_time_ms=0.0)


class PlanningAwareV5ResidualQpMpcTest(unittest.TestCase):

    def test_solver_preflight_records_versions_and_tiny_qp_success(self):
        result = run_residual_qp_solver_preflight(
            import_module=_fake_preflight_import_module,
            adapter_factory=lambda _osqp, _sparse: _PreflightOkAdapter())

        self.assertTrue(result.ok)
        self.assertEqual("osqp", result.solver_name)
        self.assertEqual("solved", result.status)
        self.assertEqual("1.2.3", result.package_versions["numpy"])
        self.assertIn("/fake/osqp.py", result.package_paths["osqp"])
        self.assertEqual((1.0,), result.solution)
        self.assertEqual(-1.0, result.objective)
        self.assertGreaterEqual(result.total_ms, 0.0)
        self.assertEqual("", result.error)

    def test_solver_preflight_missing_osqp_fails_before_live_ready(self):
        def fake_import(name):
            if name == "osqp":
                raise ImportError("osqp intentionally missing")
            return _fake_preflight_module(name)

        result = run_residual_qp_solver_preflight(import_module=fake_import)

        self.assertFalse(result.ok)
        self.assertEqual("import_error", result.status)
        self.assertIn("ImportError", result.error)
        self.assertIn("numpy", result.package_versions)
        self.assertNotIn("osqp", result.package_versions)

        with mock.patch.object(
                residual_qp,
                "run_residual_qp_solver_preflight",
                return_value=result):
            with self.assertRaisesRegex(
                    RuntimeError,
                    "residual QP solver preflight failed"):
                require_residual_qp_solver_preflight()

    def test_solver_preflight_tiny_qp_failure_fails_before_live_ready(self):
        result = run_residual_qp_solver_preflight(
            import_module=_fake_preflight_import_module,
            adapter_factory=lambda _osqp, _sparse: _PreflightFailAdapter())

        self.assertFalse(result.ok)
        self.assertEqual("solver_error", result.status)
        self.assertIn("forced_tiny_qp_failure", result.error)

        with mock.patch.object(
                residual_qp,
                "run_residual_qp_solver_preflight",
                return_value=result):
            with self.assertRaisesRegex(
                    RuntimeError,
                    "residual QP solver preflight failed"):
                require_residual_qp_solver_preflight()

    def test_zero_vector_feasibility_reports_constraint_provenance(self):
        feasible_problem = DenseQpProblem(
            hessian=((2.0,),),
            linear=(0.0,),
            lower=(-0.5,),
            upper=(0.5,),
            metadata={},
            constraint_groups=(QpConstraintGroup(
                name="support_trust_region",
                kind="support_trust_region",
                indices=(0,),
                lower=(-0.5,),
                upper=(0.5,),
                metadata={}),))
        infeasible_problem = DenseQpProblem(
            hessian=((2.0,),),
            linear=(0.0,),
            lower=(0.1,),
            upper=(0.5,),
            metadata={},
            constraint_groups=(QpConstraintGroup(
                name="future_hard_safety",
                kind="hard_safety",
                indices=(0,),
                lower=(0.1,),
                upper=(0.5,),
                metadata={}),))

        feasible = check_zero_vector_feasibility(feasible_problem)
        infeasible = check_zero_vector_feasibility(infeasible_problem)

        self.assertTrue(feasible.feasible)
        self.assertEqual(0.0, feasible.max_violation)
        self.assertEqual("none", feasible.worst_group)
        self.assertFalse(infeasible.feasible)
        self.assertAlmostEqual(0.1, infeasible.max_violation)
        self.assertEqual("future_hard_safety", infeasible.worst_group)
        self.assertAlmostEqual(0.1, infeasible.worst_lower)
        self.assertEqual(0.0, infeasible.worst_value)
        self.assertAlmostEqual(0.5, infeasible.worst_upper)

    def test_osqp_adapter_preserves_available_native_info(self):
        class _FakeOsqpSolver:
            def setup(self, **_kwargs):
                return None

            def warm_start(self, **_kwargs):
                return None

            def solve(self):
                info = types.SimpleNamespace(
                    status="solved",
                    status_val=1,
                    iter=17,
                    prim_res=1.0e-7,
                    dual_res=2.0e-7,
                    obj_val=-1.25,
                    rho_updates=3,
                    setup_time=0.0005,
                    solve_time=0.00075,
                    update_time=0.000125)
                return types.SimpleNamespace(x=(0.25,), info=info)

        fake_numpy = types.SimpleNamespace(
            asarray=lambda values, dtype=None: tuple(values))
        fake_sparse = types.SimpleNamespace(
            csc_matrix=lambda values: values,
            eye=lambda size, format=None: tuple(
                tuple(1.0 if row == col else 0.0 for col in range(size))
                for row in range(size)))
        fake_osqp = types.SimpleNamespace(OSQP=_FakeOsqpSolver)
        adapter = OsqpSolverAdapter(
            numpy_module=fake_numpy,
            osqp_module=fake_osqp,
            sparse_module=fake_sparse)
        problem = DenseQpProblem(
            hessian=((2.0,),),
            linear=(-1.0,),
            lower=(-0.5,),
            upper=(0.5,),
            metadata={})

        result = adapter.solve(problem)

        self.assertEqual("solved", result.status)
        self.assertEqual("solved", result.osqp_status)
        self.assertEqual(1, result.osqp_status_val)
        self.assertEqual(17, result.osqp_iter)
        self.assertAlmostEqual(1.0e-7, result.primal_residual)
        self.assertAlmostEqual(2.0e-7, result.dual_residual)
        self.assertTrue(result.primal_residual_available)
        self.assertTrue(result.dual_residual_available)
        self.assertEqual("osqp_info_prim_res", result.primal_residual_provenance)
        self.assertEqual("osqp_info_dual_res", result.dual_residual_provenance)
        self.assertAlmostEqual(1.0e-7, result.osqp_prim_res)
        self.assertAlmostEqual(2.0e-7, result.osqp_dual_res)
        self.assertAlmostEqual(-1.25, result.osqp_obj_val)
        self.assertEqual(3, result.osqp_rho_updates)
        self.assertAlmostEqual(0.5, result.osqp_setup_time_ms)
        self.assertAlmostEqual(0.75, result.osqp_solve_time_ms)
        self.assertAlmostEqual(0.125, result.osqp_update_time_ms)

    def test_osqp_adapter_accepts_osqp_06_residual_aliases_including_zero(self):
        class _FakeOsqpSolver:
            def setup(self, **_kwargs):
                return None

            def warm_start(self, **_kwargs):
                return None

            def solve(self):
                info = types.SimpleNamespace(
                    status="solved",
                    status_val=1,
                    iter=9,
                    pri_res=0.0,
                    dua_res=3.0e-4)
                return types.SimpleNamespace(x=(0.25,), info=info)

        fake_numpy = types.SimpleNamespace(
            asarray=lambda values, dtype=None: tuple(values))
        fake_sparse = types.SimpleNamespace(
            csc_matrix=lambda values: values,
            eye=lambda size, format=None: tuple(
                tuple(1.0 if row == col else 0.0 for col in range(size))
                for row in range(size)))
        adapter = OsqpSolverAdapter(
            numpy_module=fake_numpy,
            osqp_module=types.SimpleNamespace(OSQP=_FakeOsqpSolver),
            sparse_module=fake_sparse)
        result = adapter.solve(DenseQpProblem(
            hessian=((2.0,),), linear=(-1.0,), lower=(-0.5,),
            upper=(0.5,), metadata={}))

        self.assertEqual("solved", result.status)
        self.assertEqual(0.0, result.primal_residual)
        self.assertAlmostEqual(3.0e-4, result.dual_residual)
        self.assertEqual(0.0, result.osqp_prim_res)
        self.assertAlmostEqual(3.0e-4, result.osqp_dual_res)
        self.assertTrue(result.primal_residual_available)
        self.assertTrue(result.dual_residual_available)
        self.assertEqual("osqp_info_pri_res", result.primal_residual_provenance)
        self.assertEqual("osqp_info_dua_res", result.dual_residual_provenance)

    def test_osqp_adapter_marks_missing_native_residuals_unavailable(self):
        class _FakeOsqpSolver:
            def setup(self, **_kwargs):
                return None

            def solve(self):
                info = types.SimpleNamespace(status="solved", status_val=1, iter=4)
                return types.SimpleNamespace(x=(0.25,), info=info)

        fake_numpy = types.SimpleNamespace(
            asarray=lambda values, dtype=None: tuple(values))
        fake_sparse = types.SimpleNamespace(
            csc_matrix=lambda values: values,
            eye=lambda size, format=None: tuple(
                tuple(1.0 if row == col else 0.0 for col in range(size))
                for row in range(size)))
        adapter = OsqpSolverAdapter(
            numpy_module=fake_numpy,
            osqp_module=types.SimpleNamespace(OSQP=_FakeOsqpSolver),
            sparse_module=fake_sparse)
        result = adapter.solve(DenseQpProblem(
            hessian=((2.0,),), linear=(-1.0,), lower=(-0.5,),
            upper=(0.5,), metadata={}))

        # The retained numeric compatibility defaults are explicitly marked as
        # unmeasured; they cannot be interpreted as convergence evidence.
        self.assertEqual(0.0, result.primal_residual)
        self.assertEqual(0.0, result.dual_residual)
        self.assertIsNone(result.osqp_prim_res)
        self.assertIsNone(result.osqp_dual_res)
        self.assertFalse(result.primal_residual_available)
        self.assertFalse(result.dual_residual_available)
        self.assertEqual(
            "unavailable_osqp_info_prim_res_or_pri_res",
            result.primal_residual_provenance)
        self.assertEqual(
            "unavailable_osqp_info_dual_res_or_dua_res",
            result.dual_residual_provenance)

    def test_auto_solver_selects_osqp_backend_when_tiny_qp_passes(self):
        class _CountingOsqpAdapter(ResidualQpSolverAdapter):
            name = "osqp"
            init_count = 0
            tiny_qp_count = 0
            last_kwargs = {}

            def __init__(self, *args, **kwargs):
                type(self).init_count += 1
                type(self).last_kwargs = dict(kwargs)

            def solve(self, problem_data, warm_start=None):
                if problem_data.metadata.get(
                        "purpose") == "residual_qp_solver_preflight_tiny_qp":
                    type(self).tiny_qp_count += 1
                return SolverResult(
                    status="solved",
                    primal_solution=tuple(
                        0.0 for _ in range(len(problem_data.linear))),
                    objective=0.0)

        with mock.patch.object(
                residual_qp,
                "OsqpSolverAdapter",
                _CountingOsqpAdapter):
            solver = residual_qp._make_solver(  # pylint: disable=protected-access
                PlanningAwareV5ResidualQpMpcConfig(
                    solver_name="auto",
                    solver_max_iterations=123,
                    solver_tolerance=1.0e-6))

        self.assertIsInstance(solver, _CountingOsqpAdapter)
        self.assertEqual("osqp", solver.name)
        self.assertEqual(1, _CountingOsqpAdapter.init_count)
        self.assertEqual(1, _CountingOsqpAdapter.tiny_qp_count)
        self.assertEqual(123, _CountingOsqpAdapter.last_kwargs["max_iterations"])
        self.assertEqual(1.0e-6, _CountingOsqpAdapter.last_kwargs["tolerance"])

    def test_auto_solver_availability_is_cached_across_ticks(self):
        class _CountingOsqpAdapter(ResidualQpSolverAdapter):
            name = "osqp"
            init_count = 0
            tiny_qp_count = 0

            def __init__(self, *args, **kwargs):
                type(self).init_count += 1

            def solve(self, problem_data, warm_start=None):
                if problem_data.metadata.get(
                        "purpose") == "residual_qp_solver_preflight_tiny_qp":
                    type(self).tiny_qp_count += 1
                return SolverResult(
                    status="solved",
                    primal_solution=tuple(
                        0.0 for _ in range(len(problem_data.linear))),
                    objective=0.0,
                    total_ms=0.01)

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = _write_artifact(Path(temp_dir))
            controller = PlanningAwareV5ResidualQpMpcController(
                _active_config(artifact_dir, solver_name="auto"))
            with mock.patch.object(
                    residual_qp,
                    "OsqpSolverAdapter",
                    _CountingOsqpAdapter):
                first = controller.compute(_context(roll=1.0))
                second = controller.compute(_context(roll=1.1))

        self.assertEqual("osqp", first.diagnostics["residual_mpc_solver_name"])
        self.assertEqual("osqp", second.diagnostics["residual_mpc_solver_name"])
        self.assertEqual("solved", first.diagnostics["residual_mpc_solver_status"])
        self.assertEqual("solved", second.diagnostics["residual_mpc_solver_status"])
        self.assertEqual(1, _CountingOsqpAdapter.init_count)
        self.assertEqual(1, _CountingOsqpAdapter.tiny_qp_count)

    def test_unavailable_solver_fallback_remains_explicit_and_cached(self):
        class _UnavailableOsqpAdapter(ResidualQpSolverAdapter):
            name = "osqp"
            init_count = 0

            def __init__(self, *args, **kwargs):
                type(self).init_count += 1

            def solve(self, problem_data, warm_start=None):
                return SolverResult(
                    status="solver_unavailable",
                    error="forced_unavailable")

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = _write_artifact(Path(temp_dir))
            controller = PlanningAwareV5ResidualQpMpcController(
                _active_config(artifact_dir, solver_name="auto"))
            with mock.patch.object(
                    residual_qp,
                    "OsqpSolverAdapter",
                    _UnavailableOsqpAdapter):
                first = controller.compute(_context(roll=1.0))
                second = controller.compute(_context(roll=1.1))

        self.assertEqual(
            "solver_unavailable",
            first.diagnostics["residual_mpc_fallback_reason"])
        self.assertEqual(
            "solver_unavailable",
            second.diagnostics["residual_mpc_fallback_reason"])
        self.assertEqual(1, _UnavailableOsqpAdapter.init_count)

    def test_maximum_iterations_never_becomes_a_usable_proposal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = _write_artifact(Path(temp_dir))
            controller = PlanningAwareV5ResidualQpMpcController(
                _active_config(artifact_dir, solver_name="osqp"))
            controller._solver_adapter_cached = _MaximumIterationsAdapter()

            output = controller.compute(_context(roll=1.0))

        diagnostics = output.diagnostics
        self.assertEqual(
            "maximum iterations reached",
            diagnostics["residual_mpc_solver_status"])
        self.assertEqual(
            "maximum iterations reached",
            diagnostics["residual_mpc_initial_qp_status"])
        self.assertEqual(
            "maximum iterations reached",
            diagnostics["residual_mpc_osqp_status"])
        self.assertEqual(7, diagnostics["residual_mpc_osqp_status_val"])
        self.assertEqual(80, diagnostics["residual_mpc_osqp_iter"])
        self.assertEqual(
            "initial_qp_non_solved",
            diagnostics["residual_mpc_controller_solver_outcome"])
        self.assertEqual("not_run", diagnostics[
            "residual_mpc_first_action_resolve_status"])
        self.assertEqual(0, diagnostics["residual_mpc_qp_solution_finite"])
        self.assertEqual(0, diagnostics["residual_mpc_raw_model_nonzero"])
        self.assertEqual(0, diagnostics["residual_mpc_qp_solution_nonzero"])
        self.assertEqual(0, diagnostics[
            "residual_mpc_budget_accepted_nonzero"])
        self.assertEqual(0, diagnostics["residual_mpc_final_applied_nonzero"])
        self.assertEqual(0, diagnostics[
            "residual_mpc_would_propose_without_timing_gate"])
        self.assertEqual(0, diagnostics["residual_mpc_safety_projection_pass"])
        self.assertEqual(1, diagnostics["residual_mpc_zero_feasible"])
        self.assertEqual("phase_a_fallback", diagnostics["residual_mpc_mode"])

    def test_initial_solve_then_fixed_point_resolve_failure_is_split(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = _write_artifact(Path(temp_dir))
            controller = PlanningAwareV5ResidualQpMpcController(
                _active_config(
                    artifact_dir,
                    solver_name="osqp",
                    projection_blend_low_rel=0.5,
                    require_suspension_state_for_projection=True))
            adapter = _InitialSolvedThenMaxIterationsAdapter()
            controller._solver_adapter_cached = adapter

            output = controller.compute(_context(
                roll=1.0,
                suspension_velocities=(0.0, 0.0, 0.0, 0.0)))

        diagnostics = output.diagnostics
        self.assertEqual(2, adapter.calls)
        self.assertEqual("solved", diagnostics[
            "residual_mpc_initial_qp_status"])
        self.assertEqual("solved", diagnostics["residual_mpc_osqp_status"])
        self.assertEqual(12, diagnostics["residual_mpc_osqp_iter"])
        self.assertEqual(
            "maximum iterations reached",
            diagnostics["residual_mpc_first_action_resolve_status"])
        self.assertEqual(
            "initial_solution_for_corrective",
            diagnostics[
                "residual_mpc_first_action_resolve_warm_start_provenance"])
        self.assertEqual(1, diagnostics[
            "residual_mpc_first_action_resolve_warm_start_used"])
        self.assertGreater(diagnostics[
            "residual_mpc_first_action_resolve_warm_start_dimension"], 0)
        self.assertEqual(
            "maximum iterations reached",
            diagnostics["residual_mpc_solver_status"])
        self.assertEqual(
            "first_action_resolve_non_solved",
            diagnostics["residual_mpc_controller_solver_outcome"])
        self.assertEqual(1, diagnostics["residual_mpc_raw_model_nonzero"])
        self.assertEqual(0, diagnostics["residual_mpc_qp_solution_nonzero"])
        self.assertEqual(0, diagnostics["residual_mpc_qp_solution_finite"])
        self.assertEqual(0, diagnostics[
            "residual_mpc_budget_accepted_nonzero"])
        self.assertEqual(0, diagnostics["residual_mpc_final_applied_nonzero"])
        self.assertEqual(1, diagnostics["residual_mpc_zero_feasible"])
        self.assertEqual(1, diagnostics[
            "residual_mpc_first_action_resolve_zero_feasible"])
        self.assertEqual(
            "none",
            diagnostics[
                "residual_mpc_first_action_resolve_zero_feas_worst_group"])
        self.assertEqual("phase_a_fallback", diagnostics["residual_mpc_mode"])

    def test_fixed_point_support_comparison_accepts_in_support_and_tolerance_boundary(self):
        problem = _fixed_point_support_problem()
        in_support = compare_first_action_fixed_point_support(
            problem,
            residual_qp.MODAL_BASIS_NAMES,
            (1.0, 1.0, 1.0, 1.0),
            _fixed_dampers_from_modal((0.2, -0.3, 0.4, -0.5)))
        self.assertTrue(in_support.evaluable)
        self.assertEqual(0, in_support.violation_count)
        self.assertTrue(in_support.corrective_qp_constructed)

        for target in (
                (-1.0 - 0.5e-10, 0.0, 0.0, 0.0),
                (1.0 + 0.5e-10, 0.0, 0.0, 0.0)):
            boundary = compare_first_action_fixed_point_support(
                problem,
                residual_qp.MODAL_BASIS_NAMES,
                (1.0, 1.0, 1.0, 1.0),
                _fixed_dampers_from_modal(target),
                tolerance=1.0e-10)
            self.assertEqual(0, boundary.violation_count)

    def test_fixed_point_support_comparison_lower_and_upper_for_every_active_mode(self):
        problem = _fixed_point_support_problem()
        for index, mode in enumerate(residual_qp.MODAL_BASIS_NAMES):
            for side, target_value in (("lower", -1.1), ("upper", 1.1)):
                target = [0.0, 0.0, 0.0, 0.0]
                target[index] = target_value
                comparison = compare_first_action_fixed_point_support(
                    problem,
                    residual_qp.MODAL_BASIS_NAMES,
                    (1.0, 1.0, 1.0, 1.0),
                    _fixed_dampers_from_modal(tuple(target)))
                self.assertEqual(1, comparison.violation_count)
                violation = comparison.violations[0]
                self.assertEqual(index, violation.active_index)
                self.assertEqual(mode, violation.mode)
                self.assertEqual(side, violation.side)
                self.assertGreater(violation.absolute_gap, 0.0)
                self.assertGreater(violation.normalized_gap, 0.0)
                self.assertEqual(
                    "support_trust_region",
                    violation.binding_constraint_group)

    def test_fixed_point_support_comparison_multiple_and_permuted_modes(self):
        problem = _fixed_point_support_problem()
        modes = ("pitch", "mean", "roll_rear", "roll_front")
        # The physical modal target is mean=1.2, roll_front=-1.3,
        # roll_rear=0.0, pitch=0.0.  Permutation must only change active order.
        comparison = compare_first_action_fixed_point_support(
            problem,
            modes,
            (1.0, 1.0, 1.0, 1.0),
            _fixed_dampers_from_modal((1.2, -1.3, 0.0, 0.0)))
        self.assertEqual(modes, comparison.active_mode_order)
        for actual, expected in zip(
                comparison.active_target,
                (0.0, 1.2, 0.0, -1.3)):
            self.assertAlmostEqual(expected, actual, places=12)
        self.assertEqual(2, comparison.violation_count)
        self.assertEqual(
            {"mean", "roll_front"},
            {row.mode for row in comparison.violations})

    def test_fixed_point_support_binding_distinguishes_support_and_rate_tightening(self):
        problem = _fixed_point_support_problem(
            lower=(-0.5, -1.0, -1.0, -1.0),
            upper=(0.4, 1.0, 1.0, 1.0))
        rate_limited = compare_first_action_fixed_point_support(
            problem,
            residual_qp.MODAL_BASIS_NAMES,
            (1.0, 1.0, 1.0, 1.0),
            _fixed_dampers_from_modal((-0.6, 0.0, 0.0, 0.0)))
        self.assertEqual(
            "first_action_rate_absolute_damper_tightening",
            rate_limited.violations[0].binding_constraint_group)

        support_limited = compare_first_action_fixed_point_support(
            _fixed_point_support_problem(),
            residual_qp.MODAL_BASIS_NAMES,
            (1.0, 1.0, 1.0, 1.0),
            _fixed_dampers_from_modal((-1.1, 0.0, 0.0, 0.0)))
        self.assertEqual(
            "support_trust_region",
            support_limited.violations[0].binding_constraint_group)

    def test_fixed_point_support_comparison_round_trip_and_invalid_fail_closed(self):
        modal = (0.13, -0.21, 0.34, -0.08)
        wheel = residual_qp.modal_to_wheel_residual(modal)
        for actual, expected in zip(
                residual_qp.wheel_residual_to_modal(wheel), modal):
            self.assertAlmostEqual(expected, actual, places=12)
        valid = compare_first_action_fixed_point_support(
            _fixed_point_support_problem(),
            residual_qp.MODAL_BASIS_NAMES,
            (1.0, 1.0, 1.0, 1.0),
            tuple(1.0 + value for value in wheel))
        for actual, expected in zip(valid.full_modal_target, modal):
            self.assertAlmostEqual(expected, actual, places=12)

        for phase_a, fixed in (
                ((1.0, 1.0, 1.0), (1.0, 1.0, 1.0, 1.0)),
                ((1.0, 1.0, 1.0, 1.0), (1.0, math.nan, 1.0, 1.0))):
            invalid = compare_first_action_fixed_point_support(
                _fixed_point_support_problem(),
                residual_qp.MODAL_BASIS_NAMES,
                phase_a,
                fixed)
            self.assertFalse(invalid.evaluable)
            self.assertTrue(invalid.corrective_qp_rejected)

    def test_fixed_point_out_of_support_is_not_labelled_initial_osqp_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = _write_artifact(Path(temp_dir))
            controller = PlanningAwareV5ResidualQpMpcController(
                _active_config(
                    artifact_dir,
                    projection_blend_low_rel=0.5,
                    require_suspension_state_for_projection=True))
            with mock.patch.object(
                    residual_qp,
                    "_tightened_problem_for_first_action",
                    return_value=None):
                output = controller.compute(_context(
                    roll=1.0,
                    suspension_velocities=(0.0, 0.0, 0.0, 0.0)))

        diagnostics = output.diagnostics
        self.assertEqual("solved", diagnostics[
            "residual_mpc_initial_qp_status"])
        self.assertEqual(
            "not_run_fixed_point_out_of_support",
            diagnostics["residual_mpc_first_action_resolve_status"])
        self.assertEqual(
            "first_action_fixed_point_out_of_support",
            diagnostics["residual_mpc_controller_solver_outcome"])
        self.assertEqual(
            "solver_infeasible",
            diagnostics["residual_mpc_solver_status"])
        self.assertEqual("", diagnostics["residual_mpc_osqp_status"])
        self.assertEqual(1, diagnostics["residual_mpc_raw_model_nonzero"])
        self.assertEqual(0, diagnostics["residual_mpc_qp_solution_nonzero"])
        self.assertEqual(0, diagnostics[
            "residual_mpc_budget_accepted_nonzero"])
        self.assertEqual(0, diagnostics["residual_mpc_final_applied_nonzero"])

    def test_loader_rejects_dry_only_artifact_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = _write_artifact(Path(temp_dir), validation_status="dry_only")

            with self.assertRaises(ValueError):
                ResidualMpcArtifact.load(str(artifact_dir))

    def test_missing_manifest_falls_back_to_same_tick_phase_a(self):
        controller = PlanningAwareV5ResidualQpMpcController(
            PlanningAwareV5ResidualQpMpcConfig(
                model_artifact_dir="",
                solver_name="box_projected_gradient",
                require_suspension_state_for_projection=False))

        output = controller.compute(_context())
        diagnostics = output.diagnostics

        self.assertEqual((1.0, 1.0, 1.0, 1.0), _springs(output))
        self.assertEqual("phase_a_fallback", diagnostics["residual_mpc_mode"])
        self.assertEqual(1, diagnostics["residual_mpc_fallback_active"])
        self.assertEqual(
            "model_unavailable",
            diagnostics["residual_mpc_fallback_reason"])
        self.assertEqual(1, diagnostics["residual_mpc_phase_a_commit_match"])
        for label in _LABELS:
            self.assertAlmostEqual(
                diagnostics["residual_mpc_phase_a_shadow_damper_%s" % label],
                _dampers(output)[_LABELS.index(label)])

    def test_incremental_timing_excludes_both_measured_phase_a_calls(self):
        context = _context()
        controller = PlanningAwareV5ResidualQpMpcController(
            PlanningAwareV5ResidualQpMpcConfig(
                model_artifact_dir="",
                solver_name="box_projected_gradient",
                max_compute_ms=5000.0,
                record_stage_timing=True,
                record_solver_diagnostics=True,
                require_suspension_state_for_projection=False))
        phase_a_output = controller.phase_a_controller.compute_shadow(context)
        phase_a_output = ControllerOutput(
            command=phase_a_output.command,
            diagnostics=dict(
                phase_a_output.diagnostics,
                controller_compute_ms=9999.0))
        clock = _ManualPerfCounterNs()
        controller.phase_a_controller = _TimedPhaseAController(
            clock,
            phase_a_output,
            shadow_ms=2.0,
            commit_ms=3.0)

        def delayed_artifact_load():
            clock.advance_ms(7.0)
            return None

        with mock.patch.object(
                residual_qp.time,
                "perf_counter_ns",
                side_effect=clock), mock.patch.object(
                    controller,
                    "_load_artifact_once",
                    side_effect=delayed_artifact_load):
            output = controller.compute(context)

        diagnostics = output.diagnostics
        self.assertEqual(
            "model_unavailable",
            diagnostics["residual_mpc_fallback_reason"])
        self.assertAlmostEqual(
            2.0,
            diagnostics["residual_mpc_stage_phase_a_shadow_ms"],
            places=9)
        self.assertAlmostEqual(
            3.0,
            diagnostics["residual_mpc_stage_phase_a_commit_ms"],
            places=9)
        self.assertAlmostEqual(
            12.0,
            diagnostics["residual_mpc_controller_end_to_end_ms"],
            places=9)
        self.assertAlmostEqual(
            12.0,
            diagnostics["residual_mpc_compute_ms"],
            places=9)
        self.assertAlmostEqual(
            7.0,
            diagnostics["residual_mpc_incremental_compute_ms"],
            places=9)
        self.assertAlmostEqual(
            max(
                0.0,
                diagnostics["residual_mpc_controller_end_to_end_ms"] -
                diagnostics["residual_mpc_stage_phase_a_shadow_ms"] -
                diagnostics["residual_mpc_stage_phase_a_commit_ms"]),
            diagnostics["residual_mpc_incremental_compute_ms"],
            places=9)
        self.assertEqual(9999.0, diagnostics["controller_compute_ms"])
        self.assertNotEqual(
            diagnostics["controller_compute_ms"],
            diagnostics["residual_mpc_stage_phase_a_shadow_ms"])

    def test_incremental_budget_falls_back_on_residual_overrun(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = _write_artifact(Path(temp_dir))
            context = _context(roll=1.0)
            controller = PlanningAwareV5ResidualQpMpcShadowController(
                _active_config(
                    artifact_dir,
                    timing_budget_metric=(
                        "residual_mpc_incremental_compute_ms"),
                    residual_incremental_budget_ms=40.0,
                    record_stage_timing=True,
                    record_solver_diagnostics=True))
            phase_a_output = controller.phase_a_controller.compute_shadow(
                context)
            clock = _ManualPerfCounterNs()
            controller.phase_a_controller = _TimedPhaseAController(
                clock,
                phase_a_output,
                shadow_ms=2.0,
                commit_ms=3.0)
            current_output_vector = controller._current_output_vector
            charged = [False]

            def delayed_current_output_vector(*args, **kwargs):
                if not charged[0]:
                    clock.advance_ms(41.0)
                    charged[0] = True
                return current_output_vector(*args, **kwargs)

            with mock.patch.object(
                    residual_qp.time,
                    "perf_counter_ns",
                    side_effect=clock), mock.patch.object(
                        controller,
                        "_current_output_vector",
                        side_effect=delayed_current_output_vector):
                output = controller.compute(context)

        diagnostics = output.diagnostics
        self.assertTrue(charged[0])
        self.assertEqual(
            "compute_budget_exceeded",
            diagnostics["residual_mpc_fallback_reason"])
        self.assertEqual(
            "residual_mpc_incremental_compute_ms",
            diagnostics["residual_mpc_budget_metric_name"])
        self.assertAlmostEqual(
            diagnostics["residual_mpc_incremental_compute_ms"],
            diagnostics["residual_mpc_budget_metric_ms"],
            places=9)
        self.assertAlmostEqual(
            41.0,
            diagnostics["residual_mpc_budget_metric_ms"],
            places=9)
        self.assertAlmostEqual(
            40.0,
            diagnostics["residual_mpc_budget_ms"],
            places=9)
        self.assertEqual(
            1,
            diagnostics["residual_mpc_compute_budget_exceeded"])
        self.assertEqual(1, diagnostics["residual_mpc_raw_model_nonzero"])
        self.assertEqual(1, diagnostics["residual_mpc_qp_solution_nonzero"])
        self.assertEqual(1, diagnostics["residual_mpc_qp_solution_finite"])
        self.assertEqual(1, diagnostics[
            "residual_mpc_would_propose_without_timing_gate"])
        self.assertEqual(1, diagnostics["residual_mpc_safety_projection_pass"])
        self.assertEqual(0, diagnostics["residual_mpc_timing_gate_pass"])
        self.assertEqual(0, diagnostics[
            "residual_mpc_budget_accepted_nonzero"])
        self.assertEqual(0, diagnostics["residual_mpc_final_applied_nonzero"])
        self.assertTrue(diagnostics[
            "residual_mpc_controller_solver_outcome"].endswith(
                "__timing_rejected"))
        self.assertTrue(output.command.is_close(phase_a_output.command))

    def test_legacy_gate_still_includes_both_phase_a_evaluations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = _write_artifact(Path(temp_dir))
            context = _context(roll=1.0)

            def run(metric_name):
                values = {
                    "max_compute_ms": 40.0,
                    "record_stage_timing": True,
                    "record_solver_diagnostics": True,
                }
                if metric_name == "residual_mpc_incremental_compute_ms":
                    values.update({
                        "timing_budget_metric": metric_name,
                        "residual_incremental_budget_ms": 40.0,
                    })
                controller = PlanningAwareV5ResidualQpMpcShadowController(
                    _active_config(artifact_dir, **values))
                phase_a_output = controller.phase_a_controller.compute_shadow(
                    context)
                clock = _ManualPerfCounterNs()
                controller.phase_a_controller = _TimedPhaseAController(
                    clock,
                    phase_a_output,
                    shadow_ms=30.0,
                    commit_ms=30.0)
                with mock.patch.object(
                        residual_qp.time,
                        "perf_counter_ns",
                        side_effect=clock):
                    return controller.compute(context), phase_a_output

            legacy_output, legacy_phase_a = run("residual_mpc_compute_ms")
            incremental_output, incremental_phase_a = run(
                "residual_mpc_incremental_compute_ms")

        self.assertEqual(
            "compute_budget_exceeded",
            legacy_output.diagnostics["residual_mpc_fallback_reason"])
        self.assertAlmostEqual(
            60.0,
            legacy_output.diagnostics["residual_mpc_budget_metric_ms"],
            places=9)
        self.assertEqual(
            "shadow_qp_mpc",
            incremental_output.diagnostics["residual_mpc_mode"])
        self.assertEqual(
            "none",
            incremental_output.diagnostics["residual_mpc_fallback_reason"])
        self.assertAlmostEqual(
            0.0,
            incremental_output.diagnostics["residual_mpc_budget_metric_ms"],
            places=9)
        self.assertTrue(legacy_output.command.is_close(legacy_phase_a.command))
        self.assertTrue(
            incremental_output.command.is_close(incremental_phase_a.command))

    def test_timing_diagnostics_have_finite_csv_defaults(self):
        import transfuser_suspension_control_suite as suite

        controller = PlanningAwareV5ResidualQpMpcController(
            PlanningAwareV5ResidualQpMpcConfig(
                model_artifact_dir="",
                solver_name="box_projected_gradient",
                max_compute_ms=5000.0,
                record_stage_timing=False,
                record_solver_diagnostics=False,
                require_suspension_state_for_projection=False))
        diagnostics = controller.compute(_context()).diagnostics

        for field in _STEP02_LIFECYCLE_FIELDS + _STEP02_OSQP_FIELDS:
            self.assertIn(field, RESIDUAL_QP_MPC_DIAGNOSTIC_FIELDS)
            self.assertIn(field, suite.DIAGNOSTIC_FIELDS)
            self.assertIn(field, diagnostics)
        for field in _STEP02_OSQP_FIELDS:
            self.assertEqual("", diagnostics[field])
        for field in (
                "residual_mpc_solver_primal_residual_available",
                "residual_mpc_solver_dual_residual_available",
                "residual_mpc_solver_primal_residual_provenance",
                "residual_mpc_solver_dual_residual_provenance",
                "residual_mpc_osqp_prim_res_provenance",
                "residual_mpc_osqp_dual_res_provenance"):
            self.assertIn(field, RESIDUAL_QP_MPC_DIAGNOSTIC_FIELDS)
            self.assertIn(field, suite.DIAGNOSTIC_FIELDS)
            self.assertIn(field, diagnostics)
        self.assertEqual(0, diagnostics[
            "residual_mpc_solver_primal_residual_available"])
        self.assertEqual(0, diagnostics[
            "residual_mpc_solver_dual_residual_available"])
        self.assertEqual("unavailable", diagnostics[
            "residual_mpc_solver_primal_residual_provenance"])
        self.assertEqual("unavailable", diagnostics[
            "residual_mpc_solver_dual_residual_provenance"])
        self.assertEqual("not_run", diagnostics[
            "residual_mpc_initial_qp_status"])
        self.assertEqual("not_run", diagnostics[
            "residual_mpc_first_action_resolve_status"])
        self.assertEqual("not_run", diagnostics[
            "residual_mpc_controller_solver_outcome"])
        self.assertEqual(0, diagnostics["residual_mpc_zero_feasible"])
        self.assertEqual(
            "initial_qp_not_constructed",
            diagnostics["residual_mpc_zero_feas_worst_group"])
        self.assertIn(
            "residual_mpc_budget_metric_name",
            RESIDUAL_QP_MPC_DIAGNOSTIC_FIELDS)
        self.assertIn(
            "residual_mpc_budget_metric_name",
            suite.DIAGNOSTIC_FIELDS)
        self.assertEqual(
            "residual_mpc_compute_ms",
            diagnostics["residual_mpc_budget_metric_name"])
        for field in _STEP01_TIMING_NUMERIC_FIELDS:
            self.assertIn(field, RESIDUAL_QP_MPC_DIAGNOSTIC_FIELDS)
            self.assertIn(field, suite.DIAGNOSTIC_FIELDS)
            self.assertIn(field, diagnostics)
            self.assertTrue(
                math.isfinite(float(diagnostics[field])),
                msg="nonfinite controller diagnostic %s=%r" % (
                    field,
                    diagnostics[field]))
            self.assertGreaterEqual(
                float(diagnostics[field]),
                0.0,
                msg="negative controller diagnostic %s=%r" % (
                    field,
                    diagnostics[field]))
        self.assertGreaterEqual(
            diagnostics["residual_mpc_controller_end_to_end_ms"],
            diagnostics["residual_mpc_compute_ms"])
        for field in (
                "residual_mpc_stage_feature_build_ms",
                "residual_mpc_stage_model_predict_ms",
                "residual_mpc_stage_qp_matrix_build_ms",
                "residual_mpc_stage_qp_setup_or_update_ms",
                "residual_mpc_stage_qp_solve_wall_ms",
                "residual_mpc_stage_qp_postprocess_ms",
                "residual_mpc_stage_safety_projection_ms"):
            self.assertEqual(0.0, diagnostics[field])

        csv_buffer = io.StringIO()
        writer = csv.DictWriter(csv_buffer, fieldnames=suite.DIAGNOSTIC_FIELDS)
        writer.writeheader()
        writer.writerow({
            field: suite.format_value(diagnostics.get(field, ""))
            for field in suite.DIAGNOSTIC_FIELDS
        })
        csv_buffer.seek(0)
        row = next(csv.DictReader(csv_buffer))
        self.assertEqual(
            "residual_mpc_compute_ms",
            row["residual_mpc_budget_metric_name"])
        for field in _STEP01_TIMING_NUMERIC_FIELDS:
            self.assertNotEqual("", row[field], msg="blank CSV field %s" % field)
            self.assertTrue(
                math.isfinite(float(row[field])),
                msg="nonfinite CSV field %s=%r" % (field, row[field]))
            self.assertGreaterEqual(
                float(row[field]),
                0.0,
                msg="negative CSV field %s=%r" % (field, row[field]))

    def test_validated_artifact_solves_nonzero_active_mode_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = _write_artifact(Path(temp_dir))
            controller = PlanningAwareV5ResidualQpMpcController(
                _active_config(artifact_dir))

            output = controller.compute(_context(roll=1.0))
            diagnostics = output.diagnostics

        self.assertEqual(
            PLANNING_AWARE_V5_RESIDUAL_QP_MPC_SCHEMA_VERSION,
            diagnostics["residual_mpc_schema_version"])
        self.assertEqual(
            PLANNING_AWARE_V5_RESIDUAL_QP_MPC_VERSION,
            diagnostics["residual_mpc_controller_version"])
        self.assertEqual("residual_qp_mpc", diagnostics["residual_mpc_mode"])
        self.assertEqual(0, diagnostics["residual_mpc_fallback_active"])
        self.assertEqual("solved", diagnostics["residual_mpc_solver_status"])
        self.assertEqual("solved", diagnostics[
            "residual_mpc_initial_qp_status"])
        self.assertEqual("", diagnostics["residual_mpc_osqp_status"])
        self.assertEqual("", diagnostics["residual_mpc_osqp_status_val"])
        self.assertEqual("", diagnostics["residual_mpc_osqp_iter"])
        self.assertEqual(1, diagnostics["residual_mpc_zero_feasible"])
        self.assertEqual(1, diagnostics[
            "residual_mpc_budget_accepted_nonzero"])
        self.assertEqual(1, diagnostics["residual_mpc_final_applied_nonzero"])
        self.assertAlmostEqual(
            diagnostics["residual_mpc_budget_accepted_linf"],
            diagnostics["residual_mpc_final_applied_linf"],
            places=12)
        self.assertEqual(
            "roll_front",
            diagnostics["residual_mpc_active_modes"])
        self.assertGreater(
            abs(diagnostics["residual_mpc_requested_residual_modal_roll_front"]),
            1.0e-4)
        self.assertAlmostEqual(
            0.0,
            diagnostics["residual_mpc_requested_residual_modal_mean"],
            places=12)
        self.assertAlmostEqual(
            0.0,
            diagnostics["residual_mpc_requested_residual_modal_pitch"],
            places=12)
        self.assertGreaterEqual(
            diagnostics["residual_mpc_predicted_cost_gain_abs"],
            -1.0e-9)
        self.assertEqual((1.0, 1.0, 1.0, 1.0), _springs(output))

    def test_low_relative_velocity_first_action_retarget_is_truthful(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = _write_artifact(Path(temp_dir))
            controller = PlanningAwareV5ResidualQpMpcController(
                _active_config(
                    artifact_dir,
                    projection_blend_low_rel=0.5,
                    require_suspension_state_for_projection=True))

            output = controller.compute(
                _context(
                    roll=1.0,
                    suspension_velocities=(0.0, 0.0, 0.0, 0.0)))
            diagnostics = output.diagnostics

        self.assertEqual("residual_qp_mpc", diagnostics["residual_mpc_mode"])
        self.assertEqual(0, diagnostics["residual_mpc_fallback_active"])
        self.assertEqual(1, diagnostics[
            "residual_mpc_first_action_correction_active"])
        self.assertGreater(
            diagnostics["residual_mpc_first_action_pre_correction_linf"],
            1.0e-5)
        self.assertGreater(
            diagnostics["residual_mpc_first_action_low_rel_blend_count"],
            0)
        self.assertLessEqual(
            diagnostics["residual_mpc_first_action_qp_to_applied_linf"],
            1.0e-5)
        self.assertLessEqual(
            diagnostics["residual_mpc_first_action_fixed_point_linf"],
            1.0e-5)
        self.assertEqual("solved", diagnostics[
            "residual_mpc_initial_qp_status"])
        self.assertEqual("solved", diagnostics[
            "residual_mpc_first_action_resolve_status"])
        self.assertEqual(
            "first_action_resolve_solved",
            diagnostics["residual_mpc_controller_solver_outcome"])
        self.assertEqual(1, diagnostics[
            "residual_mpc_first_action_support_comparison_evaluable"])
        self.assertEqual(
            "roll_front",
            diagnostics["residual_mpc_first_action_active_mode_order"])
        self.assertEqual(0, diagnostics[
            "residual_mpc_first_action_support_violation_count"])
        self.assertEqual(1, diagnostics[
            "residual_mpc_first_action_corrective_qp_constructed"])
        self.assertEqual(0, diagnostics[
            "residual_mpc_first_action_corrective_qp_rejected"])
        self.assertEqual(1, diagnostics["residual_mpc_zero_feasible"])
        self.assertEqual(1, diagnostics[
            "residual_mpc_first_action_resolve_zero_feasible"])
        self.assertEqual(
            "none",
            diagnostics[
                "residual_mpc_first_action_resolve_zero_feas_worst_group"])

    def test_shadow_companion_computes_solver_but_applies_phase_a(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = _write_artifact(Path(temp_dir))
            controller = PlanningAwareV5ResidualQpMpcShadowController(
                _active_config(artifact_dir))

            output = controller.compute(_context(roll=1.0))
            diagnostics = output.diagnostics

        self.assertEqual("shadow_qp_mpc", diagnostics["residual_mpc_mode"])
        self.assertEqual(1, diagnostics["residual_mpc_shadow_mode"])
        self.assertEqual("solved", diagnostics["residual_mpc_solver_status"])
        self.assertGreater(
            abs(diagnostics["residual_mpc_requested_residual_modal_roll_front"]),
            1.0e-4)
        self.assertEqual(1, diagnostics["residual_mpc_raw_model_nonzero"])
        self.assertEqual(1, diagnostics["residual_mpc_qp_solution_nonzero"])
        self.assertEqual(1, diagnostics["residual_mpc_qp_solution_finite"])
        self.assertEqual(1, diagnostics[
            "residual_mpc_budget_accepted_nonzero"])
        self.assertGreater(
            diagnostics["residual_mpc_budget_accepted_linf"],
            1.0e-4)
        self.assertEqual(0, diagnostics["residual_mpc_final_applied_nonzero"])
        self.assertEqual(1, diagnostics["residual_mpc_timing_gate_pass"])
        self.assertEqual(1, diagnostics["residual_mpc_safety_projection_pass"])
        self.assertEqual(1, diagnostics[
            "residual_mpc_would_propose_without_timing_gate"])
        self.assertEqual(1, diagnostics["residual_mpc_phase_a_commit_match"])
        for label in _LABELS:
            self.assertAlmostEqual(
                0.0,
                diagnostics["residual_mpc_final_residual_%s" % label],
                places=12)
            self.assertAlmostEqual(
                diagnostics["residual_mpc_phase_a_shadow_damper_%s" % label],
                _dampers(output)[_LABELS.index(label)])

    def test_shadow_command_is_unchanged_by_timing_metric_selection(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = _write_artifact(Path(temp_dir))
            legacy_controller = PlanningAwareV5ResidualQpMpcShadowController(
                _active_config(
                    artifact_dir,
                    max_compute_ms=5000.0,
                    record_stage_timing=True,
                    record_solver_diagnostics=True))
            incremental_controller = (
                PlanningAwareV5ResidualQpMpcShadowController(
                    _active_config(
                        artifact_dir,
                        max_compute_ms=5000.0,
                        timing_budget_metric=(
                            "residual_mpc_incremental_compute_ms"),
                        residual_incremental_budget_ms=5000.0,
                        record_stage_timing=True,
                        record_solver_diagnostics=True)))

            legacy_output = legacy_controller.compute(_context(roll=1.0))
            incremental_output = incremental_controller.compute(
                _context(roll=1.0))

        self.assertEqual(
            "residual_mpc_compute_ms",
            legacy_output.diagnostics["residual_mpc_budget_metric_name"])
        self.assertEqual(
            "residual_mpc_incremental_compute_ms",
            incremental_output.diagnostics["residual_mpc_budget_metric_name"])
        self.assertEqual(
            "shadow_qp_mpc",
            legacy_output.diagnostics["residual_mpc_mode"])
        self.assertEqual(
            "shadow_qp_mpc",
            incremental_output.diagnostics["residual_mpc_mode"])
        self.assertTrue(
            legacy_output.command.is_close(incremental_output.command))
        for output in (legacy_output, incremental_output):
            for index, label in enumerate(_LABELS):
                self.assertAlmostEqual(
                    output.diagnostics[
                        "residual_mpc_phase_a_shadow_damper_%s" % label],
                    _dampers(output)[index],
                    places=12)

    def test_debug_qp_snapshot_is_bounded_and_written_after_compute(self):
        import transfuser_suspension_control_suite as suite

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact_dir = _write_artifact(root / "artifact")
            snapshot_path = root / "debug_snapshots.jsonl"
            disabled_path = root / "disabled_snapshots.jsonl"
            controller = PlanningAwareV5ResidualQpMpcShadowController(
                _active_config(
                    artifact_dir,
                    projection_blend_low_rel=0.5,
                    require_suspension_state_for_projection=True,
                    debug_qp_snapshot_enabled=True,
                    debug_qp_snapshot_every_n=1,
                    debug_qp_snapshot_event_window_only=False,
                    debug_qp_snapshot_max_rows=1))

            output = controller.compute(_context(
                roll=1.0,
                suspension_velocities=(0.0, 0.0, 0.0, 0.0)))
            self.assertFalse(snapshot_path.exists())
            written = suite.write_controller_debug_qp_snapshots(
                controller,
                str(snapshot_path),
                append=False)
            self.assertEqual(1, written)
            rows = snapshot_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(1, len(rows))
            snapshot = json.loads(rows[0])
            self.assertEqual(
                "residual_mpc_debug_qp_snapshot_v2",
                snapshot["schema_version"])
            self.assertEqual(100, snapshot["frame"])
            self.assertEqual(
                snapshot["key"],
                output.diagnostics["residual_mpc_debug_qp_snapshot_key"])
            self.assertEqual(
                int(snapshot["key"].split(":")[2]),
                output.diagnostics[
                    "residual_mpc_debug_qp_snapshot_candidate_ordinal"])
            self.assertEqual("box_projected_gradient", snapshot[
                "solver_settings"]["solver_name"])
            self.assertEqual(80, snapshot[
                "solver_settings"]["max_iterations"])
            self.assertTrue(snapshot["stages"])
            initial = snapshot["stages"][0]
            self.assertEqual("initial", initial["stage"])
            self.assertGreater(initial["dimensions"]["variables"], 0)
            self.assertTrue(initial["qp"]["identity_box_constraints"])
            group_names = {
                group["name"] for group in initial["constraint_groups"]
            }
            self.assertIn("support_trust_region", group_names)
            self.assertIn(
                "first_action_rate_absolute_damper_tightening",
                group_names)
            self.assertIn("zero_feasibility", initial)
            self.assertIn("solver_result", initial)
            self.assertEqual(1, initial["solve_ordinal"])
            self.assertEqual("none", initial["warm_start_input"]["provenance"])
            self.assertFalse(initial["warm_start_input"]["supplied"])
            self.assertEqual(0, initial["warm_start_input"]["dimension"])
            self.assertEqual([], initial["warm_start_input"]["values"])
            self.assertTrue(initial["solver_result"]["primal_finite"])
            self.assertEqual(
                initial["dimensions"]["variables"],
                len(initial["solver_result"]["primal_solution"]))
            self.assertIn(
                "primal_residual_provenance", initial["solver_result"])
            self.assertIn(
                "dual_residual_provenance", initial["solver_result"])
            self.assertEqual(
                "ProjectedGradientQpSolverAdapter",
                snapshot["solver_setup_arguments"]["adapter"])
            self.assertIn("arguments", snapshot["solver_setup_arguments"])
            self.assertIn("inherited_defaults", snapshot[
                "solver_setup_arguments"])
            self.assertIn("interpreter", snapshot["runtime_provenance"])
            self.assertEqual(
                {"osqp", "numpy", "scipy"},
                set(snapshot["runtime_provenance"]["packages"]))
            self.assertIn("first_action_support_comparison", snapshot)
            self.assertTrue(snapshot[
                "first_action_support_comparison"]["evaluable"])
            self.assertEqual(
                snapshot["first_action_support_comparison"]["active_mode_order"],
                output.diagnostics[
                    "residual_mpc_first_action_active_mode_order"].split(","))

            controller.compute(_context(
                roll=1.1,
                suspension_velocities=(0.0, 0.0, 0.0, 0.0)))
            written_again = suite.write_controller_debug_qp_snapshots(
                controller,
                str(snapshot_path),
                append=True)
            self.assertEqual(0, written_again)
            self.assertEqual(
                1,
                len(snapshot_path.read_text(encoding="utf-8").splitlines()))

            disabled = PlanningAwareV5ResidualQpMpcShadowController(
                _active_config(artifact_dir))
            disabled.compute(_context(roll=1.0))
            disabled_written = suite.write_controller_debug_qp_snapshots(
                disabled,
                str(disabled_path),
                append=False)
            self.assertEqual(0, disabled_written)
            self.assertFalse(disabled_path.exists())

    def test_warm_start_diagnostics_are_pre_solve_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact_dir = _write_artifact(root / "artifact")
            controller = PlanningAwareV5ResidualQpMpcShadowController(
                _active_config(
                    artifact_dir,
                    solver_name="box_projected_gradient",
                    max_compute_ms=5000.0))

            first = controller.compute(_context(roll=1.0)).diagnostics
            self.assertEqual(0, first["residual_mpc_solver_warm_start_used"])
            self.assertEqual(
                0, first["residual_mpc_initial_qp_warm_start_used"])
            self.assertEqual(
                "none", first[
                    "residual_mpc_initial_qp_warm_start_provenance"])
            self.assertEqual(
                0, first["residual_mpc_initial_qp_warm_start_dimension"])

            second = controller.compute(_context(roll=1.1)).diagnostics
            self.assertEqual(1, second["residual_mpc_solver_warm_start_used"])
            self.assertEqual(
                1, second["residual_mpc_initial_qp_warm_start_used"])
            self.assertEqual(
                "previous_controller_solution", second[
                    "residual_mpc_initial_qp_warm_start_provenance"])
            self.assertGreater(
                second["residual_mpc_initial_qp_warm_start_dimension"], 0)
            # This non-corrective row supplied no second solver input.
            self.assertEqual(
                "none", second[
                    "residual_mpc_first_action_resolve_warm_start_provenance"])

    def test_timing_budget_metric_config_selection_is_backward_compatible(self):
        import transfuser_suspension_control_suite as suite

        legacy_mapping = PlanningAwareV5ResidualQpMpcConfig.from_mapping({
            "max_compute_ms": 17.0,
        })
        explicit_mapping = PlanningAwareV5ResidualQpMpcConfig.from_mapping({
            "max_compute_ms": 17.0,
            "timing_budget_metric": "residual_mpc_incremental_compute_ms",
            "residual_incremental_budget_ms": 40.0,
            "record_stage_timing": True,
            "record_solver_diagnostics": True,
        })
        self.assertEqual(
            "residual_mpc_compute_ms",
            legacy_mapping.timing_budget_metric)
        self.assertAlmostEqual(17.0, legacy_mapping.max_compute_ms)
        self.assertFalse(legacy_mapping.record_stage_timing)
        self.assertFalse(legacy_mapping.record_solver_diagnostics)
        self.assertFalse(legacy_mapping.debug_qp_snapshot_enabled)
        self.assertEqual(0, legacy_mapping.debug_qp_snapshot_every_n)
        self.assertTrue(legacy_mapping.debug_qp_snapshot_event_window_only)
        self.assertEqual(200, legacy_mapping.debug_qp_snapshot_max_rows)
        self.assertEqual(
            "residual_mpc_incremental_compute_ms",
            explicit_mapping.timing_budget_metric)
        self.assertAlmostEqual(17.0, explicit_mapping.max_compute_ms)
        self.assertAlmostEqual(
            40.0,
            explicit_mapping.residual_incremental_budget_ms)
        self.assertTrue(explicit_mapping.record_stage_timing)
        self.assertTrue(explicit_mapping.record_solver_diagnostics)

        config_dir = (
            Path(_sim_root()) /
            "carla-0.9.15" /
            "PythonAPI" /
            "taeho" /
            "suspension_control" /
            "configs")
        legacy_path = (
            config_dir /
            "planning_aware_v5_residual_qp_mpc_shadow.yaml")
        step07_smoke_path = (
            config_dir /
            "planning_aware_v5_residual_qp_mpc_shadow_step07_smoke.yaml")
        legacy_config = suite.build_planning_aware_v5_residual_qp_mpc_config(
            str(legacy_path))
        step07_smoke_config = (
            suite.build_planning_aware_v5_residual_qp_mpc_config(
                str(step07_smoke_path)))

        self.assertEqual(
            "residual_mpc_compute_ms",
            legacy_config.timing_budget_metric)
        self.assertAlmostEqual(40.0, legacy_config.max_compute_ms)
        self.assertEqual(
            "residual_mpc_incremental_compute_ms",
            step07_smoke_config.timing_budget_metric)
        self.assertAlmostEqual(40.0, step07_smoke_config.max_compute_ms)
        self.assertAlmostEqual(
            40.0,
            step07_smoke_config.residual_incremental_budget_ms)
        self.assertTrue(step07_smoke_config.record_stage_timing)
        self.assertTrue(step07_smoke_config.record_solver_diagnostics)
        self.assertFalse(step07_smoke_config.debug_qp_snapshot_enabled)
        self.assertEqual(0, step07_smoke_config.debug_qp_snapshot_every_n)
        self.assertTrue(
            step07_smoke_config.debug_qp_snapshot_event_window_only)
        self.assertEqual(200, step07_smoke_config.debug_qp_snapshot_max_rows)

    def test_debug_capture_config_is_bounded_and_tuning_equivalent(self):
        import transfuser_suspension_control_suite as suite

        config_dir = (
            Path(_sim_root()) /
            "carla-0.9.15" /
            "PythonAPI" /
            "taeho" /
            "suspension_control" /
            "configs")
        normal = suite.build_planning_aware_v5_residual_qp_mpc_config(
            str(config_dir /
                "planning_aware_v5_residual_qp_mpc_shadow_step07_smoke.yaml"))
        debug = suite.build_planning_aware_v5_residual_qp_mpc_config(
            str(config_dir /
                "planning_aware_v5_residual_qp_mpc_shadow_step07_3_debug_capture.yaml"))
        excluded = {
            "config_path",
            "debug_qp_snapshot_enabled",
            "debug_qp_snapshot_every_n",
            "debug_qp_snapshot_event_window_only",
            "debug_qp_snapshot_max_rows",
        }

        self.assertEqual(
            {key: value for key, value in vars(normal).items()
             if key not in excluded},
            {key: value for key, value in vars(debug).items()
             if key not in excluded})
        self.assertTrue(debug.debug_qp_snapshot_enabled)
        self.assertEqual(1, debug.debug_qp_snapshot_every_n)
        self.assertTrue(debug.debug_qp_snapshot_event_window_only)
        self.assertEqual(200, debug.debug_qp_snapshot_max_rows)

    def test_config_builder_stamps_step07_shadow_source_and_strict_flag(self):
        import transfuser_suspension_control_suite as suite

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = _write_artifact(Path(temp_dir) / "artifact")
            config_path = Path(temp_dir) / "shadow_config.yaml"
            config_path.write_text(
                "\n".join((
                    "model_artifact_dir: %s" % artifact_dir,
                    "solver_name: box_projected_gradient",
                    "require_suspension_state_for_projection: false",
                    "",
                )),
                encoding="utf-8")

            config = suite.build_planning_aware_v5_residual_qp_mpc_config(
                str(config_path),
                strict_step07_shadow_validation=True)

        self.assertEqual(str(config_path), config.config_path)
        self.assertEqual(str(artifact_dir), config.model_artifact_dir)
        self.assertTrue(config.strict_step07_shadow_validation)

    def test_startup_diagnostics_report_loaded_config_and_artifact(self):
        import transfuser_suspension_control_suite as suite

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = _write_artifact(Path(temp_dir) / "artifact")
            config_path = Path(temp_dir) / "shadow_config.yaml"
            config_path.write_text(
                "\n".join((
                    "model_artifact_dir: %s" % artifact_dir,
                    "solver_name: box_projected_gradient",
                    "require_suspension_state_for_projection: false",
                    "",
                )),
                encoding="utf-8")
            config = suite.build_planning_aware_v5_residual_qp_mpc_config(
                str(config_path),
                strict_step07_shadow_validation=True)
            controller = PlanningAwareV5ResidualQpMpcShadowController(config)

            output = controller.compute(_context(roll=1.0))
            diagnostics = output.diagnostics

        self.assertEqual(
            str(config_path),
            diagnostics["residual_qp_mpc_shadow_config_path"])
        self.assertEqual(
            1,
            diagnostics["residual_qp_mpc_shadow_config_exists"])
        self.assertEqual(
            str(artifact_dir),
            diagnostics["residual_mpc_model_artifact_dir"])
        self.assertEqual(
            1,
            diagnostics["residual_mpc_model_artifact_dir_exists"])
        self.assertEqual(
            1,
            diagnostics["residual_mpc_model_artifact_loaded"])
        self.assertEqual(
            "box_projected_gradient",
            diagnostics["residual_mpc_solver_backend"])
        self.assertEqual(
            1,
            diagnostics["residual_mpc_strict_step07_shadow_validation"])

    def test_startup_diagnostics_report_fallback_without_strict_failure(self):
        missing_config = os.path.join(
            tempfile.gettempdir(),
            "missing_residual_qp_mpc_shadow_config.yaml")
        controller = PlanningAwareV5ResidualQpMpcShadowController(
            PlanningAwareV5ResidualQpMpcConfig(
                config_path=missing_config,
                model_artifact_dir="",
                solver_name="box_projected_gradient",
                strict_step07_shadow_validation=False,
                require_suspension_state_for_projection=False))

        output = controller.compute(_context(roll=1.0))
        diagnostics = output.diagnostics

        self.assertEqual("phase_a_fallback", diagnostics["residual_mpc_mode"])
        self.assertEqual(
            "model_unavailable",
            diagnostics["residual_mpc_fallback_reason"])
        self.assertEqual(
            missing_config,
            diagnostics["residual_qp_mpc_shadow_config_path"])
        self.assertEqual(
            0,
            diagnostics["residual_qp_mpc_shadow_config_exists"])
        self.assertEqual("", diagnostics["residual_mpc_model_artifact_dir"])
        self.assertEqual(
            0,
            diagnostics["residual_mpc_model_artifact_dir_exists"])
        self.assertEqual(
            0,
            diagnostics["residual_mpc_model_artifact_loaded"])
        self.assertEqual(
            "box_projected_gradient",
            diagnostics["residual_mpc_solver_backend"])
        self.assertEqual(
            0,
            diagnostics["residual_mpc_strict_step07_shadow_validation"])

    def test_strict_step07_shadow_raises_when_model_artifact_is_missing(self):
        controller = PlanningAwareV5ResidualQpMpcShadowController(
            PlanningAwareV5ResidualQpMpcConfig(
                config_path="/tmp/missing_residual_qp_mpc_shadow_config.yaml",
                model_artifact_dir="",
                solver_name="box_projected_gradient",
                strict_step07_shadow_validation=True,
                require_suspension_state_for_projection=False))

        with self.assertRaisesRegex(RuntimeError, "strict Step07 residual QP-MPC"):
            controller.compute(_context(roll=1.0))

    def test_suite_strict_startup_preflight_loads_artifact_before_route(self):
        import transfuser_suspension_control_suite as suite

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = _write_artifact(Path(temp_dir) / "artifact")
            config_path = Path(temp_dir) / "shadow_config.yaml"
            config_path.write_text(
                "\n".join((
                    "model_artifact_dir: %s" % artifact_dir,
                    "solver_name: box_projected_gradient",
                    "require_suspension_state_for_projection: false",
                    "",
                )),
                encoding="utf-8")
            args = types.SimpleNamespace(
                strict_step07_residual_shadow_validation=True,
                planning_aware_v5_residual_qp_mpc_shadow_config=str(config_path),
            )
            scenarios = suite.selected_scenarios(
                "planning_aware_v5_residual_qp_mpc_shadow")

            suite.require_step07_residual_shadow_strict_startup_preflight(
                args,
                scenarios)

    def test_suite_strict_startup_negative_artifact_matrix(self):
        import transfuser_suspension_control_suite as suite

        scenarios = suite.selected_scenarios(
            "planning_aware_v5_residual_qp_mpc_shadow")

        def rewrite_json(path, update):
            payload = json.loads(path.read_text(encoding="utf-8"))
            update(payload)
            path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")

        mutators = {
            "nonexistent_artifact_directory": lambda artifact_dir: None,
            "missing_manifest_json": lambda artifact_dir: (
                artifact_dir / "manifest.json").unlink(),
            "missing_model_npz": lambda artifact_dir: (
                artifact_dir / "model.npz").unlink(),
            "corrupt_model_npz": lambda artifact_dir: (
                (artifact_dir / "model.npz").write_bytes(b"not-a-zip-model"),
                rewrite_json(
                    artifact_dir / "manifest.json",
                    lambda payload: payload["hashes"].update({
                        "model_npz_sha256": hashlib.sha256(
                            b"not-a-zip-model").hexdigest(),
                    }))),
            "missing_preprocessing_json": lambda artifact_dir: (
                artifact_dir / "preprocessing.json").unlink(),
            "preprocessing_schema_mismatch": lambda artifact_dir: rewrite_json(
                artifact_dir / "preprocessing.json",
                lambda payload: payload.update({
                    "schema_version": "residual_mpc_preprocessing_v999",
                })),
            "manifest_hash_mismatch": lambda artifact_dir: rewrite_json(
                artifact_dir / "manifest.json",
                lambda payload: payload["hashes"].update({
                    "model_npz_sha256": "0" * 64,
                })),
        }

        for case, mutate in mutators.items():
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                artifact_dir = root / "artifact"
                if case != "nonexistent_artifact_directory":
                    _write_artifact(artifact_dir)
                mutate(artifact_dir)
                config_path = root / "shadow_config.yaml"
                config_path.write_text(
                    "\n".join((
                        "model_artifact_dir: %s" % artifact_dir,
                        "solver_name: osqp",
                        "",
                    )),
                    encoding="utf-8")
                args = types.SimpleNamespace(
                    strict_step07_residual_shadow_validation=True,
                    planning_aware_v5_residual_qp_mpc_shadow_config=(
                        str(config_path)),
                )

                with self.assertRaisesRegex(
                        RuntimeError,
                        "strict startup validation failed"):
                    suite.require_step07_residual_shadow_strict_startup_preflight(
                        args,
                        scenarios)

    def test_suite_registration_and_default_configs(self):
        import transfuser_suspension_control_suite as suite

        active = suite.selected_scenarios("planning_aware_v5_residual_qp_mpc")
        shadow = suite.selected_scenarios(
            "planning_aware_v5_residual_qp_mpc_shadow")
        self.assertEqual(
            "S34_planning_aware_v5_residual_qp_mpc",
            active[0]["name"])
        self.assertEqual(
            "S35_planning_aware_v5_residual_qp_mpc_shadow",
            shadow[0]["name"])

        args = suite.build_arg_parser().parse_args([])
        self.assertIsInstance(
            suite.make_controller("planning_aware_v5_residual_qp_mpc", args),
            PlanningAwareV5ResidualQpMpcController)
        self.assertIsInstance(
            suite.make_controller(
                "planning_aware_v5_residual_qp_mpc_shadow",
                args),
            PlanningAwareV5ResidualQpMpcShadowController)
        missing = [
            field for field in RESIDUAL_QP_MPC_DIAGNOSTIC_FIELDS
            if field not in suite.DIAGNOSTIC_FIELDS
        ]
        self.assertEqual([], missing)

    def test_step07_and_route_ablation_dry_run_mapping(self):
        sim_root = _sim_root()
        matrix_script = os.path.join(sim_root, "scripts", "run_lead_step07_matrix.sh")
        scenarios = (
            "LEAD_planning_aware_v5_residual_qp_mpc,"
            "LEAD_planning_aware_v5_residual_qp_mpc_shadow")
        with tempfile.TemporaryDirectory() as temp_dir:
            readiness_summary = _write_step07_readiness_summary(Path(temp_dir))
            result = subprocess.run(
                [
                    "bash",
                    matrix_script,
                    "--mode",
                    "smoke",
                    "--seeds",
                    "111",
                    "--scenarios",
                    scenarios,
                    "--dry-run",
                    "--skip-preflight",
                    "--skip-summary",
                    "--no-restart-carla-per-run",
                    "--residual-mpc-readiness-summary",
                    str(readiness_summary),
                ],
                cwd=sim_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=60)
        self.assertEqual(
            0,
            result.returncode,
            msg=result.stdout[-4000:] + result.stderr[-4000:])
        self.assertIn(
            "scenario=LEAD_planning_aware_v5_residual_qp_mpc kind=sidecar",
            result.stdout)
        self.assertIn(
            "--scenarios planning_aware_v5_residual_qp_mpc",
            result.stdout)
        self.assertIn(
            "--scenarios planning_aware_v5_residual_qp_mpc_shadow",
            result.stdout)
        self.assertEqual(2, result.stdout.count("--planning-provider jsonl"))

        route_script = os.path.join(
            sim_root,
            "scripts",
            "suspension",
            "run_suspension_route_ablation.sh")
        route = os.path.join(
            sim_root,
            "e2e_models",
            "carla_garage",
            "leaderboard",
            "data",
            "suspension_routes",
            "planning_aware_v2",
            "town04_r18_wp50_72_outer_scurve_ar_noscenario.xml")
        with tempfile.TemporaryDirectory() as output_dir:
            readiness_summary = _write_step07_readiness_summary(Path(output_dir))
            route_result = subprocess.run(
                [
                    "bash",
                    route_script,
                    "--route",
                    route,
                    "--seed",
                    "111",
                    "--controllers",
                    "planning_aware_v5_residual_qp_mpc",
                    "planning_aware_v5_residual_qp_mpc_shadow",
                    "--repetitions",
                    "1",
                    "--dry-run",
                    "--no-analyze",
                    "--output-dir",
                    output_dir,
                    "--residual-mpc-readiness-summary",
                    str(readiness_summary),
                ],
                cwd=sim_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=120)
            self.assertEqual(
                0,
                route_result.returncode,
                msg=route_result.stdout[-4000:] + route_result.stderr[-4000:])
            with open(os.path.join(output_dir, "run_plan.csv"), newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))

        by_controller = {row["controller"]: row for row in rows}
        self.assertEqual(
            "LEAD_planning_aware_v5_residual_qp_mpc",
            by_controller["planning_aware_v5_residual_qp_mpc"]["scenario_key"])
        self.assertEqual(
            "LEAD_planning_aware_v5_residual_qp_mpc_shadow",
            by_controller["planning_aware_v5_residual_qp_mpc_shadow"][
                "scenario_key"])

    def test_step07_residual_solver_preflight_precedes_carla_start(self):
        sim_root = _sim_root()
        matrix_script = os.path.join(sim_root, "scripts", "run_lead_step07_matrix.sh")
        with tempfile.TemporaryDirectory() as temp_dir:
            readiness_summary = _write_step07_readiness_summary(Path(temp_dir))
            result = subprocess.run(
                [
                    "bash",
                    matrix_script,
                    "--mode",
                    "smoke",
                    "--seeds",
                    "111",
                    "--scenarios",
                    "LEAD_planning_aware_v5_residual_qp_mpc_shadow",
                    "--dry-run",
                    "--skip-preflight",
                    "--skip-summary",
                    "--start-carla",
                    "--no-restart-carla-per-run",
                    "--residual-mpc-readiness-summary",
                    str(readiness_summary),
                ],
                cwd=sim_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=60)

        self.assertEqual(
            0,
            result.returncode,
            msg=result.stdout[-4000:] + result.stderr[-4000:])
        preflight_index = result.stdout.find("--residual-solver-preflight-only")
        carla_start_index = result.stdout.find("up -d carla")
        self.assertGreaterEqual(preflight_index, 0)
        self.assertGreater(carla_start_index, preflight_index)


def _write_step07_readiness_summary(temp_dir: Path) -> Path:
    sim_root = _sim_root()
    if sim_root not in sys.path:
        sys.path.insert(0, sim_root)
    from scripts.suspension.tests.test_residual_mpc_step07_1_readiness import (
        _summary,
    )

    path = temp_dir / "step07_readiness_summary.json"
    path.write_text(json.dumps(_summary()), encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()
