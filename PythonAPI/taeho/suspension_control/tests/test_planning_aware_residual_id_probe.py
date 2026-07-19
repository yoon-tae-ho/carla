"""CARLA-free checks for the planning-aware v5 residual ID probe."""

from __future__ import annotations

import math
import unittest

from suspension_control.controllers.base import (
    ControllerContext,
    PlanningInfo,
    VehicleState,
)
from suspension_control.controllers.planning_aware_residual_id_probe import (
    PLANNING_AWARE_V5_RESIDUAL_ID_PROBE_VERSION,
    RESIDUAL_ID_PROBE_TELEMETRY_SCHEMA_VERSION,
    PlanningAwareV5ResidualIdProbeConfig,
    PlanningAwareV5ResidualIdProbeController,
    _cycle_modal_values,
    sequence_hash,
)


_LABELS = ("fl", "fr", "rl", "rr")


class _FakeWheel:

    def __init__(
        self,
        index,
        velocity,
        field_valid=True,
        velocity_valid=True,
        contact_valid=True,
        wheel_in_air=False,
    ):
        self.wheel_index_raw = index
        self.wheel_name_canonical = ("FL", "FR", "RL", "RR")[index]
        self.raw_suspension_offset_m = 0.01
        self.suspension_compression_m = 0.01
        self.suspension_travel_m = 0.01
        self.suspension_velocity_mps = velocity
        self.normalized_travel = 0.5
        self.contact_valid = contact_valid
        self.wheel_in_air = wheel_in_air
        self.field_valid = field_valid
        self.velocity_valid = velocity_valid


class _FakeSuspensionState:

    def __init__(
        self,
        velocities=(-0.5, 0.5, -0.5, 0.5),
        state_valid=True,
        velocity_valid=True,
        failure_reason="",
    ):
        self.state_valid = state_valid
        self.velocity_valid = velocity_valid
        self.compression_convention_validated = True
        self.state_source = "fake"
        self.failure_reason = failure_reason
        self.wheels = tuple(
            _FakeWheel(index, velocity)
            for index, velocity in enumerate(velocities))


class _FakeNativeWheel:
    spring_strength = 35000.0
    spring_damper_rate = 4500.0


class _FakeNativeSuspension:
    wheels = (_FakeNativeWheel(),) * 4


def _state(
    frame=100,
    speed=12.0,
    elapsed_seconds=5.0,
    dt=0.05,
    roll=1.0,
    pitch=0.5,
    roll_rate=2.0,
    pitch_rate=1.0,
    yaw_rate=3.0,
    local_ax=0.1,
    local_ay=0.2,
    ax=0.3,
    ay=0.4,
):
    return VehicleState(
        frame=frame,
        speed=speed,
        elapsed_seconds=elapsed_seconds,
        dt=dt,
        roll=roll,
        pitch=pitch,
        roll_rate=roll_rate,
        pitch_rate=pitch_rate,
        yaw_rate=yaw_rate,
        local_ax=local_ax,
        local_ay=local_ay,
        ax=ax,
        ay=ay)


def _planning(**overrides):
    values = {
        "available": True,
        "source": "lead_tfv6",
        "frame": 100,
        "horizon_dt": 0.1,
        "target_speed": (12.0, 12.0, 12.0, 12.0),
        "curvature": (0.0, 0.01, 0.0, -0.01),
        "predicted_ax": (0.0, 0.0, 0.0, 0.0),
        "predicted_ay": (0.0, 1.44, 0.0, -1.44),
        "brake": (0.0, 0.0, 0.0, 0.0),
        "throttle": (0.2, 0.2, 0.2, 0.2),
        "metadata": {
            "valid_prediction": True,
            "trajectory_source": "pred_route",
            "speed_semantics": (
                "pred_target_speed_scalar_flattened_not_time_sequence"),
            "control_semantics": "current_final_control_single_sample",
            "validity_reason": "valid_prediction_with_preview",
        },
    }
    values.update(overrides)
    return PlanningInfo(**values)


def _context(
    state=None,
    planning=None,
    suspension_state=None,
    suspension_state_valid=True,
    dt=0.05,
):
    state_value = state if state is not None else _state(dt=dt)
    return ControllerContext(
        state=state_value,
        planning=planning if planning is not None else _planning(),
        native_suspension=_FakeNativeSuspension(),
        current_suspension=_FakeNativeSuspension(),
        suspension_state=(
            suspension_state
            if suspension_state is not None else _FakeSuspensionState()),
        suspension_state_valid=suspension_state_valid,
        native_spring_strength_by_wheel=(35000.0,) * 4,
        native_damper_rate_by_wheel=(4500.0,) * 4,
        dt=dt)


def _dampers(output):
    return tuple(wheel.damper_scale for wheel in output.command.wheels)


def _springs(output):
    return tuple(wheel.spring_scale for wheel in output.command.wheels)


def _probe_config(**overrides):
    values = {
        "warmup_s": 0.0,
        "active_duration_s": 0.0,
        "cooldown_s": 0.0,
        "block_duration_s": 0.20,
        "speed_gate_min_mps": 0.0,
        "speed_gate_max_mps": 40.0,
        "max_dt_s": 0.20,
        "min_command_margin_scale": 0.0,
        "require_valid_planning": True,
        "require_suspension_state_for_projection": True,
        "max_rate_up_scale_per_s": 10.0,
        "max_rate_down_scale_per_s": 10.0,
        "excitation_seed": 111,
        "run_id": "unit-run",
        "route_id": "unit-route",
        "ad_seed": "111",
    }
    values.update(overrides)
    return PlanningAwareV5ResidualIdProbeConfig(**values)


class PlanningAwareV5ResidualIdProbeControllerTest(unittest.TestCase):

    def test_config_identity_and_sequence_hash_are_deterministic(self):
        config = _probe_config()
        same = _probe_config()
        different = _probe_config(excitation_seed=112)

        self.assertEqual(
            PLANNING_AWARE_V5_RESIDUAL_ID_PROBE_VERSION,
            config.controller_version)
        self.assertEqual(
            RESIDUAL_ID_PROBE_TELEMETRY_SCHEMA_VERSION,
            config.telemetry_schema_version)
        self.assertEqual(sequence_hash(config), sequence_hash(same))
        self.assertNotEqual(sequence_hash(config), sequence_hash(different))

    def test_full_walsh_cycle_is_zero_mean_and_orthogonal(self):
        config = _probe_config()
        cycle = [_cycle_modal_values(config, index) for index in range(8)]

        for mode_index in range(4):
            mode_values = [row[mode_index] for row in cycle]
            self.assertAlmostEqual(0.0, sum(mode_values), places=12)

        for left in range(4):
            for right in range(left + 1, 4):
                dot = sum(row[left] * row[right] for row in cycle)
                self.assertAlmostEqual(0.0, dot, places=12)

    def test_compute_logs_phase_a_shadow_and_full_applied_residual_chain(self):
        controller = PlanningAwareV5ResidualIdProbeController(_probe_config())

        output = controller.compute(_context())
        diagnostics = output.diagnostics

        self.assertEqual((1.0, 1.0, 1.0, 1.0), _springs(output))
        self.assertEqual(1.0, diagnostics["residual_mpc_spring_scale_invariant"])
        self.assertEqual(
            RESIDUAL_ID_PROBE_TELEMETRY_SCHEMA_VERSION,
            diagnostics["residual_mpc_schema_version"])
        self.assertEqual(
            PLANNING_AWARE_V5_RESIDUAL_ID_PROBE_VERSION,
            diagnostics["residual_mpc_controller_version"])
        self.assertEqual("id_probe", diagnostics["residual_mpc_mode"])
        self.assertEqual("not_used_id_probe", diagnostics[
            "residual_mpc_solver_status"])
        self.assertEqual(1, diagnostics["residual_mpc_phase_a_commit_match"])
        self.assertEqual(
            "pred_target_speed_scalar_flattened_not_time_sequence",
            diagnostics["residual_mpc_preview_target_speed_semantics"])
        self.assertEqual(
            "current_final_control_single_sample",
            diagnostics["residual_mpc_preview_control_semantics"])
        self.assertEqual(
            "valid_prediction_with_preview",
            diagnostics["residual_mpc_preview_validity_reason"])

        for label in _LABELS:
            final_damper = diagnostics[
                "residual_mpc_final_damper_%s" % label]
            self.assertAlmostEqual(
                final_damper,
                _dampers(output)[_LABELS.index(label)])
            self.assertTrue(math.isfinite(diagnostics[
                "residual_mpc_phase_a_shadow_damper_%s" % label]))
            for prefix in (
                    "residual_mpc_requested_residual",
                    "residual_mpc_bounded_residual",
                    "residual_mpc_rate_limited_residual",
                    "residual_mpc_projected_residual",
                    "residual_mpc_final_residual"):
                self.assertIn("%s_%s" % (prefix, label), diagnostics)
                self.assertTrue(math.isfinite(diagnostics[
                    "%s_%s" % (prefix, label)]))

        requested = [
            diagnostics["residual_mpc_requested_residual_%s" % label]
            for label in _LABELS
        ]
        self.assertTrue(any(abs(value) > 1.0e-9 for value in requested))
        self.assertTrue(math.isfinite(diagnostics["residual_mpc_roll_deg_raw"]))
        self.assertTrue(math.isfinite(diagnostics["residual_mpc_roll_deg_filt"]))

    def test_suppressed_probe_applies_phase_a_shadow_without_residual(self):
        config = _probe_config(speed_gate_min_mps=1.0)
        controller = PlanningAwareV5ResidualIdProbeController(config)
        context = _context(state=_state(speed=0.0))

        output = controller.compute(context)
        diagnostics = output.diagnostics

        self.assertEqual("phase_a_suppressed", diagnostics["residual_mpc_mode"])
        self.assertIn("speed_below_gate", diagnostics[
            "residual_mpc_suppression_reason"])
        self.assertEqual(0, diagnostics["residual_mpc_excitation_active"])
        for label in _LABELS:
            self.assertAlmostEqual(
                0.0,
                diagnostics["residual_mpc_requested_residual_%s" % label],
                places=12)
            self.assertTrue(math.isfinite(diagnostics[
                "residual_mpc_final_residual_%s" % label]))
            self.assertTrue(math.isfinite(diagnostics[
                "residual_mpc_final_damper_%s" % label]))
        self.assertEqual((1.0, 1.0, 1.0, 1.0), _springs(output))

    def test_rate_limit_caps_selected_damper_change(self):
        config = _probe_config(
            excitation_amplitude_by_mode=(0.50, 0.50, 0.50, 0.50),
            max_rate_up_scale_per_s=0.10,
            max_rate_down_scale_per_s=0.10,
        )
        controller = PlanningAwareV5ResidualIdProbeController(config)
        controller.previous_final_damper_scales = [1.0, 1.0, 1.0, 1.0]
        context = _context(dt=0.05)

        output = controller.compute(context)
        allowed_delta = 0.10 * context.dt

        for damper in _dampers(output):
            self.assertLessEqual(abs(damper - 1.0), allowed_delta + 1.0e-12)
            self.assertGreaterEqual(damper, config.min_damper_scale)
            self.assertLessEqual(damper, config.max_damper_scale)


if __name__ == "__main__":
    unittest.main()
