"""CARLA-free checks for the state-based skyhook estimator dry-run."""

from __future__ import annotations

import unittest

from suspension_control.controllers.base import ControllerContext, VehicleState
from suspension_control.controllers.estimators import (
    SkyhookEstimatorDryRunController,
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

    def __init__(self, velocities=(-0.1, -0.1, -0.1, -0.1)):
        self.state_valid = True
        self.velocity_valid = True
        self.compression_convention_validated = False
        self.state_source = "fake"
        self.failure_reason = ""
        self.wheels = tuple(
            _FakeWheel(index, velocity)
            for index, velocity in enumerate(velocities))


class _FakeNativeWheel:

    spring_strength = 35000.0
    spring_damper_rate = 4500.0


class _FakeNativeSuspension:

    wheels = (_FakeNativeWheel(),) * 4


def _context(
    suspension_state=None,
    suspension_state_valid=True,
    invalid_reason="",
):
    return ControllerContext(
        state=VehicleState(vz=1.0, dt=0.05),
        native_suspension=_FakeNativeSuspension(),
        current_suspension=_FakeNativeSuspension(),
        suspension_state=(
            suspension_state
            if suspension_state is not None else _FakeSuspensionState()),
        suspension_state_valid=suspension_state_valid,
        suspension_state_invalid_reason=invalid_reason,
        native_spring_strength_by_wheel=(35000.0,) * 4,
        native_damper_rate_by_wheel=(4500.0,) * 4,
        dt=0.05)


def _springs(output):
    return tuple(wheel.spring_scale for wheel in output.command.wheels)


def _dampers(output):
    return tuple(wheel.damper_scale for wheel in output.command.wheels)


class SkyhookEstimatorDryRunControllerTest(unittest.TestCase):

    def test_valid_state_logs_candidates_but_returns_identity(self):
        controller = SkyhookEstimatorDryRunController()

        output = controller.compute(_context())

        self.assertEqual((1.0, 1.0, 1.0, 1.0), _springs(output))
        self.assertEqual((1.0, 1.0, 1.0, 1.0), _dampers(output))
        self.assertEqual(
            "skyhook_estimator_dryrun",
            output.diagnostics["controller"])
        self.assertEqual(
            "skyhook_estimator_dryrun_v1",
            output.diagnostics["controller_version"])
        self.assertEqual(1, output.diagnostics["skyhook_dryrun_command_identity"])
        self.assertEqual(1, output.diagnostics["suspension_state_valid"])
        self.assertEqual(0, output.diagnostics["identity_fallback_this_tick"])
        self.assertEqual(1.0, output.diagnostics["hard_mode_ratio_candidate_A"])
        self.assertEqual(1.0, output.diagnostics["soft_mode_ratio_candidate_B"])
        self.assertEqual(
            1.30,
            output.diagnostics["proposed_damper_scale_candidate_A_fl"])
        self.assertEqual(
            0.80,
            output.diagnostics["proposed_damper_scale_candidate_B_fl"])
        self.assertEqual(4500.0, output.diagnostics["native_damper_rate_fl"])

    def test_invalid_state_falls_back_to_identity_diagnostics(self):
        controller = SkyhookEstimatorDryRunController()

        output = controller.compute(_context(
            suspension_state_valid=False,
            invalid_reason="get_suspension_state_missing"))

        self.assertEqual((1.0, 1.0, 1.0, 1.0), _springs(output))
        self.assertEqual((1.0, 1.0, 1.0, 1.0), _dampers(output))
        self.assertEqual(0, output.diagnostics["suspension_state_valid"])
        self.assertEqual(1, output.diagnostics["identity_fallback_this_tick"])
        self.assertEqual(
            "get_suspension_state_missing",
            output.diagnostics["identity_fallback_reason"])
        self.assertEqual("", output.diagnostics["hard_mode_ratio_candidate_A"])


if __name__ == "__main__":
    unittest.main()
