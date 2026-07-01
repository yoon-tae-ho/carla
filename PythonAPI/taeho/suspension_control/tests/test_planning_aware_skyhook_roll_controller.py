"""CARLA-free checks for fair planning-aware skyhook_roll composition."""

from __future__ import annotations

import math
import unittest

from suspension_control.controllers.base import (
    ControllerContext,
    PlanningInfo,
    VehicleState,
)
from suspension_control.controllers.planning_aware_skyhook_roll import (
    PlanningAwareSkyhookRollConfig,
    PlanningAwareSkyhookRollController,
)
from suspension_control.controllers.skyhook_roll import (
    SkyhookRollConfig,
    SkyhookRollController,
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

    def __init__(self, velocities=(-0.5, -0.5, -0.5, -0.5)):
        self.state_valid = True
        self.velocity_valid = True
        self.compression_convention_validated = True
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


def _roll_state(frame=100, roll_rate_deg_s=20.0, local_ay=6.0, roll_deg=4.0):
    return VehicleState(
        frame=frame,
        elapsed_seconds=5.0,
        dt=0.05,
        speed=12.0,
        roll=roll_deg,
        roll_rate=roll_rate_deg_s,
        local_ay=local_ay)


def _state_matching_roll_extension(roll_rate_deg_s=20.0):
    roll_rate = math.radians(roll_rate_deg_s)
    y_left = -0.85
    y_right = 0.85
    v_left = roll_rate * y_left
    v_right = roll_rate * y_right
    return _FakeSuspensionState(
        velocities=(-v_left, -v_right, -v_left, -v_right))


def _context(state=None, planning=None):
    return ControllerContext(
        state=state if state is not None else _roll_state(),
        planning=planning if planning is not None else PlanningInfo.empty(),
        native_suspension=_FakeNativeSuspension(),
        current_suspension=_FakeNativeSuspension(),
        suspension_state=_state_matching_roll_extension(),
        suspension_state_valid=True,
        native_spring_strength_by_wheel=(35000.0,) * 4,
        native_damper_rate_by_wheel=(4500.0,) * 4,
        dt=0.05)


def _common_config(**overrides):
    values = {
        "max_damper_delta_per_step": 10.0,
        "sprung_velocity_deadband": 0.0,
        "rel_velocity_deadband": 0.0,
        "product_deadband": 0.0,
        "preview_tau_rise": 0.0,
        "preview_tau_fall": 0.0,
    }
    values.update(overrides)
    return values


def _skyhook_config(values):
    allowed = {item.name for item in SkyhookRollConfig.__dataclass_fields__.values()}
    return SkyhookRollConfig(**{
        key: value for key, value in values.items() if key in allowed
    })


def _planning_config(values):
    return PlanningAwareSkyhookRollConfig(**values)


def _zero_planning():
    return PlanningInfo(
        available=True,
        source="jsonl",
        frame=100,
        horizon_dt=0.1,
        target_speed=(12.0, 12.0, 12.0, 12.0),
        curvature=(0.0, 0.0, 0.0, 0.0),
        predicted_ay=(0.0, 0.0, 0.0, 0.0),
        predicted_ax=(0.0, 0.0, 0.0, 0.0),
        brake=(0.0, 0.0, 0.0, 0.0),
        steer=(0.0, 0.0, 0.0, 0.0),
        metadata={"valid_prediction": True})


def _high_planning():
    return PlanningInfo(
        available=True,
        source="jsonl",
        frame=100,
        horizon_dt=0.1,
        target_speed=(12.0, 12.0, 12.0, 12.0),
        curvature=(0.08, 0.09, 0.08, 0.07),
        predicted_ay=(6.0, 6.5, 6.2, 5.8),
        predicted_ax=(0.0, 0.0, 0.0, 0.0),
        brake=(0.0, 0.0, 0.0, 0.0),
        steer=(0.30, 0.30, 0.30, 0.30),
        metadata={"valid_prediction": True})


def _dampers(output):
    return tuple(wheel.damper_scale for wheel in output.command.wheels)


def _springs(output):
    return tuple(wheel.spring_scale for wheel in output.command.wheels)


class PlanningAwareSkyhookRollControllerTest(unittest.TestCase):

    def test_empty_preview_is_exact_skyhook_roll_command(self):
        values = _common_config()
        skyhook_roll = SkyhookRollController(_skyhook_config(values))
        planning_aware = PlanningAwareSkyhookRollController(
            _planning_config(values))
        context = _context(planning=PlanningInfo.empty())

        expected = skyhook_roll.compute(context)
        actual = planning_aware.compute(context)

        self.assertTrue(actual.command.is_close(expected.command))
        self.assertEqual(
            "planning_unavailable",
            actual.diagnostics["planning_aware_fallback_reason"])
        self.assertEqual(0.0, actual.diagnostics["planning_aware_risk_raw"])
        self.assertEqual(
            _dampers(expected),
            tuple(actual.diagnostics[
                "planning_aware_feedback_damper_%s" % label]
                for label in ("fl", "fr", "rl", "rr")))

    def test_zero_preview_is_exact_skyhook_roll_command(self):
        values = _common_config()
        skyhook_roll = SkyhookRollController(_skyhook_config(values))
        planning_aware = PlanningAwareSkyhookRollController(
            _planning_config(values))
        context = _context(planning=_zero_planning())

        expected = skyhook_roll.compute(context)
        actual = planning_aware.compute(context)

        self.assertTrue(actual.command.is_close(expected.command))
        self.assertEqual(1, actual.diagnostics["planning_aware_valid"])
        self.assertEqual("none", actual.diagnostics[
            "planning_aware_fallback_reason"])
        self.assertEqual(0.0, actual.diagnostics["planning_aware_risk_raw"])
        self.assertEqual((1.0, 1.0, 1.0, 1.0), _springs(actual))

    def test_high_preview_command_is_bounded_and_rate_limited(self):
        values = _common_config(
            max_damper_delta_per_step=0.004,
            max_preview_damper_extra=0.035)
        controller = PlanningAwareSkyhookRollController(_planning_config(values))

        output = controller.compute(_context(planning=_high_planning()))

        self.assertEqual((1.0, 1.0, 1.0, 1.0), _springs(output))
        self.assertEqual(1, output.diagnostics["planning_aware_valid"])
        self.assertGreater(output.diagnostics["planning_aware_risk_raw"], 0.0)
        self.assertGreater(
            max(output.diagnostics[
                "planning_feedforward_damper_add_%s" % label]
                for label in ("fl", "fr", "rl", "rr")),
            0.0)
        self.assertEqual(1, output.diagnostics["planning_aware_rate_limited_any"])
        for damper in _dampers(output):
            self.assertGreaterEqual(damper, 0.75)
            self.assertLessEqual(damper, 1.004 + 1.0e-12)


if __name__ == "__main__":
    unittest.main()
