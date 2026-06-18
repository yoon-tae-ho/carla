"""CARLA-free checks for the skyhook + roll baseline controller."""

from __future__ import annotations

import math
import unittest

from suspension_control.controllers.base import ControllerContext, VehicleState
from suspension_control.controllers.skyhook_roll import (
    SkyhookRollConfig,
    SkyhookRollController,
)


class _FakeNativeSuspension:

    def __init__(self, wheel_count):
        self.wheels = tuple(object() for _ in range(wheel_count))


def _context(state=None, dt=0.05, native_suspension=None):
    state = state if state is not None else VehicleState(dt=dt)
    return ControllerContext(
        state=state,
        dt=dt,
        native_suspension=native_suspension)


def _springs(output):
    return tuple(wheel.spring_scale for wheel in output.command.wheels)


def _dampers(output):
    return tuple(wheel.damper_scale for wheel in output.command.wheels)


class SkyhookRollControllerTest(unittest.TestCase):

    def test_zero_state_returns_base_scales(self):
        controller = SkyhookRollController()

        output = controller.compute(_context())

        self.assertEqual((1.0, 1.0, 1.0, 1.0), _springs(output))
        self.assertEqual((1.0, 1.0, 1.0, 1.0), _dampers(output))
        self.assertEqual("skyhook_roll", output.diagnostics["controller"])
        self.assertEqual("", output.diagnostics["skyhook_roll_fallback_reason"])

    def test_heave_skyhook_only_increases_damping(self):
        controller = SkyhookRollController(SkyhookRollConfig(
            max_damper_delta_per_step=1.0,
            max_spring_delta_per_step=1.0))

        output = controller.compute(_context(VehicleState(vz=1.0)))

        self.assertTrue(all(value > 1.0 for value in _dampers(output)))
        self.assertEqual((1.0, 1.0, 1.0, 1.0), _springs(output))
        self.assertGreater(output.diagnostics["skyhook_max_activity"], 0.0)

    def test_roll_rate_adds_damping_without_spring_when_no_roll_or_lat_accel(self):
        controller = SkyhookRollController(SkyhookRollConfig(
            max_damper_delta_per_step=1.0,
            max_spring_delta_per_step=1.0))

        output = controller.compute(_context(VehicleState(roll_rate=30.0)))

        self.assertGreater(max(_dampers(output)), 1.0)
        self.assertEqual((1.0, 1.0, 1.0, 1.0), _springs(output))
        self.assertGreater(
            output.diagnostics["skyhook_roll_damping_activity"],
            0.0)
        self.assertEqual(
            0.0,
            output.diagnostics["skyhook_roll_stiffness_activity"])

    def test_lateral_acceleration_keeps_spring_frozen_and_biases_outer_damping(self):
        controller = SkyhookRollController(SkyhookRollConfig(
            nominal_front_roll_distribution=0.5,
            max_damper_delta_per_step=1.0,
            max_spring_delta_per_step=1.0))

        output = controller.compute(_context(VehicleState(local_ay=6.0)))
        springs = _springs(output)
        dampers = _dampers(output)

        self.assertEqual((1.0, 1.0, 1.0, 1.0), springs)
        self.assertGreaterEqual(dampers[0], dampers[1])
        self.assertGreaterEqual(dampers[2], dampers[3])
        self.assertEqual(-1.0, output.diagnostics["skyhook_roll_outer_side_sign"])
        self.assertGreater(
            output.diagnostics["skyhook_roll_side_weight_fl"],
            output.diagnostics["skyhook_roll_side_weight_fr"])
        self.assertGreater(
            output.diagnostics["skyhook_roll_damper_add_fl"],
            output.diagnostics["skyhook_roll_damper_add_fr"])

    def test_clamp_and_rate_limit_bound_output(self):
        controller = SkyhookRollController(SkyhookRollConfig(
            max_damper_delta_per_step=0.01,
            max_spring_delta_per_step=0.005))

        output = controller.compute(_context(VehicleState(
            vz=1000.0,
            roll=45.0,
            roll_rate=1000.0,
            pitch_rate=1000.0,
            local_ay=1000.0)))

        for damper in _dampers(output):
            self.assertLessEqual(damper, 1.01)
            self.assertGreaterEqual(damper, 0.99)
        for spring in _springs(output):
            self.assertEqual(1.0, spring)

        unconstrained = SkyhookRollController(SkyhookRollConfig(
            max_damper_delta_per_step=10.0,
            max_spring_delta_per_step=10.0))
        clamped = unconstrained.compute(_context(VehicleState(
            vz=1000.0,
            roll=45.0,
            roll_rate=1000.0,
            pitch_rate=1000.0,
            local_ay=1000.0)))
        self.assertTrue(all(value <= 1.40 for value in _dampers(clamped)))
        self.assertEqual((1.0, 1.0, 1.0, 1.0), _springs(clamped))

    def test_spring_stays_frozen_even_if_roll_spring_config_is_enabled(self):
        controller = SkyhookRollController(SkyhookRollConfig(
            base_spring_scale=1.10,
            enable_roll_spring_control=True,
            roll_spring_gain=10.0,
            max_roll_spring_add=10.0,
            max_spring_delta_per_step=10.0,
            max_damper_delta_per_step=10.0))

        output = controller.compute(_context(VehicleState(
            roll=30.0,
            roll_rate=100.0,
            local_ay=8.0)))

        self.assertEqual((1.0, 1.0, 1.0, 1.0), _springs(output))
        self.assertEqual(0.0, output.diagnostics["skyhook_roll_spring_add_fl"])
        self.assertEqual(0.0, output.diagnostics["skyhook_roll_spring_add_fr"])
        self.assertEqual(0.0, output.diagnostics["skyhook_roll_spring_add_rl"])
        self.assertEqual(0.0, output.diagnostics["skyhook_roll_spring_add_rr"])

    def test_yaw_aware_under_yaw_shifts_roll_distribution_rearward(self):
        controller = SkyhookRollController(SkyhookRollConfig(
            enable_yaw_distribution=True,
            max_front_distribution_delta_per_step=1.0,
            yaw_front_distribution_kp=0.10,
            yaw_front_distribution_ki=0.0))

        output = controller.compute(_context(VehicleState(
            speed=10.0,
            steer=0.2,
            local_ay=6.0,
            yaw_rate=math.degrees(0.05))))

        self.assertEqual("skyhook_roll_yaw", output.diagnostics["controller"])
        self.assertGreater(
            output.diagnostics["skyhook_roll_under_yaw_error_norm"],
            0.0)
        self.assertLess(
            output.diagnostics["skyhook_roll_front_share"],
            controller.config.nominal_front_roll_distribution)

    def test_yaw_aware_uses_velocity_norm_when_speed_is_zero(self):
        controller = SkyhookRollController(SkyhookRollConfig(
            enable_yaw_distribution=True,
            max_front_distribution_delta_per_step=1.0,
            yaw_front_distribution_kp=0.10,
            yaw_front_distribution_ki=0.0))

        output = controller.compute(_context(VehicleState(
            vx=10.0,
            speed=0.0,
            steer=0.2,
            local_ay=6.0,
            yaw_rate=math.degrees(0.05))))

        self.assertGreater(output.diagnostics["skyhook_roll_yaw_ref"], 0.0)
        self.assertGreater(
            output.diagnostics["skyhook_roll_under_yaw_error_norm"],
            0.0)
        self.assertLess(
            output.diagnostics["skyhook_roll_front_share"],
            controller.config.nominal_front_roll_distribution)

    def test_yaw_aware_over_yaw_shifts_roll_distribution_frontward(self):
        controller = SkyhookRollController(SkyhookRollConfig(
            enable_yaw_distribution=True,
            max_front_distribution_delta_per_step=1.0,
            yaw_front_distribution_kp=0.10,
            yaw_front_distribution_ki=0.0))

        output = controller.compute(_context(VehicleState(
            speed=10.0,
            steer=0.2,
            local_ay=6.0,
            yaw_rate=math.degrees(1.0))))

        self.assertLess(
            output.diagnostics["skyhook_roll_under_yaw_error_norm"],
            0.0)
        self.assertGreater(
            output.diagnostics["skyhook_roll_front_share"],
            controller.config.nominal_front_roll_distribution)

    def test_nonfinite_state_values_do_not_propagate_to_scales(self):
        controller = SkyhookRollController(SkyhookRollConfig(
            max_damper_delta_per_step=1.0,
            max_spring_delta_per_step=1.0,
            max_front_distribution_delta_per_step=1.0))

        output = controller.compute(_context(VehicleState(
            vz=float("nan"),
            roll=float("nan"),
            roll_rate=float("inf"),
            pitch_rate=float("-inf"),
            yaw_rate=float("nan"),
            local_ay=float("inf"),
            speed=float("nan"),
            vx=3.0,
            vy=4.0,
            steer=float("nan"))))

        for value in _springs(output) + _dampers(output):
            self.assertTrue(math.isfinite(value))
            self.assertGreater(value, 0.0)

    def test_config_accepts_geometry_aliases(self):
        config = SkyhookRollConfig.from_mapping({
            "half_track": 0.9,
            "half_wheelbase": 1.6,
        })

        self.assertEqual(0.9, config.half_track_m)
        self.assertEqual(1.6, config.half_wheelbase_m)

    def test_non_four_wheel_config_falls_back_conservatively(self):
        controller = SkyhookRollController(SkyhookRollConfig(wheel_count=3))

        output = controller.compute(_context(VehicleState(vz=10.0)))

        self.assertEqual(4, len(output.command.wheels))
        self.assertEqual((1.0, 1.0, 1.0, 1.0), _springs(output))
        self.assertEqual((1.0, 1.0, 1.0, 1.0), _dampers(output))
        output.command.validate(expected_wheels=4)
        self.assertEqual(
            "unsupported_wheel_count",
            output.diagnostics["skyhook_roll_fallback_reason"])

    def test_non_four_wheel_fallback_matches_native_wheel_count(self):
        controller = SkyhookRollController(SkyhookRollConfig(wheel_count=3))

        output = controller.compute(_context(
            VehicleState(vz=10.0),
            native_suspension=_FakeNativeSuspension(wheel_count=5)))

        self.assertEqual(5, len(output.command.wheels))
        self.assertEqual((1.0, 1.0, 1.0, 1.0, 1.0), _springs(output))
        self.assertEqual((1.0, 1.0, 1.0, 1.0, 1.0), _dampers(output))
        output.command.validate(expected_wheels=5)
        self.assertEqual(
            "unsupported_wheel_count",
            output.diagnostics["skyhook_roll_fallback_reason"])


if __name__ == "__main__":
    unittest.main()
