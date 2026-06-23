"""CARLA-free checks for skyhook_roll_v2_state_damper_only."""

from __future__ import annotations

import math
import unittest

from suspension_control.controllers.base import ControllerContext, VehicleState
from suspension_control.controllers.skyhook_roll import (
    SkyhookRollConfig,
    SkyhookRollController,
)


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

    def __init__(self, velocities=(-0.5, -0.5, -0.5, -0.5), **wheel_kwargs):
        self.state_valid = True
        self.velocity_valid = True
        self.compression_convention_validated = True
        self.state_source = "fake"
        self.failure_reason = ""
        self.wheels = tuple(
            _FakeWheel(index, velocity, **wheel_kwargs)
            for index, velocity in enumerate(velocities))


class _FakeNativeWheel:
    spring_strength = 35000.0
    spring_damper_rate = 4500.0


class _FakeNativeSuspension:
    wheels = (_FakeNativeWheel(),) * 4


def _context(
    state=None,
    suspension_state=None,
    suspension_state_valid=True,
    invalid_reason="",
    native_dampers=(4500.0, 4500.0, 4500.0, 4500.0),
):
    return ControllerContext(
        state=state if state is not None else VehicleState(dt=0.05),
        native_suspension=_FakeNativeSuspension(),
        current_suspension=_FakeNativeSuspension(),
        suspension_state=(
            suspension_state
            if suspension_state is not None else _FakeSuspensionState()),
        suspension_state_valid=suspension_state_valid,
        suspension_state_invalid_reason=invalid_reason,
        native_spring_strength_by_wheel=(35000.0,) * 4,
        native_damper_rate_by_wheel=native_dampers,
        dt=0.05)


def _config(**overrides):
    values = {
        "max_damper_delta_per_step": 10.0,
        "sprung_velocity_deadband": 0.0,
        "rel_velocity_deadband": 0.0,
        "product_deadband": 0.0,
    }
    values.update(overrides)
    return SkyhookRollConfig(**values)


def _springs(output):
    return tuple(wheel.spring_scale for wheel in output.command.wheels)


def _dampers(output):
    return tuple(wheel.damper_scale for wheel in output.command.wheels)


def _roll_state(roll_rate_deg_s=20.0, local_ay=6.0, roll_deg=4.0):
    return VehicleState(
        dt=0.05,
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


class SkyhookRollStateStrictTest(unittest.TestCase):

    def test_spring_scale_is_always_identity(self):
        controller = SkyhookRollController(_config(
            base_spring_scale=2.0,
            roll_spring_enabled=True,
            enable_roll_spring_control=True,
            max_spring_delta_per_step=10.0,
            roll_c_scale=2.0))

        output = controller.compute(_context(
            state=_roll_state(),
            suspension_state=_state_matching_roll_extension()))

        self.assertEqual((1.0, 1.0, 1.0, 1.0), _springs(output))
        self.assertEqual(1.0, output.diagnostics["final_spring_scale_fl"])
        self.assertEqual(1.0, output.diagnostics["skyhook_roll_spring_scale_fl"])
        self.assertEqual(
            "skyhook_roll_v2_state_damper_only",
            output.diagnostics["controller_version"])

    def test_strict_invalid_state_falls_back_to_identity(self):
        controller = SkyhookRollController(_config())
        bad_state = _FakeSuspensionState(field_valid=False)

        output = controller.compute(_context(suspension_state=bad_state))

        self.assertEqual((1.0, 1.0, 1.0, 1.0), _springs(output))
        self.assertEqual((1.0, 1.0, 1.0, 1.0), _dampers(output))
        self.assertEqual(1, output.diagnostics["identity_fallback_this_tick"])
        self.assertEqual(
            "wheel_0_field_valid_false",
            output.diagnostics["identity_fallback_reason"])

    def test_roll_layer_is_near_zero_at_low_lateral_acceleration(self):
        controller = SkyhookRollController(_config(roll_c_scale=5.0))

        output = controller.compute(_context(
            state=_roll_state(local_ay=0.0),
            suspension_state=_state_matching_roll_extension()))

        self.assertEqual(0.0, output.diagnostics["roll_gate_global"])
        for label in ("fl", "fr", "rl", "rr"):
            self.assertEqual(
                0.0,
                output.diagnostics["F_roll_ideal_%s" % label])

    def test_roll_layer_combines_before_projection(self):
        controller = SkyhookRollController(_config(roll_c_scale=0.5))

        output = controller.compute(_context(
            state=_roll_state(),
            suspension_state=_state_matching_roll_extension()))

        self.assertGreater(output.diagnostics["roll_gate_global"], 0.0)
        self.assertNotEqual(0.0, output.diagnostics["F_roll_ideal_fl"])
        self.assertAlmostEqual(
            output.diagnostics["F_sky_ideal_fl"] +
            output.diagnostics["F_roll_ideal_fl"],
            output.diagnostics["F_total_ideal_fl"])
        self.assertGreaterEqual(
            output.diagnostics["final_target_damper_fl"],
            output.diagnostics["skyhook_only_target_damper_fl"])

    def test_roll_softening_guard_limits_override_of_soft_skyhook(self):
        controller = SkyhookRollController(_config(
            roll_c_scale=3.0,
            max_roll_extra_scale=1.0,
            roll_softening_override_limit=0.03))

        output = controller.compute(_context(
            state=VehicleState(
                dt=0.05,
                vz=1.0,
                roll=4.0,
                roll_rate=30.0,
                local_ay=6.0),
            suspension_state=_FakeSuspensionState(
                velocities=(0.5, 0.5, 0.5, 0.5))))

        self.assertLess(
            output.diagnostics["skyhook_only_target_damper_fl"],
            1.0)
        self.assertGreater(output.diagnostics["F_roll_ideal_fl"], 0.0)
        self.assertEqual(
            1,
            output.diagnostics["roll_softening_guard_active_fl"])
        self.assertLessEqual(
            output.diagnostics["final_target_damper_fl"],
            1.0)

    def test_output_contains_only_scale_commands_not_active_forces(self):
        controller = SkyhookRollController(_config(roll_c_scale=1.0))

        output = controller.compute(_context(
            state=_roll_state(),
            suspension_state=_state_matching_roll_extension()))

        for wheel in output.command.wheels:
            self.assertTrue(math.isfinite(wheel.spring_scale))
            self.assertTrue(math.isfinite(wheel.damper_scale))
            self.assertFalse(hasattr(wheel, "force"))
            self.assertFalse(hasattr(wheel, "force_n"))

    def test_yaw_distribution_config_does_not_change_applied_command(self):
        state = VehicleState(
            dt=0.05,
            speed=12.0,
            local_vx=12.0,
            steer=0.4,
            yaw_rate=90.0,
            roll=4.0,
            roll_rate=20.0,
            local_ay=6.0)
        suspension_state = _state_matching_roll_extension()
        fixed = SkyhookRollController(_config(enable_yaw_distribution=False))
        yaw_configured = SkyhookRollController(_config(
            enable_yaw_distribution=True,
            yaw_front_distribution_kp=10.0,
            yaw_front_distribution_ki=10.0))

        fixed_output = fixed.compute(_context(
            state=state,
            suspension_state=suspension_state))
        yaw_output = yaw_configured.compute(_context(
            state=state,
            suspension_state=suspension_state))

        self.assertEqual(_dampers(fixed_output), _dampers(yaw_output))
        self.assertEqual(_springs(fixed_output), _springs(yaw_output))
        self.assertEqual(
            "yaw_log_only_no_apply",
            yaw_output.diagnostics["skyhook_roll_mode"])
        self.assertEqual(
            "log_only",
            yaw_output.diagnostics["yaw_distribution_mode"])
        self.assertEqual(0, yaw_output.diagnostics["yaw_apply_enabled"])
        self.assertEqual(
            yaw_configured.config.nominal_front_roll_distribution,
            yaw_output.diagnostics["skyhook_roll_front_share"])
        self.assertNotEqual(
            yaw_output.diagnostics["front_distribution_applied"],
            yaw_output.diagnostics["front_distribution_proposed"])

    def test_positive_yaw_error_reduces_proposed_front_distribution(self):
        controller = SkyhookRollController(_config(
            enable_yaw_distribution=True,
            yaw_distribution_mode="log_only",
            yaw_front_distribution_kp=0.08))

        output = controller.compute(_context(
            state=VehicleState(
                dt=0.05,
                speed=12.0,
                local_vx=12.0,
                steer=0.20,
                yaw_rate=0.0,
                roll=4.0,
                roll_rate=20.0,
                local_ay=6.0),
            suspension_state=_state_matching_roll_extension()))

        self.assertGreater(output.diagnostics["yaw_error"], 0.0)
        self.assertGreater(output.diagnostics["yaw_activation"], 0.0)
        self.assertLess(
            output.diagnostics["front_distribution_proposed"],
            output.diagnostics["front_distribution_nominal"])
        self.assertAlmostEqual(
            1.0,
            output.diagnostics["front_distribution_proposed"] +
            output.diagnostics["rear_distribution_proposed"])

    def test_negative_yaw_error_increases_proposed_front_distribution(self):
        controller = SkyhookRollController(_config(
            enable_yaw_distribution=True,
            yaw_distribution_mode="log_only",
            yaw_front_distribution_kp=0.08))

        output = controller.compute(_context(
            state=VehicleState(
                dt=0.05,
                speed=12.0,
                local_vx=12.0,
                steer=0.05,
                yaw_rate=60.0,
                roll=4.0,
                roll_rate=20.0,
                local_ay=6.0),
            suspension_state=_state_matching_roll_extension()))

        self.assertLess(output.diagnostics["yaw_error"], 0.0)
        self.assertGreater(output.diagnostics["yaw_activation"], 0.0)
        self.assertGreater(
            output.diagnostics["front_distribution_proposed"],
            output.diagnostics["front_distribution_nominal"])

    def test_yaw_activation_is_quiet_below_point_four_g(self):
        controller = SkyhookRollController(_config(
            enable_yaw_distribution=True,
            yaw_distribution_mode="log_only"))

        output = controller.compute(_context(
            state=VehicleState(
                dt=0.05,
                speed=12.0,
                local_vx=12.0,
                steer=0.25,
                yaw_rate=0.0,
                roll=4.0,
                roll_rate=20.0,
                local_ay=0.39 * 9.81),
            suspension_state=_state_matching_roll_extension()))

        self.assertLessEqual(abs(output.diagnostics["yaw_activation"]), 1e-12)
        self.assertEqual(
            output.diagnostics["front_distribution_nominal"],
            output.diagnostics["front_distribution_proposed"])

    def test_config_accepts_geometry_and_step05_aliases(self):
        config = SkyhookRollConfig.from_mapping({
            "half_track": 0.9,
            "half_wheelbase": 1.6,
            "relative_velocity_deadband": 0.02,
            "base_damper_scale": 1.0,
        })

        self.assertEqual(0.9, config.half_track_m)
        self.assertEqual(1.6, config.half_wheelbase_m)
        self.assertEqual(0.02, config.rel_velocity_deadband)
        self.assertEqual(1.0, config.neutral_damper_scale)


if __name__ == "__main__":
    unittest.main()
