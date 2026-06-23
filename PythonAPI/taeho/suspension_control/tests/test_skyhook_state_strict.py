"""CARLA-free checks for skyhook_v2_state_strict."""

from __future__ import annotations

import math
import unittest

from suspension_control.controllers.base import ControllerContext, VehicleState
from suspension_control.controllers.skyhook import SkyhookConfig, SkyhookController


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
        state=state if state is not None else VehicleState(vz=1.0, dt=0.05),
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
    return SkyhookConfig(**values)


def _springs(output):
    return tuple(wheel.spring_scale for wheel in output.command.wheels)


def _dampers(output):
    return tuple(wheel.damper_scale for wheel in output.command.wheels)


class SkyhookStateStrictTest(unittest.TestCase):

    def test_actor_angular_velocity_degrees_are_converted_to_radians(self):
        controller = SkyhookController(_config())
        output = controller.compute(_context(
            state=VehicleState(vz=0.0, roll_rate=180.0, pitch_rate=90.0)))

        self.assertAlmostEqual(math.pi, output.diagnostics["roll_rate_rad_s"])
        self.assertAlmostEqual(math.pi / 2.0, output.diagnostics["pitch_rate_rad_s"])
        self.assertEqual(
            "deg_s_to_rad_s",
            output.diagnostics["angular_velocity_unit_converted"])

    def test_radian_angular_velocity_source_is_not_converted_twice(self):
        controller = SkyhookController(_config(
            angular_velocity_source="snapshot_rad_s",
            angular_velocity_unit="rad_s"))
        output = controller.compute(_context(
            state=VehicleState(vz=0.0, roll_rate=math.pi)))

        self.assertAlmostEqual(math.pi, output.diagnostics["roll_rate_rad_s"])
        self.assertEqual(
            "none_rad_s",
            output.diagnostics["angular_velocity_unit_converted"])

    def test_projected_force_sign_cases(self):
        cases = (
            (1.0, -0.5, "hard"),
            (1.0, 0.5, "soft"),
            (-1.0, 0.5, "hard"),
            (-1.0, -0.5, "soft"),
        )

        for v_sprung, suspension_velocity, expected in cases:
            controller = SkyhookController(_config())
            output = controller.compute(_context(
                state=VehicleState(vz=v_sprung),
                suspension_state=_FakeSuspensionState(
                    velocities=(suspension_velocity,) * 4)))
            damper = _dampers(output)[0]
            if expected == "hard":
                self.assertGreater(damper, 1.0)
                self.assertEqual(1, output.diagnostics["hard_mode_fl"])
                self.assertEqual(1, output.diagnostics["semi_active_feasible_fl"])
            else:
                self.assertLess(damper, 1.0)
                self.assertEqual(1, output.diagnostics["soft_mode_fl"])
                self.assertEqual(0, output.diagnostics["semi_active_feasible_fl"])

    def test_deadbands_return_neutral_damping(self):
        controller = SkyhookController(_config(
            sprung_velocity_deadband=0.025,
            rel_velocity_deadband=0.015))

        sprung_deadband = controller.compute(_context(state=VehicleState(vz=0.001)))
        rel_deadband = controller.compute(_context(
            state=VehicleState(vz=1.0),
            suspension_state=_FakeSuspensionState(velocities=(-0.001,) * 4)))

        self.assertEqual((1.0, 1.0, 1.0, 1.0), _dampers(sprung_deadband))
        self.assertEqual((1.0, 1.0, 1.0, 1.0), _dampers(rel_deadband))
        self.assertEqual(1, sprung_deadband.diagnostics["neutral_mode_fl"])
        self.assertEqual(1, rel_deadband.diagnostics["neutral_mode_fl"])

    def test_no_hardening_only_regression(self):
        controller = SkyhookController(_config())

        hard = controller.compute(_context(
            state=VehicleState(vz=1.0),
            suspension_state=_FakeSuspensionState(velocities=(-0.5,) * 4)))
        soft = controller.compute(_context(
            state=VehicleState(vz=1.0),
            suspension_state=_FakeSuspensionState(velocities=(0.5,) * 4)))

        self.assertGreater(max(_dampers(hard)), 1.0)
        self.assertLess(min(_dampers(soft)), 1.0)

    def test_strict_invalid_state_falls_back_to_identity(self):
        controller = SkyhookController(_config())
        state = _FakeSuspensionState(field_valid=False)

        output = controller.compute(_context(suspension_state=state))

        self.assertEqual((1.0, 1.0, 1.0, 1.0), _springs(output))
        self.assertEqual((1.0, 1.0, 1.0, 1.0), _dampers(output))
        self.assertEqual(1, output.diagnostics["identity_fallback_this_tick"])
        self.assertEqual(
            "wheel_0_field_valid_false",
            output.diagnostics["identity_fallback_reason"])

    def test_missing_state_falls_back_to_identity(self):
        controller = SkyhookController(_config())

        output = controller.compute(_context(
            suspension_state=None,
            suspension_state_valid=False,
            invalid_reason="get_suspension_state_missing"))

        self.assertEqual((1.0, 1.0, 1.0, 1.0), _dampers(output))
        self.assertEqual(
            "get_suspension_state_missing",
            output.diagnostics["identity_fallback_reason"])

    def test_rate_limiter_bounds_per_tick_change(self):
        controller = SkyhookController(_config(max_damper_delta_per_step=0.02))

        first = controller.compute(_context(
            state=VehicleState(vz=1.0),
            suspension_state=_FakeSuspensionState(velocities=(-0.5,) * 4)))
        second = controller.compute(_context(
            state=VehicleState(vz=1.0),
            suspension_state=_FakeSuspensionState(velocities=(-0.5,) * 4)))

        self.assertEqual((1.02, 1.02, 1.02, 1.02), _dampers(first))
        for left, right in zip(_dampers(first), _dampers(second)):
            self.assertLessEqual(abs(right - left), 0.020000001)
        self.assertEqual(1, first.diagnostics["rate_limited_damper_fl"])

    def test_spring_scale_is_always_identity(self):
        controller = SkyhookController(_config())

        hard = controller.compute(_context(
            state=VehicleState(vz=1.0),
            suspension_state=_FakeSuspensionState(velocities=(-0.5,) * 4)))
        soft = controller.compute(_context(
            state=VehicleState(vz=1.0),
            suspension_state=_FakeSuspensionState(velocities=(0.5,) * 4)))

        self.assertEqual((1.0, 1.0, 1.0, 1.0), _springs(hard))
        self.assertEqual((1.0, 1.0, 1.0, 1.0), _springs(soft))
        self.assertEqual(1.0, hard.diagnostics["final_spring_scale_fl"])

    def test_projected_law_uses_native_damper_rate(self):
        controller = SkyhookController(_config())

        output = controller.compute(_context(
            state=VehicleState(vz=1.0),
            suspension_state=_FakeSuspensionState(velocities=(-0.5,) * 4),
            native_dampers=(5000.0, 4500.0, 4000.0, 3500.0)))

        self.assertEqual(5000.0, output.diagnostics["C_native_fl"])
        self.assertEqual(10000.0, output.diagnostics["C_required_fl"])
        self.assertEqual(1, output.diagnostics["semi_active_feasible_fl"])

    def test_config_accepts_step05_aliases(self):
        config = SkyhookConfig.from_mapping({
            "half_track": 0.9,
            "half_wheelbase": 1.6,
            "relative_velocity_deadband": 0.02,
        })

        self.assertEqual(0.9, config.half_track_m)
        self.assertEqual(1.6, config.half_wheelbase_m)
        self.assertEqual(0.02, config.rel_velocity_deadband)


if __name__ == "__main__":
    unittest.main()
