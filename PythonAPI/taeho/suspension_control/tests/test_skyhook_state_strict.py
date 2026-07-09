"""CARLA-free checks for skyhook_v3_canonical and v2 equivalence."""

from __future__ import annotations

from dataclasses import fields
import math
import unittest

from suspension_control.controllers.base import (
    ControllerContext,
    PlanningInfo,
    VehicleState,
)
from suspension_control.controllers.skyhook import (
    SKYHOOK_V2_STATE_STRICT_VERSION,
    SKYHOOK_V3_CANONICAL_VERSION,
    SkyhookConfig,
    SkyhookController,
    canonical_skyhook_v3_projection,
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
    previous_state=None,
    planning=None,
    suspension_state=None,
    suspension_state_valid=True,
    invalid_reason="",
    native_dampers=(4500.0, 4500.0, 4500.0, 4500.0),
    dt=0.05,
):
    return ControllerContext(
        state=state if state is not None else VehicleState(vz=1.0, dt=dt),
        previous_state=previous_state,
        planning=planning if planning is not None else PlanningInfo.empty(),
        native_suspension=_FakeNativeSuspension(),
        current_suspension=_FakeNativeSuspension(),
        suspension_state=(
            suspension_state
            if suspension_state is not None else _FakeSuspensionState()),
        suspension_state_valid=suspension_state_valid,
        suspension_state_invalid_reason=invalid_reason,
        native_spring_strength_by_wheel=(35000.0,) * 4,
        native_damper_rate_by_wheel=native_dampers,
        dt=dt)


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

    def test_default_version_is_v3_and_v2_remains_constructible(self):
        self.assertEqual(
            SKYHOOK_V3_CANONICAL_VERSION,
            SkyhookConfig().controller_version)
        self.assertEqual(
            SKYHOOK_V2_STATE_STRICT_VERSION,
            SkyhookConfig(
                controller_version=SKYHOOK_V2_STATE_STRICT_VERSION
            ).controller_version)

    def test_v3_numeric_defaults_match_v2(self):
        v2 = SkyhookConfig(controller_version=SKYHOOK_V2_STATE_STRICT_VERSION)
        v3 = SkyhookConfig(controller_version=SKYHOOK_V3_CANONICAL_VERSION)

        for item in fields(SkyhookConfig):
            if item.name == "controller_version":
                continue
            self.assertEqual(
                getattr(v2, item.name),
                getattr(v3, item.name),
                item.name)

    def test_v3_valid_state_output_equals_v2_output(self):
        v2 = SkyhookController(_config(
            controller_version=SKYHOOK_V2_STATE_STRICT_VERSION,
            max_damper_delta_per_step=0.035))
        v3 = SkyhookController(_config(
            controller_version=SKYHOOK_V3_CANONICAL_VERSION,
            max_damper_delta_per_step=0.035))
        contexts = (
            _context(
                state=VehicleState(vz=1.0, frame=10, dt=0.05),
                previous_state=VehicleState(frame=9),
                suspension_state=_FakeSuspensionState(velocities=(-0.5,) * 4)),
            _context(
                state=VehicleState(vz=1.0, frame=11, dt=0.05),
                previous_state=VehicleState(frame=10),
                suspension_state=_FakeSuspensionState(velocities=(0.5,) * 4)),
            _context(
                state=VehicleState(vz=0.001, frame=12, dt=0.05),
                previous_state=VehicleState(frame=11),
                suspension_state=_FakeSuspensionState(velocities=(-0.5,) * 4)),
        )

        for context in contexts:
            v2_output = v2.compute(context)
            v3_output = v3.compute(context)
            self.assertEqual(_springs(v2_output), _springs(v3_output))
            self.assertEqual(_dampers(v2_output), _dampers(v3_output))

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
            (1.0, -0.5, 0.5, "projected_feasible", "hard"),
            (1.0, 0.5, -0.5, "soft_infeasible", "soft"),
            (-1.0, 0.5, -0.5, "projected_feasible", "hard"),
            (-1.0, -0.5, 0.5, "soft_infeasible", "soft"),
        )

        for v_sprung, suspension_velocity, v_rel, branch, expected in cases:
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
            self.assertEqual(branch, output.diagnostics["target_branch_fl"])
            self.assertEqual(v_rel, output.diagnostics["v_rel_extension_mps_fl"])
            self.assertEqual(
                v_sprung * v_rel,
                output.diagnostics["skyhook_product_fl"])

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
        self.assertEqual(
            "neutral_sprung_deadband",
            sprung_deadband.diagnostics["target_branch_fl"])
        self.assertEqual(
            "neutral_rel_deadband",
            rel_deadband.diagnostics["target_branch_fl"])

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
        self.assertEqual(2.0, output.diagnostics["required_scale_unclipped_fl"])
        self.assertEqual(1.22, output.diagnostics["raw_target_damper_fl"])
        self.assertEqual(
            1.22,
            output.diagnostics["target_after_minmax_clamp_fl"])
        self.assertEqual(1, output.diagnostics["semi_active_feasible_fl"])

    def test_canonical_projection_helper_matches_controller_wrapper(self):
        config = _config(
            sprung_velocity_deadband=0.025,
            rel_velocity_deadband=0.015)
        controller = SkyhookController(config)
        native = 5000.0
        c_sky = native * config.skyhook_c_scale
        cases = (
            (1.0, 0.5, "projected_feasible"),
            (1.0, -0.5, "soft_infeasible"),
            (0.001, 0.5, "neutral_sprung_deadband"),
            (1.0, 0.001, "neutral_rel_deadband"),
        )

        for v_eff, v_rel, expected_branch in cases:
            helper = canonical_skyhook_v3_projection(
                v_eff_i=v_eff,
                v_rel_extension_i=v_rel,
                C_native_i=native,
                C_sky_i=c_sky,
                config=config)
            wrapper = controller._target_damper_scale(
                v_sprung=v_eff,
                v_rel_extension=v_rel,
                native_damper=native)

            self.assertEqual(wrapper, helper)
            self.assertEqual(expected_branch, helper["target_branch"])

    def test_canonical_projection_helper_reports_diagnostics(self):
        result = canonical_skyhook_v3_projection(
            v_eff_i=1.0,
            v_rel_extension_i=0.5,
            C_native_i=5000.0,
            C_sky_i=5000.0,
            config=_config())

        self.assertEqual("projected_feasible", result["target_branch"])
        self.assertEqual(0.5, result["skyhook_product"])
        self.assertEqual(10000.0, result["C_required"])
        self.assertEqual(2.0, result["required_scale_unclipped"])
        self.assertEqual(1.22, result["raw_target_damper"])
        self.assertEqual(1.22, result["target_after_minmax_clamp"])
        self.assertEqual(1.22, result["final_target_damper"])
        self.assertEqual(1, result["semi_active_feasible"])
        self.assertEqual(1, result["hard_mode"])
        self.assertEqual(0, result["soft_mode"])
        self.assertEqual(0, result["neutral_mode"])

    def test_valid_compute_output_matches_canonical_projection_helper(self):
        config = _config()
        controller = SkyhookController(config)
        output = controller.compute(_context(
            state=VehicleState(vz=1.0),
            suspension_state=_FakeSuspensionState(velocities=(-0.5,) * 4),
            native_dampers=(5000.0, 4500.0, 4000.0, 3500.0)))
        helper = canonical_skyhook_v3_projection(
            v_eff_i=1.0,
            v_rel_extension_i=0.5,
            C_native_i=5000.0,
            C_sky_i=5000.0,
            config=config)

        self.assertEqual(helper["final_target_damper"], _dampers(output)[0])
        for field_name in (
                "target_branch",
                "skyhook_product",
                "required_scale_unclipped",
                "raw_target_damper",
                "target_after_minmax_clamp",
                "final_target_damper"):
            self.assertEqual(
                helper[field_name],
                output.diagnostics["%s_fl" % field_name])

    def test_switching_law_compatibility_stays_in_projection_helper(self):
        config = _config(skyhook_law="switching")
        controller = SkyhookController(config)
        helper = canonical_skyhook_v3_projection(
            v_eff_i=1.0,
            v_rel_extension_i=0.5,
            C_native_i=4500.0,
            C_sky_i=4500.0,
            config=config)
        wrapper = controller._target_damper_scale(
            v_sprung=1.0,
            v_rel_extension=0.5,
            native_damper=4500.0)

        self.assertEqual(wrapper, helper)
        self.assertEqual("projected_feasible", helper["target_branch"])
        self.assertEqual("", helper["required_scale_unclipped"])
        self.assertEqual(config.high_damper_scale, helper["raw_target_damper"])

    def test_fixed_step_limiter_ignores_dt_for_command_output(self):
        config = _config(max_damper_delta_per_step=0.02)
        fast_dt = SkyhookController(config)
        slow_dt = SkyhookController(config)
        state = VehicleState(vz=1.0, frame=20)
        previous_state = VehicleState(frame=19)
        suspension_state = _FakeSuspensionState(velocities=(-0.5,) * 4)

        fast_output = fast_dt.compute(_context(
            state=state,
            previous_state=previous_state,
            suspension_state=suspension_state,
            dt=0.01))
        slow_output = slow_dt.compute(_context(
            state=state,
            previous_state=previous_state,
            suspension_state=suspension_state,
            dt=0.20))

        self.assertEqual(_dampers(fast_output), _dampers(slow_output))
        self.assertEqual((1.02, 1.02, 1.02, 1.02), _dampers(fast_output))
        self.assertEqual(0.02, fast_output.diagnostics[
            "max_damper_delta_per_step_used"])
        self.assertEqual(1, slow_output.diagnostics["dt_gap_warning"])
        self.assertEqual(1, slow_output.diagnostics["frame_delta"])

    def test_invalid_state_resets_previous_damper_scales(self):
        controller = SkyhookController(_config(max_damper_delta_per_step=0.02))
        controller.compute(_context(
            state=VehicleState(vz=1.0),
            suspension_state=_FakeSuspensionState(velocities=(-0.5,) * 4)))
        self.assertEqual((1.02, 1.02, 1.02, 1.02), tuple(
            controller.previous_damper_scales))

        invalid = controller.compute(_context(
            suspension_state=_FakeSuspensionState(field_valid=False)))

        self.assertEqual((1.0, 1.0, 1.0, 1.0), _dampers(invalid))
        self.assertEqual((1.0, 1.0, 1.0, 1.0), tuple(
            controller.previous_damper_scales))

    def test_planning_and_control_inputs_do_not_affect_command_output(self):
        config = _config(max_damper_delta_per_step=0.02)
        baseline = SkyhookController(config)
        with_preview = SkyhookController(config)

        baseline_output = baseline.compute(_context(
            state=VehicleState(vz=1.0, throttle=0.0, brake=0.0, steer=0.0),
            suspension_state=_FakeSuspensionState(velocities=(-0.5,) * 4)))
        preview_output = with_preview.compute(_context(
            state=VehicleState(vz=1.0, throttle=1.0, brake=0.8, steer=-0.7),
            planning=PlanningInfo(
                available=True,
                source="unit",
                frame=1,
                target_speed=(30.0, 0.0),
                curvature=(0.5, -0.5),
                throttle=(1.0,),
                brake=(0.9,),
                steer=(-0.8,)),
            suspension_state=_FakeSuspensionState(velocities=(-0.5,) * 4)))

        self.assertEqual(_springs(baseline_output), _springs(preview_output))
        self.assertEqual(_dampers(baseline_output), _dampers(preview_output))

    def test_vertical_velocity_source_defaults_to_world_z_behavior(self):
        controller = SkyhookController(_config())

        output = controller.compute(_context(
            state=VehicleState(vz=1.0, roll_rate=0.0, pitch_rate=0.0),
            suspension_state=_FakeSuspensionState(velocities=(-0.5,) * 4)))

        self.assertEqual("world_z", controller.config.vertical_velocity_source)
        self.assertEqual("world_z", output.diagnostics[
            "vertical_velocity_source"])
        self.assertEqual(1.0, output.diagnostics["v_world_z"])
        self.assertEqual(1.0, output.diagnostics["v_sprung_used_fl"])

    def test_startup_prime_diagnostic_keeps_identity_command(self):
        controller = SkyhookController(_config(max_damper_delta_per_step=0.02))
        controller.compute(_context(
            state=VehicleState(vz=1.0),
            suspension_state=_FakeSuspensionState(velocities=(-0.5,) * 4)))
        state = _FakeSuspensionState()
        state.velocity_valid = False

        output = controller.compute(_context(
            state=VehicleState(frame=2, vz=1.0),
            suspension_state=state))

        self.assertEqual((1.0, 1.0, 1.0, 1.0), _dampers(output))
        self.assertEqual(1, output.diagnostics["startup_prime"])
        self.assertEqual((1.0, 1.0, 1.0, 1.0), tuple(
            controller.previous_damper_scales))

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
