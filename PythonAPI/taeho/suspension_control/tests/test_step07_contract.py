"""Focused Step 07 controller and suite-selection contract checks."""

from __future__ import annotations

import math
import unittest

from suspension_control.controllers.base import (
    ControllerContext,
    PlanningInfo,
    VehicleState,
)
from suspension_control.controllers.constant_scale import (
    ConstantScaleConfig,
    ConstantScaleController,
)
from suspension_control.controllers.skyhook_roll import SkyhookRollController
from suspension_control.controllers.target_speed_schedule import (
    TargetSpeedScheduleConfig,
    TargetSpeedScheduleController,
)


def _context(planning=None, dt=0.05):
    return ControllerContext(
        state=VehicleState(dt=dt),
        planning=planning if planning is not None else PlanningInfo.empty(),
        dt=dt)


def _damper_scales(output):
    return tuple(wheel.damper_scale for wheel in output.command.wheels)


def _spring_scales(output):
    return tuple(wheel.spring_scale for wheel in output.command.wheels)


class ConstantScaleControllerTest(unittest.TestCase):

    def test_constant_damper_1p03_all_wheels(self):
        controller = ConstantScaleController(ConstantScaleConfig(
            spring_scale=1.0,
            damper_scale=1.03,
            wheel_count=4))

        output = controller.compute(_context())

        self.assertEqual((1.0, 1.0, 1.0, 1.0), _spring_scales(output))
        self.assertEqual((1.03, 1.03, 1.03, 1.03), _damper_scales(output))
        self.assertEqual("constant_scale", output.diagnostics["controller"])


class TargetSpeedScheduleControllerTest(unittest.TestCase):

    def test_unavailable_planning_falls_back_to_identity(self):
        controller = TargetSpeedScheduleController()

        output = controller.compute(_context(PlanningInfo.empty()))

        self.assertEqual((1.0, 1.0, 1.0, 1.0), _spring_scales(output))
        self.assertEqual((1.0, 1.0, 1.0, 1.0), _damper_scales(output))
        self.assertEqual(0, output.diagnostics["tss_target_speed_valid"])
        self.assertEqual(
            "planning_unavailable",
            output.diagnostics["tss_fallback_reason"])

    def test_negative_target_speed_falls_back_to_identity(self):
        controller = TargetSpeedScheduleController()
        planning = PlanningInfo(available=True, target_speed=(-1.0,))

        output = controller.compute(_context(planning))

        self.assertEqual((1.0, 1.0, 1.0, 1.0), _damper_scales(output))
        self.assertEqual(
            "negative_target_speed",
            output.diagnostics["tss_fallback_reason"])

    def test_nonfinite_target_speed_falls_back_to_identity(self):
        controller = TargetSpeedScheduleController()
        planning = PlanningInfo(available=True, target_speed=(float("nan"),))

        output = controller.compute(_context(planning))

        self.assertEqual((1.0, 1.0, 1.0, 1.0), _damper_scales(output))
        self.assertEqual(
            "nonfinite_target_speed",
            output.diagnostics["tss_fallback_reason"])

    def test_valid_low_target_speed_increases_bounded_damping(self):
        controller = TargetSpeedScheduleController(TargetSpeedScheduleConfig(
            smoothing_time_constant_sec=0.0,
            max_scale_rate_per_sec=10.0))
        planning = PlanningInfo(available=True, target_speed=(0.0,))

        output = controller.compute(_context(planning))
        damper = _damper_scales(output)[0]

        self.assertGreater(damper, 1.0)
        self.assertLessEqual(damper, 1.05)
        self.assertEqual((1.0, 1.0, 1.0, 1.0), _spring_scales(output))
        self.assertEqual(1, output.diagnostics["tss_target_speed_valid"])
        self.assertEqual("", output.diagnostics["tss_fallback_reason"])

    def test_rate_limiter_caps_damper_change(self):
        controller = TargetSpeedScheduleController(TargetSpeedScheduleConfig(
            smoothing_time_constant_sec=0.0,
            max_scale_rate_per_sec=0.10))
        planning = PlanningInfo(available=True, target_speed=(0.0,))

        output = controller.compute(_context(planning, dt=0.05))
        damper = _damper_scales(output)[0]

        self.assertTrue(math.isclose(damper, 1.005, rel_tol=0.0, abs_tol=1e-12))
        self.assertEqual(1, output.diagnostics["tss_rate_limited"])

    def test_diagnostics_expose_forbidden_signal_usage_flags(self):
        controller = TargetSpeedScheduleController()
        planning = PlanningInfo(
            available=True,
            target_speed=(4.0,),
            curvature=(0.1,),
            steer=(0.2,),
            throttle=(0.3,),
            brake=(0.4,),
            trajectory_xy=((1.0, 2.0),))

        output = controller.compute(_context(planning))

        self.assertEqual(0, output.diagnostics["tss_used_curvature"])
        self.assertEqual(0, output.diagnostics["tss_used_trajectory"])
        self.assertEqual(0, output.diagnostics["tss_used_control"])

    def test_shadow_mode_logs_computed_but_applies_identity(self):
        controller = TargetSpeedScheduleController(TargetSpeedScheduleConfig(
            smoothing_time_constant_sec=0.0,
            max_scale_rate_per_sec=10.0,
            shadow_mode=True))
        planning = PlanningInfo(available=True, target_speed=(0.0,))

        output = controller.compute(_context(planning))

        self.assertEqual((1.0, 1.0, 1.0, 1.0), _damper_scales(output))
        self.assertGreater(output.diagnostics["tss_damper_scale_computed"], 1.0)
        self.assertEqual(1.0, output.diagnostics["tss_damper_scale_applied"])
        self.assertEqual(1, output.diagnostics["tss_shadow_mode"])

    def test_safe_config_keeps_damping_ultra_conservative(self):
        controller = TargetSpeedScheduleController(TargetSpeedScheduleConfig(
            damper_max=1.01,
            low_speed_gain=0.01,
            drop_gain=0.0,
            smoothing_time_constant_sec=0.0,
            max_scale_rate_per_sec=10.0))
        planning = PlanningInfo(available=True, target_speed=(0.0,))

        output = controller.compute(_context(planning))

        self.assertLessEqual(_damper_scales(output)[0], 1.01)
        self.assertEqual(0.0, output.diagnostics["tss_drop_component"])


class Step07SuiteSelectionTest(unittest.TestCase):

    def test_new_scenarios_are_selectable(self):
        import transfuser_suspension_control_suite as suite

        scenarios = suite.selected_scenarios(
            "constant_damper_1p03,target_speed_schedule_v0,"
            "target_speed_schedule_shadow,target_speed_schedule_v0_safe,"
            "skyhook_roll,skyhook_roll_yaw")

        self.assertEqual(
            [
                "S15_constant_damper_1p03",
                "S16_target_speed_schedule_v0",
                "S17_target_speed_schedule_shadow",
                "S18_target_speed_schedule_v0_safe",
                "S19_skyhook_roll",
                "S20_skyhook_roll_yaw",
            ],
            [scenario["name"] for scenario in scenarios])

    def test_skyhook_roll_scenarios_are_selectable_by_code(self):
        import transfuser_suspension_control_suite as suite

        scenarios = suite.selected_scenarios("S19,S20")

        self.assertEqual(
            ["S19_skyhook_roll", "S20_skyhook_roll_yaw"],
            [scenario["name"] for scenario in scenarios])

    def test_make_controller_builds_new_controllers(self):
        import transfuser_suspension_control_suite as suite

        args = suite.build_arg_parser().parse_args([])

        constant = suite.make_controller("constant_scale", args)
        schedule = suite.make_controller("target_speed_schedule", args)
        shadow = suite.make_controller("target_speed_schedule_shadow", args)
        safe = suite.make_controller("target_speed_schedule_safe", args)
        skyhook_roll = suite.make_controller("skyhook_roll", args)
        skyhook_roll_yaw = suite.make_controller("skyhook_roll_yaw", args)

        self.assertIsInstance(constant, ConstantScaleController)
        self.assertIsInstance(schedule, TargetSpeedScheduleController)
        self.assertIsInstance(shadow, TargetSpeedScheduleController)
        self.assertIsInstance(safe, TargetSpeedScheduleController)
        self.assertIsInstance(skyhook_roll, SkyhookRollController)
        self.assertIsInstance(skyhook_roll_yaw, SkyhookRollController)
        self.assertFalse(skyhook_roll.config.enable_yaw_distribution)
        self.assertTrue(skyhook_roll_yaw.config.enable_yaw_distribution)

    def test_skyhook_roll_keeps_suite_defaults_opt_in(self):
        import lead_suspension_control_suite as lead_suite
        import transfuser_suspension_control_suite as suite

        tfpp_args = suite.build_arg_parser().parse_args([])
        lead_args = lead_suite.build_arg_parser().parse_args([])

        self.assertEqual("stock,identity,pid", tfpp_args.scenarios)
        self.assertEqual("stock,identity,skyhook", lead_args.scenarios)

    def test_skyhook_roll_help_mentions_new_scenarios(self):
        import lead_suspension_control_suite as lead_suite
        import transfuser_suspension_control_suite as suite

        tfpp_help = suite.build_arg_parser().format_help()
        lead_help = lead_suite.build_arg_parser().format_help()

        for help_text in (tfpp_help, lead_help):
            self.assertIn("skyhook_roll", help_text)
            self.assertIn("skyhook_roll_yaw", help_text)
        self.assertIn("--skyhook-roll-config", tfpp_help)
        self.assertIn("--skyhook-roll-yaw-config", tfpp_help)

    def test_new_diagnostics_are_registered(self):
        import transfuser_suspension_control_suite as suite

        for field_name in (
                "tss_target_speed_raw",
                "tss_target_speed_valid",
                "tss_damper_scale_computed",
                "tss_damper_scale_applied",
                "tss_shadow_mode",
                "tss_used_curvature",
                "tss_used_trajectory",
                "tss_used_control",
                "skyhook_roll_mode",
                "skyhook_roll_roll_rad",
                "skyhook_roll_roll_rate_rad",
                "skyhook_roll_pitch_rate_rad",
                "skyhook_roll_yaw_rate_rad",
                "skyhook_roll_local_ay",
                "skyhook_roll_lat_activity",
                "skyhook_roll_damping_activity",
                "skyhook_roll_stiffness_activity",
                "skyhook_roll_front_share",
                "skyhook_roll_yaw_ref",
                "skyhook_roll_under_yaw_error_norm",
                "skyhook_roll_outer_side_sign",
                "skyhook_roll_fallback_reason",
                "skyhook_roll_side_weight_fl",
                "skyhook_roll_side_weight_fr",
                "skyhook_roll_side_weight_rl",
                "skyhook_roll_side_weight_rr",
                "skyhook_roll_spring_scale_fl",
                "skyhook_roll_spring_scale_fr",
                "skyhook_roll_spring_scale_rl",
                "skyhook_roll_spring_scale_rr",
                "skyhook_roll_damper_scale_fl",
                "skyhook_roll_damper_scale_fr",
                "skyhook_roll_damper_scale_rl",
                "skyhook_roll_damper_scale_rr",
                "skyhook_roll_damper_add_fl",
                "skyhook_roll_damper_add_fr",
                "skyhook_roll_damper_add_rl",
                "skyhook_roll_damper_add_rr",
                "skyhook_roll_spring_add_fl",
                "skyhook_roll_spring_add_fr",
                "skyhook_roll_spring_add_rl",
                "skyhook_roll_spring_add_rr"):
            self.assertIn(field_name, suite.DIAGNOSTIC_FIELDS)


if __name__ == "__main__":
    unittest.main()
