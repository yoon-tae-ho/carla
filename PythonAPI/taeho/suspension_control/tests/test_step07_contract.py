"""Focused Step 07 controller and suite-selection contract checks."""

from __future__ import annotations

import csv
import math
import os
import tempfile
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
from suspension_control.controllers.estimators import (
    SkyhookEstimatorDryRunController,
)
from suspension_control.controllers.skyhook import SkyhookController
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
            "skyhook_roll,skyhook_roll_yaw,skyhook_estimator_dryrun")

        self.assertEqual(
            [
                "S15_constant_damper_1p03",
                "S16_target_speed_schedule_v0",
                "S17_target_speed_schedule_shadow",
                "S18_target_speed_schedule_v0_safe",
                "S19_skyhook_roll",
                "S20_skyhook_roll_yaw",
                "S21_skyhook_estimator_dryrun",
            ],
            [scenario["name"] for scenario in scenarios])

    def test_skyhook_roll_scenarios_are_selectable_by_code(self):
        import transfuser_suspension_control_suite as suite

        scenarios = suite.selected_scenarios("S19,S20,S21")

        self.assertEqual(
            [
                "S19_skyhook_roll",
                "S20_skyhook_roll_yaw",
                "S21_skyhook_estimator_dryrun",
            ],
            [scenario["name"] for scenario in scenarios])

    def test_make_controller_builds_new_controllers(self):
        import transfuser_suspension_control_suite as suite

        args = suite.build_arg_parser().parse_args([])

        constant = suite.make_controller("constant_scale", args)
        schedule = suite.make_controller("target_speed_schedule", args)
        shadow = suite.make_controller("target_speed_schedule_shadow", args)
        safe = suite.make_controller("target_speed_schedule_safe", args)
        skyhook = suite.make_controller("skyhook", args)
        skyhook_roll = suite.make_controller("skyhook_roll", args)
        skyhook_roll_yaw = suite.make_controller("skyhook_roll_yaw", args)
        dryrun = suite.make_controller("skyhook_estimator_dryrun", args)

        self.assertIsInstance(constant, ConstantScaleController)
        self.assertIsInstance(schedule, TargetSpeedScheduleController)
        self.assertIsInstance(shadow, TargetSpeedScheduleController)
        self.assertIsInstance(safe, TargetSpeedScheduleController)
        self.assertIsInstance(skyhook, SkyhookController)
        self.assertIsInstance(skyhook_roll, SkyhookRollController)
        self.assertIsInstance(skyhook_roll_yaw, SkyhookRollController)
        self.assertIsInstance(dryrun, SkyhookEstimatorDryRunController)
        self.assertTrue(skyhook.requires_suspension_state)
        self.assertFalse(skyhook_roll.config.enable_yaw_distribution)
        self.assertTrue(skyhook_roll_yaw.config.enable_yaw_distribution)
        self.assertEqual(
            "log_only",
            skyhook_roll_yaw.config.yaw_distribution_mode)
        self.assertEqual(
            "skyhook_roll_yaw_v2_log_only",
            skyhook_roll_yaw.config.controller_version)
        self.assertTrue(dryrun.requires_suspension_state)

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
            compact_help = "".join(help_text.split())
            self.assertIn("skyhook_roll", compact_help)
            self.assertIn("skyhook_roll_yaw", compact_help)
            self.assertIn("skyhook_estimator_dryrun", compact_help)
        self.assertIn("--skyhook-roll-config", tfpp_help)
        self.assertIn("--skyhook-roll-yaw-config", tfpp_help)
        self.assertIn("--skyhook-estimator-dryrun-config", tfpp_help)

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
                "roll_gate_global",
                "roll_ay_gate",
                "roll_rate_gate",
                "roll_angle_gate",
                "roll_softening_guard_ratio",
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
                "yaw_distribution_mode",
                "yaw_apply_enabled",
                "yaw_rate_ref",
                "yaw_rate_actual",
                "yaw_error",
                "yaw_error_norm",
                "yaw_activation",
                "front_distribution_nominal",
                "front_distribution_applied",
                "rear_distribution_applied",
                "front_distribution_proposed",
                "rear_distribution_proposed",
                "road_wheel_angle_rad",
                "yaw_reference_speed_mps",
                "skyhook_roll_yaw_rate_ref",
                "skyhook_roll_yaw_rate_actual",
                "skyhook_roll_yaw_error",
                "skyhook_roll_yaw_activation",
                "skyhook_roll_front_distribution_proposed",
                "skyhook_roll_rear_distribution_proposed",
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
                "skyhook_roll_spring_add_rr",
                "controller_version",
                "suspension_state_valid",
                "identity_fallback_that_would_have_occurred",
                "wheel_index_raw_fl",
                "compression_m_fl",
                "suspension_velocity_mps_fl",
                "contact_valid_fl",
                "wheel_in_air_fl",
                "field_valid_fl",
                "native_damper_rate_fl",
                "v_rel_extension_candidate_A_fl",
                "v_rel_extension_candidate_B_fl",
                "skyhook_product_candidate_A_fl",
                "skyhook_product_candidate_B_fl",
                "proposed_damper_scale_candidate_A_fl",
                "proposed_damper_scale_candidate_B_fl",
                "soft_mode_ratio_candidate_A",
                "hard_mode_ratio_candidate_A",
                "soft_mode_ratio_candidate_B",
                "hard_mode_ratio_candidate_B",
                "v_rel_extension_mps_fl",
                "C_native_fl",
                "C_required_fl",
                "semi_active_feasible_fl",
                "skyhook_only_target_damper_fl",
                "final_target_damper_fl",
                "roll_softening_guard_active_fl",
                "rate_limited_damper_fl",
                "soft_mode_fl",
                "hard_mode_fl",
                "neutral_mode_fl",
                "soft_mode_ratio",
                "hard_mode_ratio",
                "neutral_mode_ratio"):
            self.assertIn(field_name, suite.DIAGNOSTIC_FIELDS)

    def test_yaw_proposal_summary_reports_event_window(self):
        import transfuser_suspension_control_suite as suite

        fieldnames = (
            "yaw_distribution_mode",
            "yaw_rate_ref",
            "yaw_rate_actual",
            "yaw_error",
            "yaw_activation",
            "front_distribution_proposed",
            "rear_distribution_proposed",
            "front_distribution_applied",
        )
        rows = (
            {
                "yaw_distribution_mode": "log_only",
                "yaw_rate_ref": "0.5",
                "yaw_rate_actual": "0.1",
                "yaw_error": "0.4",
                "yaw_activation": "0.0",
                "front_distribution_proposed": "0.55",
                "rear_distribution_proposed": "0.45",
                "front_distribution_applied": "0.55",
            },
            {
                "yaw_distribution_mode": "log_only",
                "yaw_rate_ref": "0.5",
                "yaw_rate_actual": "0.2",
                "yaw_error": "0.3",
                "yaw_activation": "1.0",
                "front_distribution_proposed": "0.52",
                "rear_distribution_proposed": "0.48",
                "front_distribution_applied": "0.55",
            },
            {
                "yaw_distribution_mode": "disabled",
                "yaw_rate_ref": "99.0",
                "yaw_rate_actual": "99.0",
                "yaw_error": "99.0",
                "yaw_activation": "1.0",
                "front_distribution_proposed": "0.99",
                "rear_distribution_proposed": "0.01",
                "front_distribution_applied": "0.55",
            },
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "controller_diagnostics.csv")
            with open(path, "w", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            summary = suite.summarize_diagnostics_csv(path)

        self.assertEqual(2, summary["yaw_proposal_rows"])
        self.assertEqual(1, summary["yaw_activation_nonzero_rows"])
        self.assertAlmostEqual(0.5, summary["yaw_activation_nonzero_ratio"])
        self.assertAlmostEqual(0.535, summary[
            "front_distribution_proposed_mean"])
        self.assertEqual(1, summary["yaw_proposal_event_window_rows"])
        self.assertAlmostEqual(0.52, summary[
            "yaw_proposal_event_front_distribution_mean"])


if __name__ == "__main__":
    unittest.main()
