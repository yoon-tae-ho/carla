"""Focused Step 07 controller and suite-selection contract checks."""

from __future__ import annotations

import csv
import importlib.util
import math
import os
import subprocess
import sys
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
from suspension_control.controllers.planning_aware_risk_damping import (
    PlanningAwareRiskDampingController,
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


def _pard_context(planning=None, dt=0.05):
    return ControllerContext(
        state=VehicleState(
            frame=100,
            elapsed_seconds=5.0,
            dt=dt,
            speed=12.0),
        planning=planning if planning is not None else PlanningInfo.empty(),
        dt=dt)


def _pard_high_risk_planning():
    return PlanningInfo(
        available=True,
        frame=100,
        horizon_dt=0.1,
        target_speed=(12.0, 12.0, 12.0, 12.0),
        curvature=(0.05, 0.06, 0.06, 0.05),
        predicted_ay=(5.2, 5.4, 5.3, 5.1),
        predicted_ax=(0.0, 0.0, 0.0, 0.0),
        brake=(0.0, 0.0, 0.0, 0.0),
        steer=(0.2, 0.2, 0.2, 0.2))


def _pard_low_risk_planning():
    return PlanningInfo(
        available=True,
        frame=100,
        horizon_dt=0.1,
        target_speed=(12.0, 12.0, 12.0, 12.0),
        curvature=(0.0, 0.0, 0.0, 0.0),
        predicted_ay=(0.0, 0.0, 0.0, 0.0),
        predicted_ax=(0.0, 0.0, 0.0, 0.0),
        brake=(0.0, 0.0, 0.0, 0.0),
        steer=(0.0, 0.0, 0.0, 0.0))


def _damper_scales(output):
    return tuple(wheel.damper_scale for wheel in output.command.wheels)


def _spring_scales(output):
    return tuple(wheel.spring_scale for wheel in output.command.wheels)


def _sim_root():
    return os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "..",
        ".."))


def _load_step07_summary_tool():
    tool_path = os.path.join(
        _sim_root(),
        "carla-0.9.15",
        "PythonAPI",
        "taeho",
        "tools",
        "summarize_lead_step07_matrix.py")
    spec = importlib.util.spec_from_file_location(
        "summarize_lead_step07_matrix_for_test",
        tool_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
            "skyhook_roll,skyhook_roll_yaw,skyhook_estimator_dryrun,"
            "constant_damper_1p02,pard_v2_shadow,"
            "pard_v2_active_ultra_safe,pard_v2_active_safe_1p06,"
            "pard_v2_active_aggressive_0p75_1p25")

        self.assertEqual(
            [
                "S15_constant_damper_1p03",
                "S16_target_speed_schedule_v0",
                "S17_target_speed_schedule_shadow",
                "S18_target_speed_schedule_v0_safe",
                "S19_skyhook_roll",
                "S20_skyhook_roll_yaw",
                "S21_skyhook_estimator_dryrun",
                "S22_constant_damper_1p02",
                "S23_pard_v2_shadow",
                "S24_pard_v2_active_ultra_safe",
                "S25_pard_v2_active_safe_1p06",
                "S26_pard_v2_active_aggressive_0p75_1p25",
            ],
            [scenario["name"] for scenario in scenarios])

    def test_new_scenarios_are_selectable_by_code(self):
        import transfuser_suspension_control_suite as suite

        scenarios = suite.selected_scenarios("S19,S20,S21,S22,S23,S24,S25,S26")

        self.assertEqual(
            [
                "S19_skyhook_roll",
                "S20_skyhook_roll_yaw",
                "S21_skyhook_estimator_dryrun",
                "S22_constant_damper_1p02",
                "S23_pard_v2_shadow",
                "S24_pard_v2_active_ultra_safe",
                "S25_pard_v2_active_safe_1p06",
                "S26_pard_v2_active_aggressive_0p75_1p25",
            ],
            [scenario["name"] for scenario in scenarios])

    def test_pard_scenario_aliases_select_ultra_safe(self):
        import transfuser_suspension_control_suite as suite

        scenarios = suite.selected_scenarios("pard_v2,planning_risk_damping")

        self.assertEqual(
            [
                "S24_pard_v2_active_ultra_safe",
                "S24_pard_v2_active_ultra_safe",
            ],
            [scenario["name"] for scenario in scenarios])

    def test_make_controller_builds_new_controllers(self):
        import transfuser_suspension_control_suite as suite

        args = suite.build_arg_parser().parse_args([])

        constant = suite.make_controller("constant_scale", args)
        constant_1p02 = suite.make_controller("constant_scale_1p02", args)
        schedule = suite.make_controller("target_speed_schedule", args)
        shadow = suite.make_controller("target_speed_schedule_shadow", args)
        safe = suite.make_controller("target_speed_schedule_safe", args)
        skyhook = suite.make_controller("skyhook", args)
        skyhook_roll = suite.make_controller("skyhook_roll", args)
        skyhook_roll_yaw = suite.make_controller("skyhook_roll_yaw", args)
        dryrun = suite.make_controller("skyhook_estimator_dryrun", args)
        pard_shadow = suite.make_controller(
            "planning_aware_risk_damping_shadow",
            args)
        pard_ultra = suite.make_controller(
            "planning_aware_risk_damping_ultra_safe",
            args)
        pard_safe = suite.make_controller(
            "planning_aware_risk_damping_safe_1p06",
            args)
        pard_aggressive = suite.make_controller(
            "planning_aware_risk_damping_aggressive_0p75_1p25",
            args)
        pard_alias = suite.make_controller("pard_v2", args)

        self.assertIsInstance(constant, ConstantScaleController)
        self.assertIsInstance(constant_1p02, ConstantScaleController)
        self.assertIsInstance(schedule, TargetSpeedScheduleController)
        self.assertIsInstance(shadow, TargetSpeedScheduleController)
        self.assertIsInstance(safe, TargetSpeedScheduleController)
        self.assertIsInstance(skyhook, SkyhookController)
        self.assertIsInstance(skyhook_roll, SkyhookRollController)
        self.assertIsInstance(skyhook_roll_yaw, SkyhookRollController)
        self.assertIsInstance(dryrun, SkyhookEstimatorDryRunController)
        self.assertIsInstance(pard_shadow, PlanningAwareRiskDampingController)
        self.assertIsInstance(pard_ultra, PlanningAwareRiskDampingController)
        self.assertIsInstance(pard_safe, PlanningAwareRiskDampingController)
        self.assertIsInstance(pard_aggressive, PlanningAwareRiskDampingController)
        self.assertIsInstance(pard_alias, PlanningAwareRiskDampingController)
        self.assertEqual(1.02, constant_1p02.config.damper_scale)
        self.assertTrue(pard_shadow.config.shadow_mode)
        self.assertEqual(1.035, pard_ultra.config.damper_max)
        self.assertEqual(1.06, pard_safe.config.damper_max)
        self.assertEqual(0.75, pard_aggressive.config.damper_min)
        self.assertEqual(1.25, pard_aggressive.config.damper_max)
        self.assertEqual(
            "centered_range",
            pard_aggressive.config.damper_schedule_mode)
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

    def test_pard_shadow_config_applies_identity_but_logs_would_command(self):
        import transfuser_suspension_control_suite as suite

        args = suite.build_arg_parser().parse_args([])
        controller = suite.make_controller(
            "planning_aware_risk_damping_shadow",
            args)

        output = controller.compute(_pard_context(_pard_high_risk_planning()))

        self.assertEqual((1.0, 1.0, 1.0, 1.0), _damper_scales(output))
        self.assertEqual((1.0, 1.0, 1.0, 1.0), _spring_scales(output))
        self.assertGreater(output.diagnostics["uniform_damper_would"], 1.0)
        self.assertEqual(1.0, output.diagnostics["uniform_damper_cmd"])
        self.assertEqual(1, output.diagnostics["shadow_mode"])

    def test_pard_ultra_safe_config_bounds_damper(self):
        import transfuser_suspension_control_suite as suite

        args = suite.build_arg_parser().parse_args([])
        controller = suite.make_controller(
            "planning_aware_risk_damping_ultra_safe",
            args)

        output = controller.compute(_pard_context(_pard_high_risk_planning()))

        self.assertLessEqual(_damper_scales(output)[0], 1.035)
        self.assertEqual(1.035, controller.config.damper_max)
        self.assertEqual("uniform", controller.config.output_mode)

    def test_pard_aggressive_config_can_soften_and_stiffen(self):
        import transfuser_suspension_control_suite as suite

        args = suite.build_arg_parser().parse_args([])
        controller = suite.make_controller(
            "planning_aware_risk_damping_aggressive_0p75_1p25",
            args)

        low_output = None
        for _ in range(12):
            low_output = controller.compute(_pard_context(_pard_low_risk_planning()))
        self.assertIsNotNone(low_output)
        self.assertLess(_damper_scales(low_output)[0], 1.0)
        self.assertGreaterEqual(_damper_scales(low_output)[0], 0.75)

        high_output = None
        for _ in range(24):
            high_output = controller.compute(
                _pard_context(_pard_high_risk_planning()))
        self.assertIsNotNone(high_output)
        self.assertGreater(_damper_scales(high_output)[0], 1.0)
        self.assertLessEqual(_damper_scales(high_output)[0], 1.25)
        self.assertEqual(
            "centered_range",
            high_output.diagnostics["damper_schedule_mode"])

    def test_constant_damper_1p02_config_applies_expected_scale(self):
        import transfuser_suspension_control_suite as suite

        args = suite.build_arg_parser().parse_args([])
        controller = suite.make_controller("constant_scale_1p02", args)

        output = controller.compute(_context())

        self.assertEqual((1.02, 1.02, 1.02, 1.02), _damper_scales(output))
        self.assertEqual((1.0, 1.0, 1.0, 1.0), _spring_scales(output))

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
            self.assertIn("constant_damper_1p02", compact_help)
            self.assertIn("pard_v2_shadow", compact_help)
            self.assertIn("pard_v2_active_ultra_safe", compact_help)
            self.assertIn("pard_v2_active_safe_1p06", compact_help)
        self.assertIn("--skyhook-roll-config", tfpp_help)
        self.assertIn("--skyhook-roll-yaw-config", tfpp_help)
        self.assertIn("--skyhook-estimator-dryrun-config", tfpp_help)
        self.assertIn("--constant-damper-1p02-config", tfpp_help)
        self.assertIn(
            "--planning-aware-risk-damping-ultra-safe-config",
            tfpp_help)
        self.assertIn("--planning-aware-risk-damping-shadow-config", tfpp_help)
        self.assertIn(
            "--planning-aware-risk-damping-safe-1p06-config",
            tfpp_help)

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
                "neutral_mode_ratio",
                "controller_name",
                "shadow_mode",
                "output_mode",
                "planning_valid",
                "planning_horizon_dt",
                "fallback_active",
                "fallback_reason",
                "suspension_state_available",
                "preview_points_used",
                "motion_gate",
                "curvature_source",
                "curvature_abs_max",
                "curvature_abs_near_topk",
                "predicted_ay_source",
                "predicted_ay_abs_max",
                "predicted_ay_abs_near_topk",
                "predicted_ax_min",
                "brake_preview_max",
                "target_speed_min_1s",
                "target_speed_drop_1s",
                "steer_preview_abs_max",
                "risk_source_mask",
                "risk_confidence",
                "r_ay_preview",
                "r_kappa_preview",
                "r_steer_preview",
                "r_lat_preview",
                "r_state_ay",
                "r_roll_rate",
                "r_roll_angle",
                "r_yaw_rate",
                "r_state_raw",
                "r_state_trim",
                "r_pred_decel",
                "r_brake_preview",
                "r_current_brake",
                "r_speed_drop",
                "r_brake_preview_combined",
                "risk_preview",
                "risk_raw",
                "risk_smooth",
                "r_lat_smooth",
                "r_brake_smooth",
                "uniform_damper_desired",
                "uniform_damper_limited",
                "uniform_damper_cmd",
                "uniform_damper_would",
                "front_damper_cmd",
                "rear_damper_cmd",
                "front_damper_would",
                "rear_damper_would",
                "rate_limit_active_uniform",
                "rate_limit_active_front",
                "rate_limit_active_rear",
                "event_held",
                "spring_FL",
                "spring_FR",
                "spring_RL",
                "spring_RR",
                "damper_FL",
                "damper_FR",
                "damper_RL",
                "damper_RR",
                "bbox_num_boxes",
                "bbox_num_vehicle_boxes",
                "bbox_num_pedestrian_boxes",
                "bbox_min_forward_distance_m",
                "state_speed",
                "state_local_ay",
                "state_roll",
                "state_roll_rate",
                "state_pitch",
                "state_pitch_rate",
                "state_yaw_rate",
                "state_steer",
                "state_brake",
                "exception_type"):
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


class Step07RunnerMappingTest(unittest.TestCase):

    def test_matrix_runner_dry_run_maps_new_scenarios(self):
        sim_root = _sim_root()
        script = os.path.join(sim_root, "scripts", "run_lead_step07_matrix.sh")
        scenarios = ",".join((
            "LEAD_constant_damper_1.02",
            "LEAD_pard_v2_shadow",
            "LEAD_pard_v2_active_ultra_safe",
            "LEAD_pard_v2_active_safe_1p06",
            "LEAD_pard_v2_active_aggressive_0p75_1p25",
        ))

        result = subprocess.run(
            [
                "bash",
                script,
                "--mode",
                "smoke",
                "--seeds",
                "101",
                "--scenarios",
                scenarios,
                "--dry-run",
                "--skip-preflight",
                "--skip-summary",
                "--no-restart-carla-per-run",
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
        output = result.stdout
        self.assertIn(
            "scenario=LEAD_constant_damper_1.02 kind=sidecar",
            output)
        self.assertIn(
            "scenario=LEAD_pard_v2_shadow kind=sidecar",
            output)
        self.assertIn(
            "scenario=LEAD_pard_v2_active_ultra_safe kind=sidecar",
            output)
        self.assertIn(
            "scenario=LEAD_pard_v2_active_safe_1p06 kind=sidecar",
            output)
        self.assertIn(
            "scenario=LEAD_pard_v2_active_aggressive_0p75_1p25 kind=sidecar",
            output)
        self.assertIn("--scenarios constant_damper_1p02", output)
        self.assertIn("--scenarios pard_v2_shadow", output)
        self.assertIn("--scenarios pard_v2_active_ultra_safe", output)
        self.assertIn("--scenarios pard_v2_active_safe_1p06", output)
        self.assertIn(
            "--scenarios pard_v2_active_aggressive_0p75_1p25",
            output)
        self.assertGreaterEqual(output.count("--planning-provider jsonl"), 4)
        self.assertGreaterEqual(output.count("--planning-preview-jsonl"), 4)

    def test_parallel_runner_synthesizes_pard_jsonl_sidecar_plan(self):
        sim_root = _sim_root()
        script = os.path.join(
            sim_root,
            "scripts",
            "run_lead_step07_parallel_matrix.sh")
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = os.path.join(temp_dir, "parallel")

            result = subprocess.run(
                [
                    "bash",
                    script,
                    "--mode",
                    "smoke",
                    "--seeds",
                    "101",
                    "--scenarios",
                    "LEAD_pard_v2_shadow",
                    "--shards",
                    "1",
                    "--output-dir",
                    output_dir,
                    "--dry-run",
                    "--skip-preflight",
                    "--skip-summary",
                    "--no-restart-carla-per-run",
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
            plan_path = os.path.join(output_dir, "step07_runner_plan.csv")
            with open(plan_path, newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(1, len(rows))
        self.assertEqual("LEAD_pard_v2_shadow", rows[0]["scenario_key"])
        self.assertEqual("sidecar", rows[0]["run_kind"])
        self.assertTrue(rows[0]["planning_jsonl_host"].endswith(
            "planning_preview.jsonl"))


class Step07SummaryPardGateTest(unittest.TestCase):

    def _write_csv(self, path, rows):
        fields = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fields.append(key)
        with open(path, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def _base_gate_row(self, tool, scenario, summary):
        row = {
            "seed": "101",
            "scenario_key": scenario,
            "checkpoint_exists": 1,
            "entry_status": "Finished",
            "global_status": "Perfect",
            "duration_game": "10.0",
            "score_composed": "100.0",
            "score_route": "100.0",
            "score_penalty": "1.0",
            "major_infraction_count": 0,
            "planning_jsonl_rows": (
                12 if scenario in tool.JSONL_SCENARIOS else 0),
            "sidecar_fatal_errors": 0,
            "verify_warning_count": 0,
            "verify_hard_error_count": 0,
            "sidecar_actor_removed_warning_count": 0,
            "sidecar_runtime_hard_error_count": 0,
            "invalid_run": 0,
        }
        row.update(summary)
        return row

    def test_summary_classifies_new_pard_and_constant_scenarios(self):
        tool = _load_step07_summary_tool()

        for scenario in (
                "LEAD_pard_v2_shadow",
                "LEAD_pard_v2_active_ultra_safe",
                "LEAD_pard_v2_active_safe_1p06",
                "LEAD_pard_v2_active_aggressive_0p75_1p25"):
            self.assertIn(scenario, tool.SIDECAR_SCENARIOS)
            self.assertIn(scenario, tool.JSONL_SCENARIOS)
            self.assertIn(scenario, tool.JSONL_SIDECAR_SCENARIOS)
            self.assertIn(scenario, tool.PARD_V2_SCENARIOS)

        self.assertIn(
            "LEAD_constant_damper_1.02",
            tool.CONSTANT_DAMPING_SCENARIOS)
        self.assertIn(
            "LEAD_constant_damper_1.03",
            tool.CONSTANT_DAMPING_SCENARIOS)
        self.assertIn(
            "LEAD_pard_v2_active_ultra_safe",
            tool.SMOKE_SCENARIOS)
        self.assertNotIn(
            "LEAD_pard_v2_active_safe_1p06",
            tool.SMOKE_SCENARIOS)
        self.assertNotIn(
            "LEAD_pard_v2_active_safe_1p06",
            tool.FULL_SCENARIOS)
        self.assertNotIn(
            "LEAD_pard_v2_active_aggressive_0p75_1p25",
            tool.SMOKE_SCENARIOS)
        self.assertNotIn(
            "LEAD_pard_v2_active_aggressive_0p75_1p25",
            tool.FULL_SCENARIOS)

        with tempfile.TemporaryDirectory() as temp_dir:
            args = type("Args", (), {
                "mode": "smoke",
                "seeds": "101",
                "scenarios": "LEAD_pard_v2_shadow,"
                             "LEAD_constant_damper_1.02",
            })()
            plan = tool.expected_plan(args, temp_dir)

        by_scenario = {row["scenario_key"]: row for row in plan}
        self.assertEqual(
            "sidecar",
            by_scenario["LEAD_pard_v2_shadow"]["run_kind"])
        self.assertTrue(by_scenario["LEAD_pard_v2_shadow"][
            "planning_jsonl_host"].endswith("planning_preview.jsonl"))
        self.assertEqual(
            "sidecar",
            by_scenario["LEAD_constant_damper_1.02"]["run_kind"])
        self.assertEqual(
            "",
            by_scenario["LEAD_constant_damper_1.02"][
                "planning_jsonl_host"])

    def test_pard_shadow_synthetic_csv_passes_sanity_gate(self):
        tool = _load_step07_summary_tool()
        row = {
            "planning_available": "1",
            "planning_source": "jsonl",
            "planning_age_frames": "4",
            "planning_jsonl_malformed_lines": "0",
            "planning_jsonl_rejected_messages": "0",
            "planning_jsonl_read_errors": "0",
            "fallback_active": "0",
            "shadow_mode": "1",
            "uniform_damper_cmd": "1.0",
            "uniform_damper_would": "1.021",
            "spring_FL": "1.0",
            "spring_FR": "1.0",
            "spring_RL": "1.0",
            "spring_RR": "1.0",
            "damper_FL": "1.0",
            "damper_FR": "1.0",
            "damper_RL": "1.0",
            "damper_RR": "1.0",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "controller_diagnostics.csv")
            self._write_csv(path, [row])

            summary = tool.summarize_diagnostics(
                path,
                "LEAD_pard_v2_shadow",
                5.0)

        self.assertEqual(1, summary["controller_diagnostic_rows"])
        self.assertEqual(1, summary["planning_available_rows"])
        self.assertEqual(4.0, summary["planning_age_max"])
        self.assertEqual(1, summary["planning_age_gate_ok"])
        self.assertEqual(0, summary["diagnostic_nonfinite_values"])
        self.assertEqual(0, summary["spring_identity_violations"])
        self.assertEqual(1, summary["pard_rows"])
        self.assertEqual(1, summary["pard_shadow_rows"])
        self.assertEqual(0, summary["pard_shadow_actual_identity_violations"])
        self.assertEqual(1, summary["pard_would_command_rows"])
        self.assertAlmostEqual(1.0, summary["pard_damper_max"])
        self.assertAlmostEqual(1.021, summary["pard_would_damper_max"])

        gate_rows, _ = tool.evaluate_gates(
            [self._base_gate_row(
                tool,
                "LEAD_pard_v2_shadow",
                summary)],
            2.0,
            1.0,
            5.0)
        self.assertEqual("pass", gate_rows[0]["gate_status"])
        self.assertEqual("", gate_rows[0]["failed_checks"])

    def test_pard_active_synthetic_csv_reports_gate_failures(self):
        tool = _load_step07_summary_tool()
        row = {
            "planning_available": "1",
            "planning_source": "jsonl",
            "planning_age_frames": "6",
            "planning_jsonl_malformed_lines": "0",
            "planning_jsonl_rejected_messages": "0",
            "planning_jsonl_read_errors": "0",
            "fallback_active": "0",
            "shadow_mode": "0",
            "uniform_damper_cmd": "1.04",
            "uniform_damper_would": "1.04",
            "spring_FL": "0.99",
            "spring_FR": "1.0",
            "spring_RL": "1.0",
            "spring_RR": "1.0",
            "damper_FL": "0.99",
            "damper_FR": "1.04",
            "damper_RL": "1.04",
            "damper_RR": "1.04",
            "risk_smooth": "nan",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "controller_diagnostics.csv")
            self._write_csv(path, [row])

            summary = tool.summarize_diagnostics(
                path,
                "LEAD_pard_v2_active_ultra_safe",
                5.0)

        self.assertEqual(1, summary["diagnostic_nonfinite_values"])
        self.assertEqual(1, summary["spring_identity_violations"])
        self.assertEqual(0, summary["planning_age_gate_ok"])
        self.assertLess(summary["pard_damper_min"], 1.0)
        self.assertGreater(summary["pard_damper_max"], 1.035)
        self.assertGreaterEqual(summary["pard_damper_bound_violations"], 2)

        gate_rows, _ = tool.evaluate_gates(
            [self._base_gate_row(
                tool,
                "LEAD_pard_v2_active_ultra_safe",
                summary)],
            2.0,
            1.0,
            5.0)
        failed = set(gate_rows[0]["failed_checks"].split(";"))
        self.assertIn("diagnostic_nonfinite", failed)
        self.assertIn("planning_age", failed)
        self.assertIn("spring_not_identity", failed)
        self.assertIn("pard_damper_bounds", failed)

    def test_pard_aggressive_synthetic_csv_allows_centered_range(self):
        tool = _load_step07_summary_tool()
        row = {
            "planning_available": "1",
            "planning_source": "jsonl",
            "planning_age_frames": "1",
            "planning_jsonl_malformed_lines": "0",
            "planning_jsonl_rejected_messages": "0",
            "planning_jsonl_read_errors": "0",
            "fallback_active": "0",
            "shadow_mode": "0",
            "uniform_damper_cmd": "0.76",
            "uniform_damper_would": "0.76",
            "spring_FL": "1.0",
            "spring_FR": "1.0",
            "spring_RL": "1.0",
            "spring_RR": "1.0",
            "damper_FL": "0.76",
            "damper_FR": "0.76",
            "damper_RL": "0.76",
            "damper_RR": "0.76",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "controller_diagnostics.csv")
            self._write_csv(path, [row])

            summary = tool.summarize_diagnostics(
                path,
                "LEAD_pard_v2_active_aggressive_0p75_1p25",
                5.0)

        self.assertEqual(0, summary["pard_damper_bound_violations"])
        self.assertAlmostEqual(0.76, summary["pard_damper_min"])
        self.assertAlmostEqual(0.76, summary["pard_damper_max"])
        self.assertEqual(1, summary["pard_would_command_rows"])

        gate_rows, _ = tool.evaluate_gates(
            [self._base_gate_row(
                tool,
                "LEAD_pard_v2_active_aggressive_0p75_1p25",
                summary)],
            2.0,
            1.0,
            5.0)
        self.assertEqual("pass", gate_rows[0]["gate_status"])

    def test_pard_aggressive_synthetic_csv_flags_out_of_range(self):
        tool = _load_step07_summary_tool()
        row = {
            "planning_available": "1",
            "planning_source": "jsonl",
            "planning_age_frames": "1",
            "planning_jsonl_malformed_lines": "0",
            "planning_jsonl_rejected_messages": "0",
            "planning_jsonl_read_errors": "0",
            "fallback_active": "0",
            "shadow_mode": "0",
            "uniform_damper_cmd": "1.0",
            "uniform_damper_would": "1.0",
            "spring_FL": "1.0",
            "spring_FR": "1.0",
            "spring_RL": "1.0",
            "spring_RR": "1.0",
            "damper_FL": "0.74",
            "damper_FR": "1.26",
            "damper_RL": "1.0",
            "damper_RR": "1.0",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "controller_diagnostics.csv")
            self._write_csv(path, [row])

            summary = tool.summarize_diagnostics(
                path,
                "LEAD_pard_v2_active_aggressive_0p75_1p25",
                5.0)

        self.assertEqual(2, summary["pard_damper_bound_violations"])
        gate_rows, _ = tool.evaluate_gates(
            [self._base_gate_row(
                tool,
                "LEAD_pard_v2_active_aggressive_0p75_1p25",
                summary)],
            2.0,
            1.0,
            5.0)
        self.assertIn(
            "pard_damper_bounds",
            gate_rows[0]["failed_checks"].split(";"))

    def test_constant_damping_synthetic_csv_checks_expected_scales(self):
        tool = _load_step07_summary_tool()
        with tempfile.TemporaryDirectory() as temp_dir:
            path_1p02 = os.path.join(temp_dir, "constant_1p02.csv")
            path_1p03 = os.path.join(temp_dir, "constant_1p03.csv")
            path_bad = os.path.join(temp_dir, "constant_bad.csv")
            self._write_csv(path_1p02, [{
                "damper_scale": "1.02",
                "spring_scale": "1.0",
            }])
            self._write_csv(path_1p03, [{
                "damper_scale": "1.03",
                "spring_scale": "1.0",
            }])
            self._write_csv(path_bad, [{
                "damper_scale": "1.03",
                "spring_scale": "1.0",
            }])

            summary_1p02 = tool.summarize_diagnostics(
                path_1p02,
                "LEAD_constant_damper_1.02",
                5.0)
            summary_1p03 = tool.summarize_diagnostics(
                path_1p03,
                "LEAD_constant_damper_1.03",
                5.0)
            summary_bad = tool.summarize_diagnostics(
                path_bad,
                "LEAD_constant_damper_1.02",
                5.0)

        self.assertEqual(0, summary_1p02["damper_bound_violations"])
        self.assertEqual(0, summary_1p03["damper_bound_violations"])
        self.assertAlmostEqual(1.02, summary_1p02["damper_mean"])
        self.assertAlmostEqual(1.03, summary_1p03["damper_mean"])
        self.assertEqual(1, summary_bad["damper_bound_violations"])


if __name__ == "__main__":
    unittest.main()
