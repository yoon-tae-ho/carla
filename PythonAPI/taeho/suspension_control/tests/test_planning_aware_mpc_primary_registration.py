"""Registration checks for planning-aware v3 MPC-primary suite profiles."""

from __future__ import annotations

import os
import unittest

from suspension_control.controllers.planning_aware_mpc_primary import (
    PLANNING_AWARE_V3_MPC_PRIMARY_VERSION,
    PLANNING_AWARE_V3_REQUIRED_DIAGNOSTIC_FIELDS,
    PlanningAwareV3MpcPrimaryController,
)


class PlanningAwareV3MpcPrimaryRegistrationTest(unittest.TestCase):

    def test_config_paths_exist_and_load_expected_profiles(self):
        import transfuser_suspension_control_suite as suite

        paths = (
            suite.DEFAULT_PLANNING_AWARE_V3_MPC_PRIMARY_SAFE_CONFIG,
            suite.DEFAULT_PLANNING_AWARE_V3_MPC_PRIMARY_AUTHORITY_CONFIG,
            suite.DEFAULT_PLANNING_AWARE_V3_MPC_SKYHOOK_PRIOR_CONFIG,
        )
        for path in paths:
            self.assertTrue(os.path.isfile(path), path)

        safe = suite.build_planning_aware_v3_mpc_primary_config(paths[0])
        authority = suite.build_planning_aware_v3_mpc_primary_config(paths[1])
        skyhook_prior = suite.build_planning_aware_v3_mpc_primary_config(paths[2])

        self.assertEqual(1.2, safe.max_rate_up_scale_per_s)
        self.assertEqual(1.2, safe.max_rate_down_scale_per_s)
        self.assertFalse(safe.use_skyhook_prior)
        self.assertEqual(2.0, authority.max_rate_up_scale_per_s)
        self.assertEqual(1.6, authority.max_rate_down_scale_per_s)
        self.assertFalse(authority.use_skyhook_prior)
        self.assertTrue(skyhook_prior.use_skyhook_prior)
        self.assertGreater(skyhook_prior.W_prior, 0.0)

    def test_suite_scenarios_are_registered_without_s27_s28_collision(self):
        import transfuser_suspension_control_suite as suite

        scenarios = suite.selected_scenarios(
            "planning_aware,skyhook_roll_v3,"
            "planning_aware_v3_mpc_primary_safe,"
            "planning_aware_v3_mpc_primary_authority,"
            "planning_aware_v3_mpc_skyhook_prior")
        names = [scenario["name"] for scenario in scenarios]

        self.assertEqual(
            [
                "S27_planning_aware",
                "S28_skyhook_roll_v3",
                "S29_planning_aware_v3_mpc_primary_safe",
                "S30_planning_aware_v3_mpc_primary_authority",
                "S31_planning_aware_v3_mpc_skyhook_prior",
            ],
            names)
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual("planning_aware", scenarios[0]["controller"])
        self.assertEqual("skyhook_roll_v3", scenarios[1]["controller"])
        for scenario in scenarios[2:]:
            self.assertTrue(scenario["uses_suspension_api"])
            self.assertTrue(scenario["needs_suspension_state"])

    def test_primary_alias_selects_safe_profile(self):
        import transfuser_suspension_control_suite as suite

        scenarios = suite.selected_scenarios("planning_aware_v3_mpc_primary")

        self.assertEqual(1, len(scenarios))
        self.assertEqual(
            "S29_planning_aware_v3_mpc_primary_safe",
            scenarios[0]["name"])
        self.assertEqual(
            "planning_aware_v3_mpc_primary_safe",
            scenarios[0]["controller"])

    def test_make_controller_constructs_each_profile(self):
        import transfuser_suspension_control_suite as suite

        args = suite.build_arg_parser().parse_args([])
        safe = suite.make_controller("planning_aware_v3_mpc_primary_safe", args)
        alias = suite.make_controller("planning_aware_v3_mpc_primary", args)
        authority = suite.make_controller(
            "planning_aware_v3_mpc_primary_authority",
            args)
        skyhook_prior = suite.make_controller(
            "planning_aware_v3_mpc_skyhook_prior",
            args)

        for controller in (safe, alias, authority, skyhook_prior):
            self.assertIsInstance(controller, PlanningAwareV3MpcPrimaryController)
            self.assertEqual(
                "planning_aware_v3_mpc_primary",
                controller.name)
            self.assertTrue(controller.requires_suspension_state)
            self.assertEqual(
                PLANNING_AWARE_V3_MPC_PRIMARY_VERSION,
                controller.config.controller_version)

        self.assertEqual(1.2, safe.config.max_rate_up_scale_per_s)
        self.assertEqual(1.2, alias.config.max_rate_up_scale_per_s)
        self.assertEqual(2.0, authority.config.max_rate_up_scale_per_s)
        self.assertTrue(skyhook_prior.config.use_skyhook_prior)
        self.assertGreater(skyhook_prior.config.W_prior, 0.0)

    def test_required_diagnostics_are_registered(self):
        import transfuser_suspension_control_suite as suite

        missing = [
            field for field in PLANNING_AWARE_V3_REQUIRED_DIAGNOSTIC_FIELDS
            if field not in suite.DIAGNOSTIC_FIELDS
        ]
        self.assertEqual([], missing)
        for field in (
                "controller_version",
                "mpc_feasible_candidate_count",
                "mpc_final_rate_guard_active",
                "mpc_final_finite_guard_active",
                "mpc_ltr_soft_guard_active",
                "mpc_ltr_hard_guard_active",
                "mpc_optimizer_failed_reason",
                "planning_aware_v3_shadow_identity_fallback",
                "planning_aware_v3_shadow_fallback_reason",
                "planning_aware_v3_outer_side_sign_from_ay"):
            self.assertIn(field, suite.DIAGNOSTIC_FIELDS)

    def test_help_mentions_scenarios_and_config_paths(self):
        import lead_suspension_control_suite as lead_suite
        import transfuser_suspension_control_suite as suite

        tfpp_help = suite.build_arg_parser().format_help()
        lead_help = lead_suite.build_arg_parser().format_help()

        for help_text in (tfpp_help, lead_help):
            compact = "".join(help_text.split())
            self.assertIn("planning_aware_v3_mpc_primary_safe", compact)
            self.assertIn("planning_aware_v3_mpc_primary_authority", compact)
            self.assertIn("planning_aware_v3_mpc_skyhook_prior", compact)
        self.assertIn("--planning-aware-v3-mpc-primary-safe-config", tfpp_help)
        self.assertIn(
            "--planning-aware-v3-mpc-primary-authority-config",
            tfpp_help)
        self.assertIn(
            "--planning-aware-v3-mpc-skyhook-prior-config",
            tfpp_help)


if __name__ == "__main__":
    unittest.main()
