"""Registration checks for planning-aware v4 MPC Phase A suite profile."""

from __future__ import annotations

import csv
import os
import subprocess
import tempfile
import unittest

from suspension_control.controllers.planning_aware_mpc_phaseA import (
    PLANNING_AWARE_V4_MPC_PHASEA_VERSION,
    PLANNING_AWARE_V4_PHASEA_DIAGNOSTIC_FIELDS,
    PlanningAwareV4MpcPhaseAController,
)


def _sim_root() -> str:
    return os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "..",
        ".."))


class PlanningAwareV4MpcPhaseARegistrationTest(unittest.TestCase):

    def test_config_path_exists_and_loads_authority_phaseA_defaults(self):
        import transfuser_suspension_control_suite as suite

        path = suite.DEFAULT_PLANNING_AWARE_V4_MPC_PHASEA_AUTHORITY_CONFIG

        self.assertTrue(os.path.isfile(path), path)

        config = suite.build_planning_aware_v4_mpc_phaseA_config(path)

        self.assertEqual(
            PLANNING_AWARE_V4_MPC_PHASEA_VERSION,
            config.controller_version)
        self.assertEqual("authority", config.base_profile)
        self.assertEqual(1.5, config.horizon_s)
        self.assertEqual(2.0, config.max_horizon_s)
        self.assertEqual(0.75, config.min_damper_scale)
        self.assertEqual(1.25, config.max_damper_scale)
        self.assertEqual(2.0, config.max_rate_up_scale_per_s)
        self.assertEqual(1.6, config.max_rate_down_scale_per_s)
        self.assertEqual(1.0, config.spring_scale)
        self.assertFalse(config.use_skyhook_prior)
        self.assertEqual(0.0, config.W_prior)
        self.assertTrue(config.projection_aware_rerank_enabled)
        self.assertEqual(16, config.projection_rerank_top_k)
        self.assertTrue(config.projection_rerank_include_previous)
        self.assertTrue(config.projection_rerank_include_neutral)
        self.assertTrue(config.projection_rerank_include_raw_winner)
        self.assertTrue(config.projection_rerank_include_second_best)
        self.assertTrue(config.previous_candidate_enabled)
        self.assertTrue(config.neutral_candidate_enabled)
        self.assertFalse(config.comfort_margin_v4_enabled)
        self.assertFalse(config.selective_skyhook_prior_enabled)
        self.assertFalse(config.sequence_mpc_enabled)
        self.assertFalse(config.full_mpc_enabled)
        self.assertTrue(config.compute_time_diagnostics_enabled)
        self.assertEqual(0.0, config.debug_compute_sleep_ms)

    def test_suite_scenario_is_registered_as_s32(self):
        import transfuser_suspension_control_suite as suite

        by_key = suite.selected_scenarios(
            "planning_aware_v4_mpc_phaseA_authority")
        by_code = suite.selected_scenarios("S32")

        expected = {
            "name": "S32_planning_aware_v4_mpc_phaseA_authority",
            "label": "S32 planning-aware v4 MPC Phase A authority",
            "controller": "planning_aware_v4_mpc_phaseA_authority",
            "uses_suspension_api": True,
            "needs_suspension_state": True,
        }
        self.assertEqual([expected], by_key)
        self.assertEqual([expected], by_code)

    def test_make_controller_constructs_phaseA_controller(self):
        import transfuser_suspension_control_suite as suite

        args = suite.build_arg_parser().parse_args([])

        controller = suite.make_controller(
            "planning_aware_v4_mpc_phaseA_authority",
            args)

        self.assertIsInstance(controller, PlanningAwareV4MpcPhaseAController)
        self.assertEqual(
            "planning_aware_v4_mpc_phaseA_authority",
            controller.name)
        self.assertTrue(controller.requires_suspension_state)
        self.assertEqual(
            PLANNING_AWARE_V4_MPC_PHASEA_VERSION,
            controller.config.controller_version)
        self.assertEqual(2.0, controller.config.max_rate_up_scale_per_s)
        self.assertEqual(1.6, controller.config.max_rate_down_scale_per_s)

    def test_help_mentions_phaseA_scenario_and_config_arg(self):
        import lead_suspension_control_suite as lead_suite
        import transfuser_suspension_control_suite as suite

        for help_text in (
                suite.build_arg_parser().format_help(),
                lead_suite.build_arg_parser().format_help()):
            compact = "".join(help_text.split())
            self.assertIn(
                "planning_aware_v4_mpc_phaseA_authority",
                compact)
            self.assertTrue(
                "--planning-aware-v4-mpc-phaseA-authority-config" in
                help_text or
                "--planning-aware-v4-mpc-phasea-authority-config" in
                help_text)

    def test_phaseA_diagnostics_are_registered(self):
        import transfuser_suspension_control_suite as suite

        missing = [
            field for field in PLANNING_AWARE_V4_PHASEA_DIAGNOSTIC_FIELDS
            if field not in suite.DIAGNOSTIC_FIELDS
        ]

        self.assertEqual([], missing)

    def test_step07_runner_maps_phaseA_key_to_jsonl_sidecar(self):
        sim_root = _sim_root()
        script = os.path.join(sim_root, "scripts", "run_lead_step07_matrix.sh")

        result = subprocess.run(
            [
                "bash",
                script,
                "--mode",
                "smoke",
                "--scenarios",
                "LEAD_planning_aware_v4_mpc_phaseA_authority",
                "--seeds",
                "111",
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
            "scenario=LEAD_planning_aware_v4_mpc_phaseA_authority "
            "kind=sidecar",
            output)
        self.assertIn(
            "--scenarios planning_aware_v4_mpc_phaseA_authority",
            output)
        self.assertIn("--planning-provider jsonl", output)
        self.assertIn("--planning-preview-jsonl", output)

    def test_route_ablation_maps_phaseA_controller_to_lead_key(self):
        sim_root = _sim_root()
        script = os.path.join(
            sim_root,
            "scripts",
            "suspension",
            "run_suspension_route_ablation.sh")
        route = os.path.join(
            sim_root,
            "e2e_models",
            "carla_garage",
            "leaderboard",
            "data",
            "suspension_routes",
            "planning_aware_v2",
            "town04_r18_wp50_72_outer_scurve_ar_noscenario.xml")

        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    "bash",
                    script,
                    "--route",
                    route,
                    "--seed",
                    "111",
                    "--controllers",
                    "planning_aware_v4_mpc_phaseA_authority",
                    "--repetitions",
                    "1",
                    "--dry-run",
                    "--no-analyze",
                    "--output-dir",
                    temp_dir,
                ],
                cwd=sim_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=120)

            self.assertEqual(
                0,
                result.returncode,
                msg=result.stdout[-4000:] + result.stderr[-4000:])
            plan_path = os.path.join(temp_dir, "run_plan.csv")
            with open(plan_path, newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(1, len(rows))
        self.assertEqual(
            "planning_aware_v4_mpc_phaseA_authority",
            rows[0]["controller"])
        self.assertEqual(
            "LEAD_planning_aware_v4_mpc_phaseA_authority",
            rows[0]["scenario_key"])


if __name__ == "__main__":
    unittest.main()
