#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON:-python3}"

"${PYTHON_BIN}" -m py_compile \
  suspension_control/controllers/base.py \
  suspension_control/controllers/rl_residual.py \
  suspension_control/rl/action_projection.py \
  suspension_control/rl/carla_backend.py \
  suspension_control/rl/carla_online_env.py \
  suspension_control/rl/carla_route_backend.py \
  suspension_control/rl/carla_route_env.py \
  suspension_control/rl/diagnostics.py \
  suspension_control/rl/task_info.py \
  suspension_control/rl/observations.py \
  suspension_control/rl/reward.py \
  suspension_control/rl/reward_calibration.py \
  suspension_control/rl/reward_sanity.py \
  suspension_control/rl/policy.py \
  suspension_control/rl/policy_export.py \
  suspension_control/rl/phase4_acceptance.py \
  suspension_control/rl/phase4b_policy_eval_acceptance.py \
  suspension_control/rl/phase4d_train_acceptance.py \
  suspension_control/rl/phase4d_train_manifest.py \
  suspension_control/rl/phase4d_eval_artifact.py \
  suspension_control/rl/phase4d_eval_reference.py \
  suspension_control/rl/phase4d_compare_only.py \
  suspension_control/rl/phase4d_policy_selection.py \
  suspension_control/rl/phase4d_stageA_repeat.py \
  suspension_control/rl/phase4d_authority_sensitivity_common.py \
  suspension_control/rl/phase4d_action_authority_sweep.py \
  suspension_control/rl/phase4d_scripted_residual_sensitivity.py \
  suspension_control/rl/phase4d_authority_sensitivity_report.py \
  suspension_control/rl/phase4_route_health.py \
  suspension_control/rl/phase4c_horizon_sweep.py \
  suspension_control/rl/train_sac.py \
  suspension_control/rl/train_real_carla_sac.py \
  suspension_control/rl/carla_env.py \
  suspension_control/rl/rollout_logger.py \
  suspension_control/runtime/planning_provider.py \
  suspension_control/runtime/route_progress.py \
  suspension_control/rl/evaluate_policy.py \
  suspension_control/metrics/comfort.py \
  suspension_control/metrics/stability.py \
  suspension_zero_residual_equivalence.py \
  transfuser_suspension_control_suite.py

if "${PYTHON_BIN}" -c "import pytest" >/dev/null 2>&1; then
  "${PYTHON_BIN}" -m pytest -q \
    tests/test_residual_action_projector.py \
    tests/test_carla_rl_env_fake_backend.py \
    tests/test_phase4_fake_backend_env.py \
    tests/test_phase4_rollout_logging.py \
    tests/test_phase4_live_backend_contract.py \
    tests/test_phase4_normalizer_export.py \
    tests/test_phase4_policy_export_adapter.py \
    tests/test_phase4_training_cli_fake.py \
    tests/test_phase4_acceptance_report.py \
    tests/test_phase4_eval_acceptance.py \
    tests/test_phase4b_policy_eval_acceptance.py \
    tests/test_phase4b_split_reference_acceptance.py \
    tests/test_phase4d_infraction_parser.py \
    tests/test_phase4d_train_acceptance.py \
    tests/test_phase4d_train_manifest.py \
    tests/test_phase4d_train_only_runner.py \
    tests/test_phase4d_eval_artifact.py \
    tests/test_phase4d_eval_s4_runner.py \
    tests/test_phase4d_eval_reference.py \
    tests/test_phase4d_eval_s8_runner.py \
    tests/test_phase4d_compare_only.py \
    tests/test_phase4d_compare_wrapper.py \
    tests/test_phase4d_policy_selection.py \
    tests/test_phase4d_stageA_repeat.py \
    tests/test_phase4d_stageA_runner.py \
    tests/test_phase4d_stageA_aggregate.py \
    tests/test_phase4d_authority_sensitivity_plan.py \
    tests/test_phase4d_runner_scripts.py \
    tests/test_phase4b_policy_eval_runner.py \
    tests/test_phase4_route_health_analyzer.py \
    tests/test_phase4c_horizon_sweep_summary.py \
    tests/test_phase4c_horizon_sweep_runner.py \
    tests/test_phase4_runner_help.py \
    tests/test_rl_residual_controller.py \
    tests/test_rl_observations.py \
    tests/test_rl_reward.py \
    tests/test_planning_provider.py \
    tests/test_rl_policy_fallback.py \
    tests/test_policy_export.py \
    tests/test_route_progress.py \
    tests/test_reward_task_info.py \
    tests/test_phase3_sidecar_reward_logging.py \
    tests/test_phase3b_summary_aggregation.py \
    tests/test_phase3b_reward_calibration_report.py \
    tests/test_phase3b_acceptance_writer.py \
    tests/test_reward_sanity_report.py \
    tests/test_phase3_acceptance_writer.py \
    tests/test_live_backend_unavailable.py \
    tests/test_rl_carla_env.py \
    tests/test_zero_residual_equivalence.py \
    tests/test_phase2_acceptance.py \
    tests/test_profile_metric_source.py
else
  "${PYTHON_BIN}" - <<'PY'
import importlib
import inspect
import tempfile
from pathlib import Path

modules = (
    "tests.test_residual_action_projector",
    "tests.test_carla_rl_env_fake_backend",
    "tests.test_phase4_fake_backend_env",
    "tests.test_phase4_rollout_logging",
    "tests.test_phase4_live_backend_contract",
    "tests.test_phase4_normalizer_export",
    "tests.test_phase4_policy_export_adapter",
    "tests.test_phase4_training_cli_fake",
    "tests.test_phase4_acceptance_report",
    "tests.test_phase4_eval_acceptance",
    "tests.test_phase4b_policy_eval_acceptance",
    "tests.test_phase4b_split_reference_acceptance",
    "tests.test_phase4d_infraction_parser",
    "tests.test_phase4d_train_acceptance",
    "tests.test_phase4d_train_manifest",
    "tests.test_phase4d_train_only_runner",
    "tests.test_phase4d_eval_artifact",
    "tests.test_phase4d_eval_s4_runner",
    "tests.test_phase4d_eval_reference",
    "tests.test_phase4d_eval_s8_runner",
    "tests.test_phase4d_compare_only",
    "tests.test_phase4d_compare_wrapper",
    "tests.test_phase4d_policy_selection",
    "tests.test_phase4d_stageA_repeat",
    "tests.test_phase4d_stageA_runner",
    "tests.test_phase4d_stageA_aggregate",
    "tests.test_phase4d_authority_sensitivity_plan",
    "tests.test_phase4d_runner_scripts",
    "tests.test_phase4b_policy_eval_runner",
    "tests.test_phase4_route_health_analyzer",
    "tests.test_phase4c_horizon_sweep_summary",
    "tests.test_phase4c_horizon_sweep_runner",
    "tests.test_phase4_runner_help",
    "tests.test_rl_residual_controller",
    "tests.test_rl_observations",
    "tests.test_rl_reward",
    "tests.test_planning_provider",
    "tests.test_rl_policy_fallback",
    "tests.test_policy_export",
    "tests.test_route_progress",
    "tests.test_reward_task_info",
    "tests.test_phase3_sidecar_reward_logging",
    "tests.test_phase3b_summary_aggregation",
    "tests.test_phase3b_reward_calibration_report",
    "tests.test_phase3b_acceptance_writer",
    "tests.test_reward_sanity_report",
    "tests.test_phase3_acceptance_writer",
    "tests.test_live_backend_unavailable",
    "tests.test_rl_carla_env",
    "tests.test_zero_residual_equivalence",
    "tests.test_phase2_acceptance",
    "tests.test_profile_metric_source",
)

count = 0
for module_name in modules:
    module = importlib.import_module(module_name)
    for name, function in sorted(vars(module).items()):
        if not name.startswith("test_") or not callable(function):
            continue
        kwargs = {}
        signature = inspect.signature(function)
        with tempfile.TemporaryDirectory() as temp_dir:
            if "tmp_path" in signature.parameters:
                kwargs["tmp_path"] = Path(temp_dir)
            function(**kwargs)
        count += 1
print("pytest is not installed; ran %d pytest-style tests with stdlib fallback" % count)
PY
fi

"${PYTHON_BIN}" -m suspension_control.rl.evaluate_policy \
  --dry-run \
  --steps 100 \
  --output-dir /tmp/rl_suspension_dry_run
