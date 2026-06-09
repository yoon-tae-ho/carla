import csv
import json
import subprocess
import sys

from suspension_control.rl.phase4d_eval_reference import (
    write_phase4d_eval_s8_acceptance,
)


def write_suite_summary(path, *, updates=None):
    row = {
        "seed": "100",
        "scenario": "S8_rl_zero_residual_skyhook",
        "label": "S8 zero-residual RL over skyhook",
        "controller": "rl_zero_residual_skyhook",
        "status": "ok",
        "return_code": "0",
        "entry_status": "Finished",
        "global_status": "Perfect",
        "score_route": "100",
        "score_composed": "100",
        "score_penalty": "0",
        "route_completion_proxy": "1.0",
        "sidecar_command_verifies": "10",
        "rl_policy_available_ratio": "1.0",
        "fallback_ratio": "0",
        "low_speed_mask_ratio": "0",
        "rl_mean_abs_action": "0",
        "rl_mean_abs_residual_damper": "0",
        "effective_control_ratio": "0",
        "hard_safety_gate_ratio": "0",
        "soft_safety_gain_ratio": "0",
        "collision_count": "0",
        "lane_invasion_count": "0",
        "red_light_count": "0",
        "route_timeout_count": "0",
        "blocked_vehicle_count": "0",
        "sidecar_fatal_errors": "0",
        "sidecar_runtime_errors": "0",
        "result_json": "/workspace/out/seed100_S8_rl_zero_residual_skyhook/tfpp/result.json",
        "profile_csv": "/workspace/out/seed100_S8_rl_zero_residual_skyhook/profile.csv",
        "controller_diagnostics_csv": "/workspace/out/seed100_S8_rl_zero_residual_skyhook/controller_diagnostics.csv",
        "metrics_by_episode_csv": "/workspace/out/seed100_S8_rl_zero_residual_skyhook/metrics_by_episode.csv",
        "sidecar_events_csv": "/workspace/out/seed100_S8_rl_zero_residual_skyhook/sidecar_events.csv",
        "route_stdout_log": "/workspace/out/seed100_S8_rl_zero_residual_skyhook/route_stdout.log",
    }
    row.update(updates or {})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    return row


def test_phase4d_eval_s8_acceptance_passes_clean_zero_reference(tmp_path):
    output_dir = tmp_path / "s8"
    suite = output_dir / "suite_summary.csv"
    write_suite_summary(suite)

    _csv, _json, _manifest_json, row, manifest = write_phase4d_eval_s8_acceptance(
        output_dir=str(output_dir),
        suite_summary_path=str(suite),
        routes_subset="00",
        seed="100",
        route_return_code=0)

    assert row["phase4d_eval_s8_status"] == "pass"
    assert row["reference_status"] == "accepted_reference"
    assert row["failed_checks"] == ""
    assert row["workflow_role"] == "eval_s8_reference"
    assert row["scenario"] == "S8_rl_zero_residual_skyhook"
    assert row["controller"] == "rl_zero_residual_skyhook"
    assert row["baseline"] == "skyhook"
    assert row["planning_provider"] == "empty"
    assert row["mean_abs_action"] == 0.0
    assert row["mean_abs_residual_damper"] == 0.0
    assert manifest["reference_status"] == "accepted_reference"
    assert manifest["summary_path"].endswith(
        "seed100_S8_rl_zero_residual_skyhook/summary.json")
    assert (output_dir / "phase4d_eval_s8_acceptance.json").is_file()
    assert (output_dir / "phase4d_eval_s8_manifest.json").is_file()


def test_phase4d_eval_s8_acceptance_rejects_nonzero_action(tmp_path):
    output_dir = tmp_path / "s8"
    suite = output_dir / "suite_summary.csv"
    write_suite_summary(suite, updates={"rl_mean_abs_action": "0.001"})

    _csv, _json, _manifest_json, row, manifest = write_phase4d_eval_s8_acceptance(
        output_dir=str(output_dir),
        suite_summary_path=str(suite),
        routes_subset="00",
        seed="100",
        route_return_code=0)

    assert row["phase4d_eval_s8_status"] == "fail"
    assert row["reference_status"] == "invalid_reference"
    assert "mean_abs_action" in row["failed_checks"]
    assert manifest["reference_status"] == "invalid_reference"


def test_phase4d_eval_s8_acceptance_warns_for_runtime_and_fallback(tmp_path):
    output_dir = tmp_path / "s8"
    suite = output_dir / "suite_summary.csv"
    write_suite_summary(
        suite,
        updates={
            "sidecar_runtime_errors": "1",
            "fallback_ratio": "0.1",
            "low_speed_mask_ratio": "0.1",
            "soft_safety_gain_ratio": "0.1",
        })

    _csv, _json, _manifest_json, row, manifest = write_phase4d_eval_s8_acceptance(
        output_dir=str(output_dir),
        suite_summary_path=str(suite),
        routes_subset="00",
        seed="100",
        route_return_code=0)

    assert row["phase4d_eval_s8_status"] == "pass"
    assert row["failed_checks"] == ""
    assert "sidecar_runtime_errors" in row["warnings"]
    assert "fallback_ratio" in row["warnings"]
    assert "low_speed_mask_ratio" in row["warnings"]
    assert "soft_safety_gain_ratio" in row["warnings"]
    assert manifest["reference_status"] == "accepted_reference"


def test_phase4d_eval_s8_acceptance_rejects_infraction_and_route_failure(tmp_path):
    output_dir = tmp_path / "s8"
    suite = output_dir / "suite_summary.csv"
    write_suite_summary(
        suite,
        updates={
            "status": "route_command_failed",
            "return_code": "1",
            "collision_count": "1",
            "sidecar_fatal_errors": "1",
        })

    _csv, _json, _manifest_json, row, _manifest = write_phase4d_eval_s8_acceptance(
        output_dir=str(output_dir),
        suite_summary_path=str(suite),
        routes_subset="00",
        seed="100",
        route_return_code=1)

    assert row["phase4d_eval_s8_status"] == "fail"
    assert "route_return_code" in row["failed_checks"]
    assert "scenario_status" in row["failed_checks"]
    assert "scenario_return_code" in row["failed_checks"]
    assert "collision_count" in row["failed_checks"]
    assert "sidecar_fatal_errors" in row["failed_checks"]


def test_phase4d_eval_reference_cli_writes_outputs(tmp_path):
    output_dir = tmp_path / "s8"
    suite = output_dir / "suite_summary.csv"
    write_suite_summary(suite)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "suspension_control.rl.phase4d_eval_reference",
            "--output-dir",
            str(output_dir),
            "--suite-summary",
            str(suite),
            "--routes-subset",
            "00",
            "--seed",
            "100",
            "--fail-on-reject",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)

    assert "phase4d eval s8 acceptance:" in result.stdout
    assert (output_dir / "phase4d_eval_s8_acceptance.csv").is_file()
    assert (output_dir / "phase4d_eval_s8_acceptance.json").is_file()
    assert (output_dir / "phase4d_eval_s8_manifest.json").is_file()
