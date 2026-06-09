import csv
import json
import subprocess
import sys

from suspension_control.rl.phase4d_eval_artifact import (
    prepare_phase4d_eval_s4,
    write_phase4d_eval_s4_acceptance,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def make_train_artifact(tmp_path, *, accepted=True, summary_updates=None, manifest_updates=None):
    root = tmp_path / "phase4d_train"
    train_dir = root / "train"
    train_dir.mkdir(parents=True)
    policy_path = train_dir / "policy.ts"
    normalizer_path = train_dir / "normalizer.json"
    policy_path.write_bytes(b"policy")
    normalizer_path.write_text(json.dumps({
        "action_semantics": "normalized_damper_residual_v1",
        "feature_names": ["speed", "roll"],
    }) + "\n")
    summary = {
        "status": "trained",
        "backend": "live",
        "algorithm": "sac",
        "baseline": "skyhook",
        "planning_provider": "empty",
        "routes_subset": "00",
        "seed": 100,
        "total_timesteps": 4096,
        "action_semantics": "normalized_damper_residual_v1",
        "policy_path": str(policy_path),
        "normalizer_path": str(normalizer_path),
        "learn_error": "",
    }
    summary.update(summary_updates or {})
    write_json(train_dir / "training_summary.json", summary)
    acceptance = {
        "phase4d_train_status": "pass" if accepted else "fail",
        "eval_allowed": 1 if accepted else 0,
        "invalid_for_eval": 0 if accepted else 1,
        "failed_checks": "" if accepted else "training_status",
    }
    write_json(root / "phase4d_train_acceptance.json", acceptance)
    manifest = {
        "artifact_id": "phase4d-train-test",
        "artifact_status": "accepted_for_eval" if accepted else "invalid_for_eval",
        "train_dir": str(train_dir),
        "policy_path": str(policy_path),
        "normalizer_path": str(normalizer_path),
        "policy_sha256": "",
        "normalizer_sha256": "",
        "baseline": "skyhook",
        "planning_provider": "empty",
        "routes_subset": "00",
        "train_seed": 100,
        "action_semantics": "normalized_damper_residual_v1",
    }
    manifest.update(manifest_updates or {})
    write_json(root / "phase4d_train_manifest.json", manifest)
    return train_dir, root


def write_suite_summary(path, *, updates=None):
    row = {
        "seed": "100",
        "scenario": "S4_rl_residual_skyhook",
        "status": "ok",
        "return_code": "0",
        "entry_status": "Finished",
        "global_status": "Perfect",
        "score_route": "100",
        "score_composed": "100",
        "score_penalty": "0",
        "route_completion_proxy": "1.0",
        "sidecar_command_verifies": "12",
        "rl_policy_available_ratio": "1.0",
        "fallback_ratio": "0.2",
        "low_speed_mask_ratio": "0.2",
        "rl_fallback_rows": "3",
        "rl_fallback_reasons": "safety_gate_zero",
        "rl_mean_abs_action": "0.12",
        "rl_mean_abs_residual_damper": "0.009",
        "effective_control_ratio": "0.75",
        "hard_safety_gate_ratio": "0",
        "soft_safety_gain_ratio": "0.01",
        "collision_count": "0",
        "lane_invasion_count": "0",
        "red_light_count": "0",
        "route_timeout_count": "0",
        "blocked_vehicle_count": "0",
        "sidecar_fatal_errors": "0",
        "sidecar_runtime_errors": "0",
        "rl_diagnostic_rows": "100",
        "reward_rows": "100",
        "reward_row_ratio": "1.0",
    }
    row.update(updates or {})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    return row


def test_phase4d_eval_artifact_preflight_accepts_train_manifest(tmp_path):
    train_dir, _root = make_train_artifact(tmp_path)

    row, manifest = prepare_phase4d_eval_s4(
        train_dir=str(train_dir),
        output_dir=str(tmp_path / "eval_s4"))

    assert row["eval_status"] == "ready_for_eval"
    assert row["phase4d_eval_s4_status"] == "ready_for_eval"
    assert row["failed_checks"] == ""
    assert row["policy_path"].endswith("policy.ts")
    assert row["normalizer_path"].endswith("normalizer.json")
    assert row["routes_subset"] == "00"
    assert row["seed"] == "100"
    assert manifest["workflow_role"] == "eval_s4_only"
    assert manifest["source_train_artifact_id"] == "phase4d-train-test"


def test_phase4d_eval_artifact_preflight_rejects_invalid_train_without_carla(tmp_path):
    train_dir, _root = make_train_artifact(tmp_path, accepted=False)
    output_dir = tmp_path / "eval_s4"

    row, manifest = prepare_phase4d_eval_s4(
        train_dir=str(train_dir),
        output_dir=str(output_dir))

    assert row["eval_status"] == "skipped_due_to_invalid_train_artifact"
    assert row["phase4d_eval_s4_status"] == "fail"
    assert "eval_allowed" in row["failed_checks"]
    assert (output_dir / "phase4d_eval_s4_acceptance.json").is_file()
    assert manifest["artifact_status"] == "invalid_train_artifact"


def test_phase4d_eval_s4_acceptance_passes_nonzero_residual_and_records_fallback(tmp_path):
    train_dir, _root = make_train_artifact(tmp_path)
    output_dir = tmp_path / "eval_s4"
    suite = output_dir / "suite_summary.csv"
    write_suite_summary(suite)

    _csv, _json, _manifest_json, row, manifest = write_phase4d_eval_s4_acceptance(
        train_dir=str(train_dir),
        output_dir=str(output_dir),
        suite_summary_path=str(suite),
        route_return_code=0)

    assert row["phase4d_eval_s4_status"] == "pass"
    assert row["eval_status"] == "completed"
    assert row["failed_checks"] == ""
    assert row["fallback_ratio"] == "0.2"
    assert "fallback_ratio" in row["warnings"]
    assert "low_speed_mask_ratio" in row["warnings"]
    assert "soft_safety_gain_ratio" in row["warnings"]
    assert row["mean_abs_action"] == 0.12
    assert row["mean_abs_residual_damper"] == 0.009
    assert manifest["artifact_status"] == "accepted_eval"


def test_phase4d_eval_s4_acceptance_fails_hard_safety_but_not_fallback(tmp_path):
    train_dir, _root = make_train_artifact(tmp_path)
    output_dir = tmp_path / "eval_s4"
    suite = output_dir / "suite_summary.csv"
    write_suite_summary(suite, updates={"hard_safety_gate_ratio": "0.1"})

    _csv, _json, _manifest_json, row, _manifest = write_phase4d_eval_s4_acceptance(
        train_dir=str(train_dir),
        output_dir=str(output_dir),
        suite_summary_path=str(suite),
        route_return_code=0)

    assert row["phase4d_eval_s4_status"] == "fail"
    assert "hard_safety_gate_ratio" in row["failed_checks"]
    assert "fallback_ratio" not in row["failed_checks"]


def test_phase4d_eval_s4_acceptance_warns_for_minor_runtime_errors(tmp_path):
    train_dir, _root = make_train_artifact(tmp_path)
    output_dir = tmp_path / "eval_s4"
    suite = output_dir / "suite_summary.csv"
    sidecar_events = output_dir / "seed100_S4_rl_residual_skyhook" / "sidecar_events.csv"
    sidecar_events.parent.mkdir(parents=True, exist_ok=True)
    with open(sidecar_events, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["event", "message"])
        writer.writeheader()
        writer.writerow({
            "event": "runtime_error",
            "message": "wheel 0 damper scale mismatch: expected 1 got 0.96",
        })
        writer.writerow({"event": "command_verified", "message": "readback matched command"})
    write_suite_summary(
        suite,
        updates={
            "sidecar_command_verifies": "123",
            "sidecar_runtime_errors": "4",
            "sidecar_events_csv": str(sidecar_events),
        })

    _csv, _json, _manifest_json, row, manifest = write_phase4d_eval_s4_acceptance(
        train_dir=str(train_dir),
        output_dir=str(output_dir),
        suite_summary_path=str(suite),
        route_return_code=0)

    assert row["phase4d_eval_s4_status"] == "pass"
    assert row["failed_checks"] == ""
    assert "sidecar_runtime_errors" in row["warnings"]
    assert "minor_command_readback_mismatch" in row["warnings"]
    assert row["sidecar_runtime_error_ratio"] == 4.0 / 123.0
    assert row["sidecar_consecutive_runtime_errors"] == 1
    assert row["sidecar_final_verification_recovered"] == 1
    assert manifest["artifact_status"] == "accepted_eval"


def test_phase4d_eval_s4_acceptance_fails_excessive_runtime_errors(tmp_path):
    train_dir, _root = make_train_artifact(tmp_path)
    output_dir = tmp_path / "eval_s4"
    suite = output_dir / "suite_summary.csv"
    write_suite_summary(
        suite,
        updates={
            "sidecar_command_verifies": "20",
            "sidecar_runtime_errors": "2",
        })

    _csv, _json, _manifest_json, row, _manifest = write_phase4d_eval_s4_acceptance(
        train_dir=str(train_dir),
        output_dir=str(output_dir),
        suite_summary_path=str(suite),
        route_return_code=0)

    assert row["phase4d_eval_s4_status"] == "fail"
    assert "sidecar_runtime_error_ratio" in row["failed_checks"]
    assert "sidecar_runtime_errors" in row["warnings"]


def test_phase4d_eval_s4_acceptance_fails_global_status_not_perfect(tmp_path):
    train_dir, _root = make_train_artifact(tmp_path)
    output_dir = tmp_path / "eval_s4"
    suite = output_dir / "suite_summary.csv"
    write_suite_summary(suite, updates={"global_status": "SUCCESS"})

    _csv, _json, _manifest_json, row, _manifest = write_phase4d_eval_s4_acceptance(
        train_dir=str(train_dir),
        output_dir=str(output_dir),
        suite_summary_path=str(suite),
        route_return_code=0)

    assert row["phase4d_eval_s4_status"] == "fail"
    assert "global_status" in row["failed_checks"]


def test_phase4d_eval_artifact_cli_print_shell(tmp_path):
    train_dir, _root = make_train_artifact(tmp_path)
    output_dir = tmp_path / "eval_s4"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "suspension_control.rl.phase4d_eval_artifact",
            "--action",
            "prepare",
            "--train-dir",
            str(train_dir),
            "--output-dir",
            str(output_dir),
            "--print-shell",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)

    assert "PHASE4D_PREFLIGHT_OK=1" in result.stdout
    assert "PHASE4D_POLICY_PATH=" in result.stdout
    assert (output_dir / "phase4d_eval_s4_manifest.json").is_file()
