import hashlib
import csv
import json
import subprocess
import sys
from pathlib import Path

from suspension_control.rl.phase4d_action_authority_sweep import (
    write_action_authority_aggregate,
)
from suspension_control.rl.phase4d_scripted_residual_sensitivity import (
    write_scripted_residual_aggregate,
)
from suspension_control.rl.phase4d_authority_sensitivity_report import (
    write_authority_sensitivity_report,
)
from suspension_control.rl.phase4d_authority_sensitivity_common import (
    action_scale_label,
    parse_action_scales,
    parse_scripted_residuals,
    scripted_scenario_name,
    write_action_authority_plan,
    write_scripted_residual_plan,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def make_train_manifest(tmp_path, *, updates=None, policy_bytes=b"policy"):
    root = tmp_path / "phase4d_train"
    train_dir = root / "train"
    train_dir.mkdir(parents=True)
    policy_path = train_dir / "policy.ts"
    normalizer_path = train_dir / "normalizer.json"
    policy_path.write_bytes(policy_bytes)
    normalizer_path.write_text(json.dumps({
        "action_semantics": "normalized_damper_residual_v1",
        "feature_names": ["speed", "roll"],
    }) + "\n")
    manifest = {
        "artifact_id": "phase4d-train-test-artifact",
        "workflow_role": "train_only",
        "artifact_status": "accepted_for_eval",
        "eval_allowed": 1,
        "invalid_for_eval": 0,
        "baseline": "skyhook",
        "planning_provider": "empty",
        "routes_subset": "00",
        "train_seed": 100,
        "total_timesteps": 4096,
        "action_semantics": "normalized_damper_residual_v1",
        "train_dir": str(train_dir),
        "policy_path": str(policy_path),
        "normalizer_path": str(normalizer_path),
        "policy_sha256": _sha256(policy_path),
        "normalizer_sha256": _sha256(normalizer_path),
    }
    manifest.update(updates or {})
    manifest_path = root / "phase4d_train_manifest.json"
    write_json(manifest_path, manifest)
    return root, train_dir, manifest_path


def test_action_authority_plan_accepts_artifact_and_writes_scale_plan(tmp_path):
    _root, train_dir, manifest_path = make_train_manifest(tmp_path)
    output_dir = tmp_path / "authority"

    plan_path, acceptance_path, manifest_out_path, result = write_action_authority_plan(
        output_dir=str(output_dir),
        train_manifest_path=str(manifest_path),
        train_dir=str(train_dir),
        eval_seeds=["100", "101"],
        routes_subset="00",
        baseline="skyhook",
        planning_provider="empty",
        action_scales=[0.0, 1.0, 3.0, 5.0, 10.0],
        dry_run=True,
    )

    acceptance = result["acceptance"]
    plan = result["plan"]
    manifest = result["manifest"]
    assert acceptance["phase4d_action_authority_status"] == "ready_for_execution"
    assert acceptance["preflight_ok"] == 1
    assert acceptance["action_scales"] == "0.0,1.0,3.0,5.0,10.0"
    assert "S4_gain_10p0_rl_residual_skyhook" in acceptance["scenario_names"]
    assert [row["eval_seed"] for row in plan["seed_plans"]] == ["100", "101"]
    assert plan["seed_plans"][0]["action_scale_plans"][2]["output_dir"].endswith("eval_seed_100/gain_3p0")
    assert plan["seed_plans"][0]["action_scale_plans"][4]["scenario"] == "S4_gain_10p0_rl_residual_skyhook"
    assert manifest["workflow_role"] == "phase4d_action_authority_sweep"
    assert plan_path.endswith("phase4d_action_authority_plan.json")
    assert acceptance_path.endswith("phase4d_action_authority_acceptance.json")
    assert manifest_out_path.endswith("phase4d_action_authority_manifest.json")
    assert "transfuser_suspension_control_suite.py" not in (output_dir / "phase4d_action_authority_plan.json").read_text()


def test_action_authority_plan_rejects_missing_policy_without_carla(tmp_path):
    _root, train_dir, manifest_path = make_train_manifest(tmp_path)
    (train_dir / "policy.ts").unlink()

    _plan_path, _acceptance_path, _manifest_path, result = write_action_authority_plan(
        output_dir=str(tmp_path / "authority"),
        train_manifest_path=str(manifest_path),
        train_dir=str(train_dir),
        eval_seeds=["100"],
        routes_subset="00",
        baseline="skyhook",
        planning_provider="empty",
        action_scales=[0.0],
        dry_run=True,
    )

    assert result["acceptance"]["phase4d_action_authority_status"] == "preflight_failed"
    assert "policy_path" in result["acceptance"]["failed_checks"]


def test_action_authority_plan_rejects_checksum_mismatch(tmp_path):
    _root, train_dir, manifest_path = make_train_manifest(
        tmp_path,
        updates={"normalizer_sha256": "not-the-real-hash"})

    _plan_path, _acceptance_path, _manifest_path, result = write_action_authority_plan(
        output_dir=str(tmp_path / "authority"),
        train_manifest_path=str(manifest_path),
        train_dir=str(train_dir),
        eval_seeds=["100"],
        routes_subset="00",
        baseline="skyhook",
        planning_provider="empty",
        action_scales=[1.0],
        dry_run=True,
    )

    assert result["acceptance"]["phase4d_action_authority_status"] == "preflight_failed"
    assert "normalizer_sha256" in result["acceptance"]["failed_checks"]


def test_action_scale_parser_and_labels():
    assert parse_action_scales("0.0,1.0,3.0,5.0,10.0") == [0.0, 1.0, 3.0, 5.0, 10.0]
    assert action_scale_label(0.0) == "0p0"
    assert action_scale_label(5.0) == "5p0"
    assert action_scale_label(10.0) == "10p0"
    try:
        parse_action_scales("1.0,nope")
    except ValueError as exc:
        assert "invalid action scale" in str(exc)
    else:
        raise AssertionError("invalid action scale should fail")


def test_scripted_residual_plan_and_canonical_names(tmp_path):
    output_dir = tmp_path / "scripted"

    plan_path, acceptance_path, manifest_out_path, result = write_scripted_residual_plan(
        output_dir=str(output_dir),
        eval_seeds=["100"],
        routes_subset="00",
        baseline="skyhook",
        planning_provider="empty",
        scripted_residuals=["zero", "+0.05", "-0.05", "+0.10", "-0.10"],
        dry_run=True,
    )

    acceptance = result["acceptance"]
    plan = result["plan"]
    assert acceptance["phase4d_scripted_residual_status"] == "ready_for_execution"
    assert acceptance["scripted_residuals"] == "zero,const_p0p05,const_m0p05,const_p0p10,const_m0p10"
    assert "SC_const_p0p10_residual_skyhook" in acceptance["scenario_names"]
    assert plan["seed_plans"][0]["residual_plans"][1]["scripted_residual_value"] == 0.05
    assert plan["seed_plans"][0]["residual_plans"][4]["output_dir"].endswith("eval_seed_100/const_m0p10")
    assert plan_path.endswith("phase4d_scripted_residual_plan.json")
    assert acceptance_path.endswith("phase4d_scripted_residual_acceptance.json")
    assert manifest_out_path.endswith("phase4d_scripted_residual_manifest.json")
    assert parse_scripted_residuals("zero,const_p0p05,const_m0p05") == [
        "zero",
        "const_p0p05",
        "const_m0p05",
    ]
    assert scripted_scenario_name("const_m0p10") == "SC_const_m0p10_residual_skyhook"


def test_action_authority_cli_dry_run_writes_plan(tmp_path):
    _root, train_dir, manifest_path = make_train_manifest(tmp_path)
    output_dir = tmp_path / "authority_cli"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "suspension_control.rl.phase4d_action_authority_sweep",
            "--action",
            "plan",
            "--train-manifest",
            str(manifest_path),
            "--train-dir",
            str(train_dir),
            "--eval-seeds",
            "100",
            "--action-scales",
            "0.0,1.0,10.0",
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)

    assert "phase4d action authority plan:" in result.stdout
    assert (output_dir / "phase4d_action_authority_plan.json").is_file()
    assert "transfuser_suspension_control_suite.py" not in result.stdout


def test_scripted_residual_cli_dry_run_writes_plan(tmp_path):
    output_dir = tmp_path / "scripted_cli"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "suspension_control.rl.phase4d_scripted_residual_sensitivity",
            "--action",
            "plan",
            "--eval-seeds",
            "100",
            "--scripted-residuals",
            "zero,const_p0p05,const_m0p05",
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)

    assert "phase4d scripted residual plan:" in result.stdout
    assert (output_dir / "phase4d_scripted_residual_plan.json").is_file()
    assert "transfuser_suspension_control_suite.py" not in result.stdout


def test_action_authority_aggregate_pass_with_warnings_and_stageA_effect(tmp_path):
    _root, train_dir, manifest_path = make_train_manifest(tmp_path)
    output_dir = tmp_path / "authority_aggregate"
    stageA_report = tmp_path / "stageA_report.json"
    write_json(stageA_report, {
        "workflow_role": "stageA_same_artifact_eval_repeat",
        "metrics": {
            "delta_rms_vertical_acc": {"std": 0.01, "count": 3},
            "delta_comfort_score": {"std": 0.01, "count": 3},
        },
    })
    write_action_authority_plan(
        output_dir=str(output_dir),
        train_manifest_path=str(manifest_path),
        train_dir=str(train_dir),
        eval_seeds=["100"],
        routes_subset="00",
        baseline="skyhook",
        planning_provider="empty",
        action_scales=[0.0, 1.0, 3.0, 5.0, 10.0],
        stageA_report_path=str(stageA_report),
        dry_run=False,
    )
    _write_s8_reference(output_dir, seed="100")
    residual_by_scale = {
        0.0: 0.0,
        1.0: 0.02,
        3.0: 0.04,
        5.0: 0.06,
        10.0: 0.08,
    }
    for scale, residual in residual_by_scale.items():
        _write_gain_output(
            output_dir,
            seed="100",
            scale=scale,
            final_residual=residual,
            residual_scale_clip_ratio=0.25 if scale == 10.0 else 0.0,
            delta_rms_vertical_acc=-0.1 if scale == 5.0 else -0.01 * scale,
        )

    _summary_csv, _summary_json, _acceptance_csv, acceptance_json, report_json, report = (
        write_action_authority_aggregate(output_dir=str(output_dir)))

    acceptance = _read_json(acceptance_json)
    report_doc = _read_json(report_json)
    assert report["phase4d_action_authority_status"] == "AUTHORITY_SWEEP_PASS_WITH_WARNINGS"
    assert acceptance["aggregate_ok"] == 1
    assert acceptance["authority_monotonicity_ok"] == 1
    assert acceptance["metric_effect_above_noise"] == 1
    assert acceptance["clamp_limited_gain"] == 1
    assert "clamp_or_saturation_limited" in acceptance["warnings"]
    assert report_doc["diagnosis"]["AUTHORITY_DIAGNOSIS"] == "CLAMP_LIMITED_AUTHORITY"

    rows = _csv_rows(output_dir / "phase4d_action_authority_summary.csv")
    zero = [row for row in rows if row["action_scale"] == "0"][0]
    gain10 = [row for row in rows if row["action_scale"] == "10"][0]
    assert abs(float(zero["final_mean_abs_residual_damper"])) <= 1.0e-12
    assert gain10["residual_scale_clip_ratio"] == "0.25"
    assert gain10["authority_row_status"] == "pass_with_warnings"


def test_action_authority_aggregate_incomplete_when_outputs_missing(tmp_path):
    _root, train_dir, manifest_path = make_train_manifest(tmp_path)
    output_dir = tmp_path / "authority_missing"
    write_action_authority_plan(
        output_dir=str(output_dir),
        train_manifest_path=str(manifest_path),
        train_dir=str(train_dir),
        eval_seeds=["100"],
        routes_subset="00",
        baseline="skyhook",
        planning_provider="empty",
        action_scales=[1.0],
        dry_run=False,
    )

    _summary_csv, _summary_json, _acceptance_csv, acceptance_json, _report_json, report = (
        write_action_authority_aggregate(output_dir=str(output_dir)))

    acceptance = _read_json(acceptance_json)
    assert report["phase4d_action_authority_status"] == "AUTHORITY_SWEEP_FAIL_SAFETY"
    assert acceptance["aggregate_ok"] == 0
    assert "s8_reference_safety" in acceptance["failed_checks"]


def test_action_authority_in_container_runner_dry_run_writes_plan(tmp_path):
    _root, train_dir, manifest_path = make_train_manifest(tmp_path)
    output_dir = tmp_path / "authority_runner"

    result = subprocess.run(
        [
            "bash",
            str(_taeho_root() / "scripts" / "run_phase4d_action_authority_sweep.sh"),
            "--dry-run",
            "--train-manifest",
            str(manifest_path),
            "--train-dir",
            str(train_dir),
            "--eval-seeds",
            "100",
            "--routes-subset",
            "00",
            "--action-scales",
            "0.0,1.0",
            "--output-dir",
            str(output_dir),
        ],
        cwd=str(_taeho_root()),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)

    assert "Phase 4-D action authority dry-run complete." in result.stdout
    assert "transfuser_suspension_control_suite.py" not in result.stdout
    assert (output_dir / "phase4d_action_authority_plan.json").is_file()
    plan = _read_json(output_dir / "phase4d_action_authority_plan.json")
    assert plan["seed_plans"][0]["action_scale_plans"][0]["output_dir"].endswith(
        "eval_seed_100/gain_0p0")


def test_action_authority_in_container_runner_print_command_lists_gain_configs(tmp_path):
    result = subprocess.run(
        [
            "bash",
            str(_taeho_root() / "scripts" / "run_phase4d_action_authority_sweep.sh"),
            "--print-command",
            "--train-manifest",
            str(tmp_path / "train" / "phase4d_train_manifest.json"),
            "--train-dir",
            str(tmp_path / "train" / "train"),
            "--eval-seeds",
            "100",
            "--routes-subset",
            "00",
            "--action-scales",
            "0.0,1.0,10.0",
            "--output-dir",
            str(tmp_path / "authority"),
        ],
        cwd=str(_taeho_root()),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)

    stdout = result.stdout
    assert "AUTHORITY_PLAN:" in stdout
    assert "AUTHORITY_S8_SEED_100:" in stdout
    assert "AUTHORITY_GAIN_CONFIG_SEED_100_10p0" in stdout
    assert "AUTHORITY_S4_SEED_100_GAIN_1p0:" in stdout
    assert "--rl-residual-config" in stdout
    assert "AUTHORITY_COMPARE_SEED_100_GAIN_10p0:" in stdout
    assert "AUTHORITY_AGGREGATE:" in stdout
    assert "train_real_carla_sac" not in stdout


def test_action_authority_docker_wrapper_print_command_translates_paths():
    sim_root = _sim_root()
    train_root = sim_root / "e2e_models" / "outputs" / "phase4d_train"
    output_root = sim_root / "e2e_models" / "outputs" / "phase4d_authority"

    result = subprocess.run(
        [
            "bash",
            str(sim_root / "docker" / "run_phase4d_action_authority_sweep.sh"),
            "--print-command",
            "--train-manifest",
            str(train_root / "phase4d_train_manifest.json"),
            "--train-dir",
            str(train_root / "train"),
            "--stageA-report",
            str(sim_root / "e2e_models" / "outputs" / "stageA" / "phase4d_stageA_stochasticity_report.json"),
            "--output-dir",
            str(output_root),
        ],
        cwd=str(sim_root / "docker"),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)

    stdout = result.stdout
    assert "docker compose --env-file .env -f compose.dev.yaml run --rm runner" in stdout
    assert "bash scripts/run_phase4d_action_authority_sweep.sh" in stdout
    assert "/workspace/e2e_models/outputs/phase4d_train/phase4d_train_manifest.json" in stdout
    assert "/workspace/e2e_models/outputs/phase4d_train/train" in stdout
    assert "/workspace/e2e_models/outputs/phase4d_authority" in stdout
    assert str(sim_root) not in stdout


def test_scripted_residual_aggregate_pass_with_warnings_and_sign_response(tmp_path):
    output_dir = tmp_path / "scripted_aggregate"
    stageA_report = tmp_path / "stageA_report.json"
    write_json(stageA_report, {
        "workflow_role": "stageA_same_artifact_eval_repeat",
        "metrics": {
            "delta_rms_vertical_acc": {"std": 0.01, "count": 3},
            "delta_comfort_score": {"std": 0.01, "count": 3},
        },
    })
    write_scripted_residual_plan(
        output_dir=str(output_dir),
        eval_seeds=["100"],
        routes_subset="00",
        baseline="skyhook",
        planning_provider="empty",
        scripted_residuals=["zero", "const_p0p05", "const_m0p05", "const_p0p10", "const_m0p10"],
        stageA_report_path=str(stageA_report),
        dry_run=False,
    )
    _write_scripted_output(output_dir, seed="100", residual="zero", residual_value=0.0, rms_vertical_acc=1.0)
    _write_scripted_output(output_dir, seed="100", residual="const_p0p05", residual_value=0.05, rms_vertical_acc=0.9)
    _write_scripted_output(output_dir, seed="100", residual="const_m0p05", residual_value=-0.05, rms_vertical_acc=1.1)
    _write_scripted_output(
        output_dir,
        seed="100",
        residual="const_p0p10",
        residual_value=0.10,
        rms_vertical_acc=0.75,
        residual_scale_clip_ratio=0.2)
    _write_scripted_output(output_dir, seed="100", residual="const_m0p10", residual_value=-0.10, rms_vertical_acc=1.25)

    _summary_csv, _summary_json, _acceptance_csv, acceptance_json, report_json, report = (
        write_scripted_residual_aggregate(output_dir=str(output_dir)))

    acceptance = _read_json(acceptance_json)
    report_doc = _read_json(report_json)
    assert report["phase4d_scripted_residual_status"] == "SCRIPTED_SENSITIVITY_PASS_WITH_WARNINGS"
    assert acceptance["aggregate_ok"] == 1
    assert acceptance["scripted_effect_above_noise"] == 1
    assert acceptance["signed_residual_direction_consistency"] == 1
    assert acceptance["magnitude_response_consistency"] == 1
    assert report_doc["diagnosis"]["SENSITIVITY_DIAGNOSIS"] == "SCRIPTED_SENSITIVITY_DETECTED"

    rows = _csv_rows(output_dir / "phase4d_scripted_residual_summary.csv")
    plus = [row for row in rows if row["scripted_residual"] == "const_p0p05"][0]
    minus = [row for row in rows if row["scripted_residual"] == "const_m0p05"][0]
    large = [row for row in rows if row["scripted_residual"] == "const_p0p10"][0]
    assert abs(float(plus["delta_rms_vertical_acc_vs_zero"]) + 0.1) <= 1.0e-9
    assert abs(float(minus["delta_rms_vertical_acc_vs_zero"]) - 0.1) <= 1.0e-9
    assert large["scripted_row_status"] == "pass_with_warnings"
    assert (output_dir / "eval_seed_100" / "compare_const_p0p05" / "phase4d_scripted_residual_compare.json").is_file()


def test_scripted_residual_aggregate_incomplete_when_outputs_missing(tmp_path):
    output_dir = tmp_path / "scripted_missing"
    write_scripted_residual_plan(
        output_dir=str(output_dir),
        eval_seeds=["100"],
        routes_subset="00",
        baseline="skyhook",
        planning_provider="empty",
        scripted_residuals=["zero", "const_p0p05"],
        dry_run=False,
    )

    _summary_csv, _summary_json, _acceptance_csv, acceptance_json, _report_json, report = (
        write_scripted_residual_aggregate(output_dir=str(output_dir)))

    acceptance = _read_json(acceptance_json)
    assert report["phase4d_scripted_residual_status"] == "SCRIPTED_SENSITIVITY_INCOMPLETE"
    assert acceptance["aggregate_ok"] == 0
    assert "incomplete_outputs" in acceptance["failed_checks"]


def test_scripted_residual_in_container_runner_dry_run_writes_plan(tmp_path):
    output_dir = tmp_path / "scripted_runner"

    result = subprocess.run(
        [
            "bash",
            str(_taeho_root() / "scripts" / "run_phase4d_scripted_residual_sensitivity.sh"),
            "--dry-run",
            "--eval-seeds",
            "100",
            "--routes-subset",
            "00",
            "--scripted-residuals",
            "zero,const_p0p05,const_m0p05",
            "--output-dir",
            str(output_dir),
        ],
        cwd=str(_taeho_root()),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)

    assert "Phase 4-D scripted residual dry-run complete." in result.stdout
    assert "transfuser_suspension_control_suite.py" not in result.stdout
    assert (output_dir / "phase4d_scripted_residual_plan.json").is_file()
    plan = _read_json(output_dir / "phase4d_scripted_residual_plan.json")
    assert plan["seed_plans"][0]["residual_plans"][1]["output_dir"].endswith(
        "eval_seed_100/const_p0p05")


def test_scripted_residual_in_container_runner_print_command_lists_configs(tmp_path):
    result = subprocess.run(
        [
            "bash",
            str(_taeho_root() / "scripts" / "run_phase4d_scripted_residual_sensitivity.sh"),
            "--print-command",
            "--eval-seeds",
            "100",
            "--routes-subset",
            "00",
            "--scripted-residuals",
            "zero,const_p0p05,const_m0p05",
            "--output-dir",
            str(tmp_path / "scripted"),
        ],
        cwd=str(_taeho_root()),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)

    stdout = result.stdout
    assert "SCRIPTED_PLAN:" in stdout
    assert "SCRIPTED_CONFIG_SEED_100_const_p0p05" in stdout
    assert "scripted_residual_value=0.05" in stdout
    assert "SCRIPTED_EVAL_SEED_100_const_m0p05:" in stdout
    assert "--rl-residual-config" in stdout
    assert "SCRIPTED_AGGREGATE:" in stdout
    assert "train_real_carla_sac" not in stdout


def test_scripted_residual_docker_wrapper_print_command_translates_paths():
    sim_root = _sim_root()
    output_root = sim_root / "e2e_models" / "outputs" / "phase4d_scripted"
    stageA_report = sim_root / "e2e_models" / "outputs" / "stageA" / "phase4d_stageA_stochasticity_report.json"

    result = subprocess.run(
        [
            "bash",
            str(sim_root / "docker" / "run_phase4d_scripted_residual_sensitivity.sh"),
            "--print-command",
            "--eval-seeds",
            "100",
            "--routes-subset",
            "00",
            "--stageA-report",
            str(stageA_report),
            "--output-dir",
            str(output_root),
        ],
        cwd=str(sim_root / "docker"),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)

    stdout = result.stdout
    assert "docker compose --env-file .env -f compose.dev.yaml run --rm runner" in stdout
    assert "bash scripts/run_phase4d_scripted_residual_sensitivity.sh" in stdout
    assert "/workspace/e2e_models/outputs/stageA/phase4d_stageA_stochasticity_report.json" in stdout
    assert "/workspace/e2e_models/outputs/phase4d_scripted" in stdout
    assert str(sim_root) not in stdout


def test_combined_report_reward_too_conservative_from_synthetic_reports(tmp_path):
    authority_dir = tmp_path / "authority"
    scripted_dir = tmp_path / "scripted"
    output_dir = tmp_path / "combined"
    write_json(authority_dir / "phase4d_action_authority_report.json", {
        "phase4d_action_authority_status": "AUTHORITY_SWEEP_PASS_WITH_WARNINGS",
        "aggregate_ok": 1,
        "diagnosis": {
            "AUTHORITY_DIAGNOSIS": "AUTHORITY_EFFECT_WITHIN_NOISE",
            "authority_monotonicity_ok": 1,
            "metric_effect_above_noise": 0,
            "safety_limited_gain": 0,
            "clamp_limited_gain": 0,
        },
        "rows": [
            {"action_scale": 0.0, "final_mean_abs_residual_damper": 0.0},
            {"action_scale": 1.0, "final_mean_abs_residual_damper": 0.01},
            {"action_scale": 10.0, "final_mean_abs_residual_damper": 0.08},
        ],
    })
    write_json(authority_dir / "phase4d_action_authority_acceptance.json", {
        "phase4d_action_authority_status": "AUTHORITY_SWEEP_PASS_WITH_WARNINGS",
        "aggregate_ok": 1,
    })
    write_json(scripted_dir / "phase4d_scripted_residual_report.json", {
        "phase4d_scripted_residual_status": "SCRIPTED_SENSITIVITY_PASS",
        "aggregate_ok": 1,
        "diagnosis": {
            "SENSITIVITY_DIAGNOSIS": "SCRIPTED_SENSITIVITY_DETECTED",
            "SAFETY_AUTHORITY_STATUS": "safe",
            "scripted_effect_above_noise": 1,
            "signed_residual_direction_consistency": 1,
            "magnitude_response_consistency": 1,
        },
        "rows": [
            {"scripted_residual": "const_p0p10", "final_mean_abs_residual_damper": 0.10},
        ],
    })
    write_json(scripted_dir / "phase4d_scripted_residual_acceptance.json", {
        "phase4d_scripted_residual_status": "SCRIPTED_SENSITIVITY_PASS",
        "aggregate_ok": 1,
    })

    report_csv, report_json, report = write_authority_sensitivity_report(
        authority_dir=str(authority_dir),
        scripted_dir=str(scripted_dir),
        output_dir=str(output_dir),
    )

    assert report["diagnosis"] == "DIAGNOSIS_REWARD_TOO_CONSERVATIVE"
    assert report["acceptance"]["diagnosis_ok"] == 1
    assert report["acceptance"]["authority_metric_effect_above_noise"] == 0
    assert report["acceptance"]["scripted_effect_above_noise"] == 1
    assert Path(report_json).is_file()
    rows = _csv_rows(Path(report_csv))
    assert rows[0]["diagnosis"] == "DIAGNOSIS_REWARD_TOO_CONSERVATIVE"


def test_combined_report_incomplete_when_scripted_report_missing(tmp_path):
    authority_dir = tmp_path / "authority"
    scripted_dir = tmp_path / "scripted"
    write_json(authority_dir / "phase4d_action_authority_report.json", {
        "phase4d_action_authority_status": "AUTHORITY_SWEEP_PASS",
        "aggregate_ok": 1,
        "diagnosis": {
            "AUTHORITY_DIAGNOSIS": "AUTHORITY_EFFECT_DETECTED",
            "authority_monotonicity_ok": 1,
            "metric_effect_above_noise": 1,
        },
        "rows": [{"final_mean_abs_residual_damper": 0.05}],
    })

    _report_csv, _report_json, report = write_authority_sensitivity_report(
        authority_dir=str(authority_dir),
        scripted_dir=str(scripted_dir),
        output_dir=str(tmp_path / "combined"),
    )

    assert report["diagnosis"] == "DIAGNOSIS_INCOMPLETE"
    assert "scripted_report_missing" in report["failed_checks"]
    assert report["acceptance"]["diagnosis_ok"] == 0


def test_combined_in_container_wrapper_print_command_lists_split_outputs(tmp_path):
    result = subprocess.run(
        [
            "bash",
            str(_taeho_root() / "scripts" / "run_phase4d_authority_sensitivity_diagnostics.sh"),
            "--print-command",
            "--train-manifest",
            str(tmp_path / "train" / "phase4d_train_manifest.json"),
            "--train-dir",
            str(tmp_path / "train" / "train"),
            "--eval-seeds",
            "100",
            "--routes-subset",
            "00",
            "--output-dir",
            str(tmp_path / "diagnostics"),
        ],
        cwd=str(_taeho_root()),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)

    stdout = result.stdout
    assert "COMBINED_AUTHORITY_SWEEP:" in stdout
    assert "COMBINED_SCRIPTED_RESIDUAL:" in stdout
    assert "COMBINED_REPORT:" in stdout
    assert "action_authority" in stdout
    assert "scripted_residual" in stdout
    assert "combined_report" in stdout
    assert "train_real_carla_sac" not in stdout


def test_combined_docker_wrapper_print_command_translates_paths():
    sim_root = _sim_root()
    train_root = sim_root / "e2e_models" / "outputs" / "phase4d_train"
    output_root = sim_root / "e2e_models" / "outputs" / "phase4d_authority_sensitivity"

    result = subprocess.run(
        [
            "bash",
            str(sim_root / "docker" / "run_phase4d_authority_sensitivity_diagnostics.sh"),
            "--print-command",
            "--train-manifest",
            str(train_root / "phase4d_train_manifest.json"),
            "--train-dir",
            str(train_root / "train"),
            "--stageA-report",
            str(sim_root / "e2e_models" / "outputs" / "stageA" / "phase4d_stageA_stochasticity_report.json"),
            "--output-dir",
            str(output_root),
        ],
        cwd=str(sim_root / "docker"),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)

    stdout = result.stdout
    assert "docker compose --env-file .env -f compose.dev.yaml run --rm runner" in stdout
    assert "bash scripts/run_phase4d_authority_sensitivity_diagnostics.sh" in stdout
    assert "/workspace/e2e_models/outputs/phase4d_train/phase4d_train_manifest.json" in stdout
    assert "/workspace/e2e_models/outputs/phase4d_authority_sensitivity" in stdout
    assert str(sim_root) not in stdout


def _write_s8_reference(output_dir, *, seed):
    s8_dir = output_dir / ("eval_seed_%s" % seed) / "s8_reference"
    s8_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(s8_dir / "suite_summary.csv", [_s8_summary(seed)])
    write_json(s8_dir / "phase4d_eval_s8_acceptance.json", {
        "phase4d_eval_s8_status": "pass",
        "failed_checks": "",
        "route_return_code": 0,
    })
    write_json(s8_dir / "phase4d_eval_s8_manifest.json", {
        "artifact_id": "s8-%s" % seed,
        "reference_status": "pass",
    })


def _write_gain_output(
    output_dir,
    *,
    seed,
    scale,
    final_residual,
    residual_scale_clip_ratio,
    delta_rms_vertical_acc,
):
    label = action_scale_label(scale)
    gain_dir = output_dir / ("eval_seed_%s" % seed) / ("gain_%s" % label)
    compare_dir = output_dir / ("eval_seed_%s" % seed) / ("compare_gain_%s" % label)
    gain_dir.mkdir(parents=True, exist_ok=True)
    compare_dir.mkdir(parents=True, exist_ok=True)
    row = _s4_summary(seed)
    row.update({
        "rl_action_scale_mean": scale,
        "raw_mean_abs_action": 0.2,
        "raw_mean_abs_residual_damper": 0.02,
        "scaled_mean_abs_residual_damper": 0.02 * scale,
        "final_mean_abs_residual_damper": final_residual,
        "mean_abs_residual_damper": final_residual,
        "effective_control_ratio": 1 if scale > 0 else 0,
        "residual_scale_clip_ratio": residual_scale_clip_ratio,
        "damper_final_clamp_ratio": 0,
        "residual_saturation_ratio": residual_scale_clip_ratio,
    })
    _write_csv(gain_dir / "suite_summary.csv", [row])
    write_json(gain_dir / "phase4d_eval_s4_acceptance.json", {
        "phase4d_eval_s4_status": "pass",
        "failed_checks": "",
        "route_return_code": 0,
        "seed": seed,
    })
    write_json(gain_dir / "phase4d_eval_s4_manifest.json", {
        "artifact_id": "s4-%s-%s" % (seed, label),
        "source_train_artifact_id": "phase4d-train-test-artifact",
        "eval_seed": seed,
    })
    write_json(compare_dir / "phase4d_split_comparison.json", {
        "phase4d_compare_status": "pass",
        "policy_selection_status": "POLICY_ACCEPTED_ENGINEERING",
        "warmup_excluded_comfort_comfort_score_delta_s4_minus_s8": -0.03 * scale,
        "warmup_excluded_comfort_rms_vertical_acc_delta_s4_minus_s8": delta_rms_vertical_acc,
        "warmup_excluded_comfort_rms_vertical_jerk_delta_s4_minus_s8": -0.01 * scale,
        "warmup_excluded_comfort_peak_abs_vertical_acc_delta_s4_minus_s8": -0.01 * scale,
    })


def _write_scripted_output(
    output_dir,
    *,
    seed,
    residual,
    residual_value,
    rms_vertical_acc,
    residual_scale_clip_ratio=0.0,
):
    residual_dir = output_dir / ("eval_seed_%s" % seed) / residual
    residual_dir.mkdir(parents=True, exist_ok=True)
    row = _s4_summary(seed)
    row.update({
        "rl_residual_modes": "scripted",
        "rl_scripted_residual_kinds": residual,
        "raw_mean_abs_action": abs(residual_value),
        "raw_mean_abs_residual_damper": abs(residual_value),
        "scaled_mean_abs_residual_damper": abs(residual_value),
        "final_mean_abs_residual_damper": abs(residual_value),
        "mean_abs_residual_damper": abs(residual_value),
        "effective_control_ratio": 1 if residual_value else 0,
        "residual_scale_clip_ratio": residual_scale_clip_ratio,
        "damper_final_clamp_ratio": 0,
        "residual_saturation_ratio": residual_scale_clip_ratio,
        "warmup_excluded_comfort_rms_vertical_acc": rms_vertical_acc,
        "warmup_excluded_comfort_comfort_score": rms_vertical_acc,
        "warmup_excluded_comfort_rms_vertical_jerk": 2.0 * rms_vertical_acc,
        "warmup_excluded_comfort_peak_abs_vertical_acc": 3.0 * rms_vertical_acc,
    })
    _write_csv(residual_dir / "suite_summary.csv", [row])


def _s4_summary(seed):
    return {
        "seed": seed,
        "scenario": "S4_rl_residual_skyhook",
        "status": "ok",
        "return_code": 0,
        "global_status": "Perfect",
        "score_composed": 100,
        "policy_available_ratio": 1,
        "sidecar_fatal_errors": 0,
        "sidecar_runtime_errors": 0,
        "sidecar_command_verifies": 100,
        "collision_count": 0,
        "lane_invasion_count": 0,
        "red_light_count": 0,
        "route_timeout_count": 0,
        "blocked_vehicle_count": 0,
        "fallback_ratio": 0,
        "low_speed_mask_ratio": 0,
        "soft_safety_gain_ratio": 0,
        "hard_safety_gate_ratio": 0,
    }


def _s8_summary(seed):
    return {
        "seed": seed,
        "scenario": "S8_rl_zero_residual_skyhook",
        "status": "ok",
        "return_code": 0,
        "global_status": "Perfect",
        "score_composed": 100,
        "sidecar_fatal_errors": 0,
        "collision_count": 0,
        "lane_invasion_count": 0,
        "red_light_count": 0,
        "route_timeout_count": 0,
        "blocked_vehicle_count": 0,
    }


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _csv_rows(path):
    with open(path, "r", newline="") as csv_file:
        return [dict(row) for row in csv.DictReader(csv_file)]


def _read_json(path):
    with open(path) as json_file:
        return json.load(json_file)


def _taeho_root():
    return Path(__file__).resolve().parents[1]


def _sim_root():
    return Path(__file__).resolve().parents[4]


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
