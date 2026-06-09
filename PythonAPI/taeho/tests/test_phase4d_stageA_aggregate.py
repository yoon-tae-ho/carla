import csv
import json
import subprocess
import sys

from suspension_control.rl.phase4d_compare_only import write_phase4d_compare_only
from suspension_control.rl.phase4d_stageA_repeat import (
    phase4d_stageA_seed_reuse_decision,
    write_phase4d_stageA_repeat_aggregate,
    write_phase4d_stageA_repeat_plan,
)
from tests.test_phase4d_stageA_repeat import make_train_manifest


def test_stageA_aggregate_three_passing_seeds_and_writes_stochasticity(tmp_path):
    stage_root, train_dir, manifest_path = _stage_plan(tmp_path, seeds=("100", "101", "102"))
    for seed in ("100", "101", "102"):
        _write_seed(stage_root, manifest_path, train_dir, seed=seed)

    summary_csv, summary_json, acceptance_json, _manifest_json, stochasticity_json, report = (
        write_phase4d_stageA_repeat_aggregate(output_dir=str(stage_root)))

    rows = _csv_rows(summary_csv)
    assert report["stageA_status"] == "STAGE_A_PASS_REPEATABLE_SAFETY"
    assert len(rows) == 3
    assert rows[0]["eval_seed"] == "100"
    assert rows[0]["seed_stageA_status"] == "STAGE_A_PASS"
    assert rows[0]["comfort_score_improved"] == "1"
    assert abs(float(rows[0]["delta_comfort_score"]) + 0.1) <= 1.0e-9
    assert abs(float(rows[1]["delta_rms_vertical_acc"]) + 0.1) <= 1.0e-9

    summary = _read_json(summary_json)
    acceptance = _read_json(acceptance_json)
    stochasticity = _read_json(stochasticity_json)
    assert summary["stageA_status"] == report["stageA_status"]
    assert acceptance["aggregate_ok"] == 1
    assert stochasticity["seed_status_counts"]["STAGE_A_PASS"] == 3
    assert stochasticity["metrics"]["delta_rms_vertical_acc"]["count"] == 3
    assert stochasticity["num_eval_seeds"] == 3
    assert stochasticity["num_s4_pass"] == 3
    assert stochasticity["metric_repeatability"]["comfort_score"]["improved_count"] == 3


def test_stageA_aggregate_mixed_performance_when_no_key_gain_repeats(tmp_path):
    stage_root, train_dir, manifest_path = _stage_plan(tmp_path, seeds=("100", "101", "102"))
    for seed in ("100", "101", "102"):
        _write_seed(
            stage_root,
            manifest_path,
            train_dir,
            seed=seed,
            s4_updates={
                "warmup_excluded_comfort_comfort_score": "1.0",
                "warmup_excluded_comfort_rms_lateral_jerk": "1.0",
                "warmup_excluded_comfort_rms_longitudinal_acc": "1.0",
                "warmup_excluded_comfort_rms_longitudinal_jerk": "1.0",
                "warmup_excluded_stability_rms_yaw_rate": "1.0",
                "warmup_excluded_stability_peak_abs_yaw_rate": "1.0",
            })

    _summary_csv, _summary_json, acceptance_json, _manifest_json, _stochasticity_json, report = (
        write_phase4d_stageA_repeat_aggregate(output_dir=str(stage_root)))

    acceptance = _read_json(acceptance_json)
    assert report["stageA_status"] == "STAGE_A_PASS_MIXED_PERFORMANCE"
    assert acceptance["aggregate_ok"] == 1


def test_stageA_aggregate_s4_safety_failure_becomes_stageA_safety_fail(tmp_path):
    stage_root, train_dir, manifest_path = _stage_plan(tmp_path, seeds=("100", "101"))
    _write_seed(stage_root, manifest_path, train_dir, seed="100")
    _write_seed(
        stage_root,
        manifest_path,
        train_dir,
        seed="101",
        s4_acceptance_updates={
            "phase4d_eval_s4_status": "fail",
            "failed_checks": "collision_count",
        },
        s4_updates={"collision_count": "1"},
        write_compare=False)

    _summary_csv, _summary_json, acceptance_json, _manifest_json, _stochasticity_json, report = (
        write_phase4d_stageA_repeat_aggregate(output_dir=str(stage_root)))

    acceptance = _read_json(acceptance_json)
    assert report["stageA_status"] == "STAGE_A_FAIL_SAFETY"
    assert acceptance["stageA_status"] == "STAGE_A_FAIL_SAFETY"
    assert acceptance["safety_failed_seed_count"] == 1
    assert "collision_count" in acceptance["failed_checks"]


def test_stageA_aggregate_control_failure_becomes_stageA_control_fail(tmp_path):
    stage_root, train_dir, manifest_path = _stage_plan(tmp_path, seeds=("100", "101"))
    _write_seed(stage_root, manifest_path, train_dir, seed="100")
    _write_seed(
        stage_root,
        manifest_path,
        train_dir,
        seed="101",
        s4_acceptance_updates={
            "phase4d_eval_s4_status": "fail",
            "failed_checks": "mean_abs_action;effective_control_ratio",
        },
        s4_updates={
            "rl_mean_abs_action": "0",
            "rl_mean_abs_residual_damper": "0",
            "effective_control_ratio": "0",
        },
        write_compare=False)

    _summary_csv, _summary_json, acceptance_json, _manifest_json, _stochasticity_json, report = (
        write_phase4d_stageA_repeat_aggregate(output_dir=str(stage_root)))

    acceptance = _read_json(acceptance_json)
    assert report["stageA_status"] == "STAGE_A_FAIL_CONTROL_VALIDITY"
    assert acceptance["control_failed_seed_count"] == 1
    assert "mean_abs_action" in acceptance["failed_checks"]


def test_stageA_aggregate_missing_s8_output_is_incomplete(tmp_path):
    stage_root, train_dir, manifest_path = _stage_plan(tmp_path, seeds=("100",))
    _write_seed(stage_root, manifest_path, train_dir, seed="100", write_s8=False, write_compare=False)

    summary_csv, _summary_json, acceptance_json, _manifest_json, _stochasticity_json, report = (
        write_phase4d_stageA_repeat_aggregate(output_dir=str(stage_root)))

    rows = _csv_rows(summary_csv)
    acceptance = _read_json(acceptance_json)
    assert rows[0]["seed_stageA_status"] == "STAGE_A_INCOMPLETE"
    assert report["stageA_status"] == "STAGE_A_INCOMPLETE"
    assert acceptance["incomplete_seed_count"] == 1
    assert "s8_missing" in acceptance["failed_checks"]


def test_stageA_compare_and_aggregate_treat_comfort_score_as_lower_is_better(tmp_path):
    stage_root, train_dir, manifest_path = _stage_plan(tmp_path, seeds=("100",))
    _write_seed(
        stage_root,
        manifest_path,
        train_dir,
        seed="100",
        s4_updates={"warmup_excluded_comfort_comfort_score": "0.9"},
        s8_updates={"warmup_excluded_comfort_comfort_score": "1.0"})

    rows = _csv_rows(write_phase4d_stageA_repeat_aggregate(output_dir=str(stage_root))[0])
    assert abs(float(rows[0]["delta_comfort_score"]) + 0.1) <= 1.0e-9
    assert rows[0]["comfort_score_improved"] == "1"

    comparison = _read_json(stage_root / "eval_seed_100" / "compare" / "phase4d_split_comparison.json")
    assert comparison["warmup_excluded_comfort_comfort_score_improvement"] > 0


def test_stageA_reuse_requires_matching_lineage(tmp_path):
    stage_root, train_dir, manifest_path = _stage_plan(tmp_path, seeds=("100",))
    _write_seed(stage_root, manifest_path, train_dir, seed="100")

    ok = phase4d_stageA_seed_reuse_decision(
        output_dir=str(stage_root),
        train_manifest_path=str(manifest_path),
        eval_seed="100",
        routes_subset="00",
        baseline="skyhook",
        planning_provider="empty")
    mismatch = phase4d_stageA_seed_reuse_decision(
        output_dir=str(stage_root),
        train_manifest_path=str(manifest_path),
        eval_seed="101",
        routes_subset="00",
        baseline="skyhook",
        planning_provider="empty")

    assert ok["reuse_ok"] == 1
    assert mismatch["reuse_ok"] == 0
    assert "s4_eval_seed" in mismatch["failed_checks"]


def test_stageA_aggregate_cli_writes_outputs(tmp_path):
    stage_root, train_dir, manifest_path = _stage_plan(tmp_path, seeds=("100",))
    _write_seed(stage_root, manifest_path, train_dir, seed="100")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "suspension_control.rl.phase4d_stageA_repeat",
            "--action",
            "aggregate",
            "--output-dir",
            str(stage_root),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)

    assert "phase4d Stage A repeat summary:" in result.stdout
    assert (stage_root / "phase4d_stageA_repeat_summary.csv").is_file()
    assert (stage_root / "phase4d_stageA_stochasticity_report.json").is_file()


def _stage_plan(tmp_path, *, seeds):
    _train_root, train_dir, manifest_path = make_train_manifest(tmp_path)
    stage_root = tmp_path / "stageA"
    write_phase4d_stageA_repeat_plan(
        output_dir=str(stage_root),
        train_manifest_path=str(manifest_path),
        train_dir=str(train_dir),
        eval_seeds=list(seeds),
        routes_subset="00",
        baseline="skyhook",
        planning_provider="empty",
        dry_run=False)
    return stage_root, train_dir, manifest_path


def _write_seed(
    stage_root,
    train_manifest_path,
    train_dir,
    *,
    seed,
    s4_updates=None,
    s8_updates=None,
    s4_acceptance_updates=None,
    write_s8=True,
    write_compare=True,
):
    seed_dir = stage_root / ("eval_seed_%s" % seed)
    s4_dir = seed_dir / "s4"
    s8_dir = seed_dir / "s8"
    compare_dir = seed_dir / "compare"
    s4_dir.mkdir(parents=True, exist_ok=True)
    s8_dir.mkdir(parents=True, exist_ok=True)
    compare_dir.mkdir(parents=True, exist_ok=True)
    train_manifest = _read_json(train_manifest_path)
    s4 = _s4_summary(seed)
    s8 = _s8_summary(seed)
    s4.update(s4_updates or {})
    s8.update(s8_updates or {})
    _write_json(s4_dir / "suite_summary.json", [s4])
    _write_json(s4_dir / "suite_summary.csv", [s4])
    s4_acceptance = {
        "phase4d_eval_s4_status": "pass",
        "eval_status": "completed",
        "failed_checks": "",
        "warnings": "",
        "routes_subset": "00",
        "seed": seed,
        "baseline": "skyhook",
        "planning_provider": "empty",
        "policy_sha256": train_manifest["policy_sha256"],
        "normalizer_sha256": train_manifest["normalizer_sha256"],
        "source_train_artifact_id": train_manifest["artifact_id"],
    }
    s4_acceptance.update(s4_acceptance_updates or {})
    _write_json(s4_dir / "phase4d_eval_s4_acceptance.json", s4_acceptance)
    _write_json(s4_dir / "phase4d_eval_s4_manifest.json", {
        "artifact_id": "phase4d-eval-s4-%s" % seed,
        "source_train_artifact_id": train_manifest["artifact_id"],
        "artifact_status": "accepted_eval",
        "routes_subset": "00",
        "eval_seed": seed,
        "baseline": "skyhook",
        "planning_provider": "empty",
        "policy_path": str(train_dir / "policy.ts"),
        "normalizer_path": str(train_dir / "normalizer.json"),
        "policy_sha256": train_manifest["policy_sha256"],
        "normalizer_sha256": train_manifest["normalizer_sha256"],
    })
    if write_s8:
        _write_json(s8_dir / "suite_summary.json", [s8])
        _write_json(s8_dir / "suite_summary.csv", [s8])
        _write_json(s8_dir / "phase4d_eval_s8_acceptance.json", {
            "phase4d_eval_s8_status": "pass",
            "reference_status": "accepted_reference",
            "failed_checks": "",
            "warnings": "",
            "routes_subset": "00",
            "eval_seed": seed,
            "baseline": "skyhook",
            "planning_provider": "empty",
        })
        _write_json(s8_dir / "phase4d_eval_s8_manifest.json", {
            "artifact_id": "phase4d-eval-s8-%s" % seed,
            "reference_status": "accepted_reference",
            "routes_subset": "00",
            "eval_seed": seed,
            "baseline": "skyhook",
            "planning_provider": "empty",
            "controller": "rl_zero_residual_skyhook",
        })
    if write_compare and write_s8:
        write_phase4d_compare_only(
            output_dir=str(compare_dir),
            s4_summary_path=str(s4_dir / "suite_summary.json"),
            s4_acceptance_path=str(s4_dir / "phase4d_eval_s4_acceptance.json"),
            s4_manifest_path=str(s4_dir / "phase4d_eval_s4_manifest.json"),
            s8_summary_path=str(s8_dir / "suite_summary.json"),
            s8_acceptance_path=str(s8_dir / "phase4d_eval_s8_acceptance.json"),
            s8_manifest_path=str(s8_dir / "phase4d_eval_s8_manifest.json"),
            train_manifest_path=str(train_manifest_path))


def _s4_summary(seed):
    row = _base_summary(seed, "S4_rl_residual_skyhook")
    row.update({
        "controller": "rl_residual_skyhook",
        "rl_policy_available_ratio": "1.0",
        "rl_mean_abs_action": "0.2",
        "rl_mean_abs_residual_damper": "0.02",
        "mean_abs_action": "0.2",
        "mean_abs_residual_damper": "0.02",
        "effective_control_ratio": "0.8",
        "warmup_excluded_comfort_comfort_score": "0.9",
        "warmup_excluded_comfort_rms_vertical_acc": "0.9",
        "warmup_excluded_comfort_rms_vertical_jerk": "0.9",
        "warmup_excluded_comfort_peak_abs_vertical_acc": "0.9",
        "warmup_excluded_comfort_peak_abs_vertical_jerk": "0.9",
        "warmup_excluded_comfort_rms_lateral_acc": "0.9",
        "warmup_excluded_comfort_rms_lateral_jerk": "0.9",
        "warmup_excluded_comfort_rms_longitudinal_acc": "0.9",
        "warmup_excluded_comfort_rms_longitudinal_jerk": "0.9",
        "warmup_excluded_stability_rms_roll": "0.9",
        "warmup_excluded_stability_rms_pitch": "0.9",
        "warmup_excluded_stability_rms_yaw_rate": "0.9",
        "warmup_excluded_stability_peak_abs_roll": "0.9",
        "warmup_excluded_stability_peak_abs_pitch": "0.9",
        "warmup_excluded_stability_peak_abs_yaw_rate": "0.9",
    })
    return row


def _s8_summary(seed):
    row = _base_summary(seed, "S8_rl_zero_residual_skyhook")
    row.update({
        "controller": "rl_zero_residual_skyhook",
        "rl_policy_available_ratio": "1.0",
        "rl_mean_abs_action": "0.0",
        "rl_mean_abs_residual_damper": "0.0",
        "mean_abs_action": "0.0",
        "mean_abs_residual_damper": "0.0",
        "effective_control_ratio": "0.0",
        "warmup_excluded_comfort_comfort_score": "1.0",
        "warmup_excluded_comfort_rms_vertical_acc": "1.0",
        "warmup_excluded_comfort_rms_vertical_jerk": "1.0",
        "warmup_excluded_comfort_peak_abs_vertical_acc": "1.0",
        "warmup_excluded_comfort_peak_abs_vertical_jerk": "1.0",
        "warmup_excluded_comfort_rms_lateral_acc": "1.0",
        "warmup_excluded_comfort_rms_lateral_jerk": "1.0",
        "warmup_excluded_comfort_rms_longitudinal_acc": "1.0",
        "warmup_excluded_comfort_rms_longitudinal_jerk": "1.0",
        "warmup_excluded_stability_rms_roll": "1.0",
        "warmup_excluded_stability_rms_pitch": "1.0",
        "warmup_excluded_stability_rms_yaw_rate": "1.0",
        "warmup_excluded_stability_peak_abs_roll": "1.0",
        "warmup_excluded_stability_peak_abs_pitch": "1.0",
        "warmup_excluded_stability_peak_abs_yaw_rate": "1.0",
    })
    return row


def _base_summary(seed, scenario):
    return {
        "seed": seed,
        "scenario": scenario,
        "status": "ok",
        "return_code": "0",
        "global_status": "Perfect",
        "score_route": "100",
        "score_composed": "100",
        "score_penalty": "1.0",
        "sidecar_command_verifies": "10",
        "fallback_ratio": "0",
        "hard_safety_gate_ratio": "0",
        "collision_count": "0",
        "lane_invasion_count": "0",
        "red_light_count": "0",
        "route_timeout_count": "0",
        "blocked_vehicle_count": "0",
        "metric_source": "metrics_by_episode_main_actor",
        "metric_main_episode_index": "1",
        "metric_main_actor_id": "9502",
        "profile_total_rows": "6202",
        "profile_main_rows": "6197",
        "profile_excluded_non_main_rows": "5",
        "warmup_excluded_profile_rows": "6141",
        "warmup_excluded_start_elapsed_seconds": "3.3",
        "metric_actor_switch_detected": "1",
        "metric_elapsed_reset_detected": "1",
        "metric_invalid": "0",
    }


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, list):
        if path.suffix == ".csv":
            with open(path, "w", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=sorted(value[0].keys()))
                writer.writeheader()
                writer.writerows(value)
            return
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _read_json(path):
    with open(path) as json_file:
        return json.load(json_file)


def _csv_rows(path):
    with open(path, newline="") as csv_file:
        return [dict(row) for row in csv.DictReader(csv_file)]
