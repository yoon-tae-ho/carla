import json

from suspension_control.rl.phase4d_compare_only import (
    PAPER_WARNING,
    phase4d_compare_only_report,
    write_phase4d_compare_only,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def make_inputs(tmp_path, *, s4_updates=None, s8_updates=None, s4_acceptance_updates=None,
                s4_manifest_updates=None, s8_manifest_updates=None,
                include_s8_acceptance=True, include_s8_manifest=True):
    root = tmp_path / "inputs"
    s4_summary = root / "s4" / "suite_summary.json"
    s8_summary = root / "s8" / "suite_summary.json"
    s4_acceptance = root / "s4" / "phase4d_eval_s4_acceptance.json"
    s4_manifest = root / "s4" / "phase4d_eval_s4_manifest.json"
    s8_acceptance = root / "s8" / "phase4d_eval_s8_acceptance.json"
    s8_manifest = root / "s8" / "phase4d_eval_s8_manifest.json"
    train_manifest = root / "train" / "phase4d_train_manifest.json"

    s4 = _s4_row()
    s8 = _s8_row()
    s4.update(s4_updates or {})
    s8.update(s8_updates or {})
    write_json(s4_summary, [s4])
    write_json(s8_summary, [s8])

    s4_accept = {
        "phase4d_eval_s4_status": "pass",
        "eval_status": "completed",
        "routes_subset": "00",
        "seed": "100",
        "baseline": "skyhook",
        "planning_provider": "empty",
    }
    s4_accept.update(s4_acceptance_updates or {})
    write_json(s4_acceptance, s4_accept)

    s4_man = {
        "artifact_id": "phase4d-eval-s4-test",
        "source_train_artifact_id": "phase4d-train-test",
        "artifact_status": "accepted_eval",
        "routes_subset": "00",
        "eval_seed": "100",
        "baseline": "skyhook",
        "planning_provider": "empty",
        "policy_path": "/workspace/train/policy.ts",
        "normalizer_path": "/workspace/train/normalizer.json",
    }
    s4_man.update(s4_manifest_updates or {})
    write_json(s4_manifest, s4_man)

    write_json(train_manifest, {
        "artifact_id": "phase4d-train-test",
        "workflow_role": "train_only",
        "artifact_status": "accepted_for_eval",
        "routes_subset": "00",
        "train_seed": 100,
    })

    if include_s8_acceptance:
        write_json(s8_acceptance, {
            "phase4d_eval_s8_status": "pass",
            "reference_status": "accepted_reference",
            "routes_subset": "00",
            "eval_seed": "100",
            "baseline": "skyhook",
            "planning_provider": "empty",
        })
    if include_s8_manifest:
        s8_man = {
            "artifact_id": "phase4d-eval-s8-test",
            "reference_status": "accepted_reference",
            "routes_subset": "00",
            "eval_seed": "100",
            "baseline": "skyhook",
            "planning_provider": "empty",
            "controller": "rl_zero_residual_skyhook",
        }
        s8_man.update(s8_manifest_updates or {})
        write_json(s8_manifest, s8_man)

    return {
        "s4_summary": s4_summary,
        "s4_acceptance": s4_acceptance,
        "s4_manifest": s4_manifest,
        "s8_summary": s8_summary,
        "s8_acceptance": s8_acceptance if include_s8_acceptance else "",
        "s8_manifest": s8_manifest if include_s8_manifest else "",
        "train_manifest": train_manifest,
    }


def test_phase4d_compare_only_candidate_for_repeat_writes_outputs(tmp_path):
    paths = make_inputs(tmp_path)

    _csv, _json, selection_json, manifest_json, acceptance_json, report = write_phase4d_compare_only(
        output_dir=str(tmp_path / "out"),
        s4_summary_path=str(paths["s4_summary"]),
        s4_acceptance_path=str(paths["s4_acceptance"]),
        s4_manifest_path=str(paths["s4_manifest"]),
        s8_summary_path=str(paths["s8_summary"]),
        s8_acceptance_path=str(paths["s8_acceptance"]),
        s8_manifest_path=str(paths["s8_manifest"]),
        train_manifest_path=str(paths["train_manifest"]))

    assert report["policy_selection_report"]["policy_selection_status"] == "POLICY_CANDIDATE_FOR_REPEAT"
    assert report["comparison"]["comparison_ready"] == 1
    assert report["comparison"]["repeat_improved_metric_count"] >= 3
    assert PAPER_WARNING in report["policy_selection_report"]["warnings"]
    assert report["comparison"]["warmup_excluded_comfort_rms_vertical_acc_improvement"] > 0
    assert report["comparison"]["score_composed_improvement"] == 0
    for path in (selection_json, manifest_json, acceptance_json):
        with open(path) as json_file:
            assert json.load(json_file)


def test_phase4d_compare_only_accepts_engineering_when_not_enough_improvements(tmp_path):
    paths = make_inputs(
        tmp_path,
        s4_updates={
            "warmup_excluded_comfort_rms_vertical_acc": "1.0",
            "warmup_excluded_comfort_rms_vertical_jerk": "1.0",
            "warmup_excluded_comfort_rms_lateral_jerk": "1.0",
            "warmup_excluded_comfort_rms_lateral_acc": "1.0",
            "warmup_excluded_comfort_rms_longitudinal_acc": "1.0",
            "warmup_excluded_comfort_peak_abs_vertical_jerk": "1.0",
            "warmup_excluded_stability_peak_abs_roll": "1.0",
            "warmup_excluded_stability_peak_abs_pitch": "1.0",
            "warmup_excluded_stability_peak_abs_yaw_rate": "1.0",
            "warmup_excluded_stability_peak_abs_lateral_acc": "1.0",
        })

    report = phase4d_compare_only_report(
        output_dir=str(tmp_path / "out"),
        s4_summary_path=str(paths["s4_summary"]),
        s4_acceptance_path=str(paths["s4_acceptance"]),
        s4_manifest_path=str(paths["s4_manifest"]),
        s8_summary_path=str(paths["s8_summary"]),
        s8_acceptance_path=str(paths["s8_acceptance"]),
        s8_manifest_path=str(paths["s8_manifest"]),
        train_manifest_path=str(paths["train_manifest"]))

    assert report["policy_selection_report"]["policy_selection_status"] == "POLICY_ACCEPTED_ENGINEERING"
    assert PAPER_WARNING in report["policy_selection_report"]["warnings"]


def test_phase4d_compare_only_rejects_safety_before_control(tmp_path):
    paths = make_inputs(
        tmp_path,
        s4_updates={
            "collision_count": "1",
            "rl_policy_available_ratio": "0.0",
        })

    report = phase4d_compare_only_report(
        output_dir=str(tmp_path / "out"),
        s4_summary_path=str(paths["s4_summary"]),
        s4_acceptance_path=str(paths["s4_acceptance"]),
        s4_manifest_path=str(paths["s4_manifest"]),
        s8_summary_path=str(paths["s8_summary"]),
        s8_acceptance_path=str(paths["s8_acceptance"]),
        s8_manifest_path=str(paths["s8_manifest"]),
        train_manifest_path=str(paths["train_manifest"]))

    assert report["policy_selection_report"]["policy_selection_status"] == "POLICY_REJECTED_SAFETY"
    assert "collision_count" in report["policy_selection_report"]["safety_failed_checks"]


def test_phase4d_compare_only_rejects_control_invalid(tmp_path):
    paths = make_inputs(
        tmp_path,
        s4_updates={
            "rl_mean_abs_action": "0",
            "rl_mean_abs_residual_damper": "0",
            "effective_control_ratio": "0",
        })

    report = phase4d_compare_only_report(
        output_dir=str(tmp_path / "out"),
        s4_summary_path=str(paths["s4_summary"]),
        s4_acceptance_path=str(paths["s4_acceptance"]),
        s4_manifest_path=str(paths["s4_manifest"]),
        s8_summary_path=str(paths["s8_summary"]),
        s8_acceptance_path=str(paths["s8_acceptance"]),
        s8_manifest_path=str(paths["s8_manifest"]),
        train_manifest_path=str(paths["train_manifest"]))

    assert report["policy_selection_report"]["policy_selection_status"] == "POLICY_REJECTED_CONTROL_INVALID"
    assert "mean_abs_action" in report["policy_selection_report"]["control_failed_checks"]


def test_phase4d_compare_only_allows_partial_s8_lineage_with_warning(tmp_path):
    paths = make_inputs(tmp_path, include_s8_acceptance=False, include_s8_manifest=False)

    report = phase4d_compare_only_report(
        output_dir=str(tmp_path / "out"),
        s4_summary_path=str(paths["s4_summary"]),
        s4_acceptance_path=str(paths["s4_acceptance"]),
        s4_manifest_path=str(paths["s4_manifest"]),
        s8_summary_path=str(paths["s8_summary"]),
        train_manifest_path=str(paths["train_manifest"]))

    assert report["comparison"]["s8_reference_lineage_status"] == "partial"
    assert "s8_reference_lineage_partial" in report["policy_selection_report"]["warnings"]
    assert report["policy_selection_report"]["policy_selection_status"] in (
        "POLICY_ACCEPTED_ENGINEERING",
        "POLICY_CANDIDATE_FOR_REPEAT",
    )


def test_phase4d_compare_only_metadata_mismatch_warns(tmp_path):
    paths = make_inputs(tmp_path, s8_manifest_updates={"eval_seed": "101"})

    report = phase4d_compare_only_report(
        output_dir=str(tmp_path / "out"),
        s4_summary_path=str(paths["s4_summary"]),
        s4_acceptance_path=str(paths["s4_acceptance"]),
        s4_manifest_path=str(paths["s4_manifest"]),
        s8_summary_path=str(paths["s8_summary"]),
        s8_acceptance_path=str(paths["s8_acceptance"]),
        s8_manifest_path=str(paths["s8_manifest"]),
        train_manifest_path=str(paths["train_manifest"]))

    assert report["policy_selection_report"]["policy_selection_status"] == "COMPARISON_METADATA_WARNING"
    assert "eval_seed_mismatch" in report["policy_selection_report"]["metadata_warnings"]


def test_phase4d_compare_only_not_ready_without_s4_acceptance(tmp_path):
    paths = make_inputs(tmp_path)

    report = phase4d_compare_only_report(
        output_dir=str(tmp_path / "out"),
        s4_summary_path=str(paths["s4_summary"]),
        s4_acceptance_path=str(tmp_path / "missing_acceptance.json"),
        s4_manifest_path=str(paths["s4_manifest"]),
        s8_summary_path=str(paths["s8_summary"]),
        s8_acceptance_path=str(paths["s8_acceptance"]),
        s8_manifest_path=str(paths["s8_manifest"]),
        train_manifest_path=str(paths["train_manifest"]))

    assert report["policy_selection_report"]["policy_selection_status"] == "COMPARISON_NOT_READY"
    assert "s4_acceptance_missing" in report["policy_selection_report"]["failed_checks"]


def _s4_row():
    row = _base_row("S4_rl_residual_skyhook")
    row.update({
        "controller": "rl_residual_skyhook",
        "rl_policy_available_ratio": "1.0",
        "rl_mean_abs_action": "0.2",
        "rl_mean_abs_residual_damper": "0.02",
        "effective_control_ratio": "0.8",
        "warmup_excluded_comfort_rms_vertical_acc": "0.90",
        "warmup_excluded_comfort_rms_lateral_acc": "0.90",
        "warmup_excluded_comfort_rms_longitudinal_acc": "0.90",
        "warmup_excluded_comfort_rms_vertical_jerk": "0.90",
        "warmup_excluded_comfort_rms_lateral_jerk": "0.90",
        "warmup_excluded_stability_peak_abs_roll": "0.90",
        "warmup_excluded_stability_peak_abs_pitch": "0.90",
        "warmup_excluded_stability_peak_abs_yaw_rate": "0.90",
        "warmup_excluded_stability_peak_abs_lateral_acc": "0.90",
    })
    return row


def _s8_row():
    row = _base_row("S8_rl_zero_residual_skyhook")
    row.update({
        "controller": "rl_zero_residual_skyhook",
        "rl_policy_available_ratio": "1.0",
        "rl_mean_abs_action": "0",
        "rl_mean_abs_residual_damper": "0",
        "effective_control_ratio": "0",
        "warmup_excluded_comfort_rms_vertical_acc": "1.0",
        "warmup_excluded_comfort_rms_lateral_acc": "1.0",
        "warmup_excluded_comfort_rms_longitudinal_acc": "1.0",
        "warmup_excluded_comfort_rms_vertical_jerk": "1.0",
        "warmup_excluded_comfort_rms_lateral_jerk": "1.0",
        "warmup_excluded_stability_peak_abs_roll": "1.0",
        "warmup_excluded_stability_peak_abs_pitch": "1.0",
        "warmup_excluded_stability_peak_abs_yaw_rate": "1.0",
        "warmup_excluded_stability_peak_abs_lateral_acc": "1.0",
    })
    return row


def _base_row(scenario):
    return {
        "seed": "100",
        "scenario": scenario,
        "status": "ok",
        "return_code": "0",
        "score_route": "100",
        "score_composed": "100",
        "score_penalty": "1.0",
        "sidecar_command_verifies": "10",
        "fallback_ratio": "0",
        "hard_safety_gate_ratio": "0",
        "observation_clip_ratio": "0",
        "route_completion_proxy": "1.0",
        "route_progress_monotonic_fraction_max": "1.0",
        "reward_rows": "100",
        "reward_row_ratio": "1.0",
        "diagnostic_rows": "100",
        "mean_reward_total": "0.10",
        "collision_count": "0",
        "lane_invasion_count": "0",
        "red_light_count": "0",
        "route_timeout_count": "0",
        "blocked_vehicle_count": "0",
    }
