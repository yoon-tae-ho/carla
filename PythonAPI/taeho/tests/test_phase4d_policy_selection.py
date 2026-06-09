import csv
import json

from suspension_control.rl.phase4d_policy_selection import (
    phase4d_policy_selection_report,
    write_phase4d_policy_selection,
)


def test_phase4d_policy_selection_candidate_for_repeat_and_writes_files(tmp_path):
    reference, candidate, training = _write_inputs(tmp_path)

    csv_path, json_path, report = write_phase4d_policy_selection(
        str(tmp_path / "out"),
        candidate_summary_path=str(candidate),
        reference_summary_path=str(reference),
        training_summary_path=str(training),
    )

    assert report["phase4d_status"] == "candidate_for_repeat"
    assert report["summary"]["engineering_pass"] == 1
    assert report["summary"]["reward_acceptable"] == 1
    assert report["summary"]["comfort_promising"] == 1
    assert report["summary"]["stability_risk"] == 0
    assert "paper_candidate" not in report["policy_tags"]
    assert report["metrics"]["warmup_excluded_comfort_comfort_score"]["improvement_pct"] >= 0.02

    csv_row = _csv_rows(csv_path)[0]
    json_report = _json_report(json_path)
    assert csv_row["phase4d_status"] == "candidate_for_repeat"
    assert json_report["phase4d_status"] == "candidate_for_repeat"


def test_phase4d_policy_selection_fails_nonzero_collision(tmp_path):
    reference, candidate, training = _write_inputs(
        tmp_path,
        candidate_overrides={"collision_count": "1"},
    )

    report = phase4d_policy_selection_report(
        candidate_summary_path=str(candidate),
        reference_summary_path=str(reference),
        training_summary_path=str(training),
    )

    assert report["phase4d_status"] == "fail"
    assert report["summary"]["infraction_ok"] == 0
    assert "infraction" in report["summary"]["failed_checks"]


def test_phase4d_policy_selection_fails_policy_unavailable_fallback(tmp_path):
    reference, candidate, training = _write_inputs(
        tmp_path,
        candidate_overrides={
            "rl_policy_available_ratio": "0.0",
            "rl_fallback_reasons": "policy_unavailable",
        },
    )

    report = phase4d_policy_selection_report(
        candidate_summary_path=str(candidate),
        reference_summary_path=str(reference),
        training_summary_path=str(training),
    )

    assert report["phase4d_status"] == "fail"
    assert report["summary"]["policy_available_ok"] == 0
    assert report["summary"]["no_policy_fallback_ok"] == 0


def test_phase4d_policy_selection_allows_safety_gate_zero_fallback(tmp_path):
    reference, candidate, training = _write_inputs(
        tmp_path,
        candidate_overrides={
            "rl_fallback_reasons": "safety_gate_zero",
            "fallback_ratio": "0.20",
            "low_speed_mask_ratio": "0.20",
        },
    )

    report = phase4d_policy_selection_report(
        candidate_summary_path=str(candidate),
        reference_summary_path=str(reference),
        training_summary_path=str(training),
    )

    assert report["summary"]["allowed_fallback_only_ok"] == 1
    assert report["summary"]["no_policy_fallback_ok"] == 1
    assert report["summary"]["fallback_ratio_ok"] == 1
    assert report["summary"]["engineering_pass"] == 1


def test_phase4d_policy_selection_reward_threshold_safe_but_not_better(tmp_path):
    reference, candidate, training = _write_inputs(
        tmp_path,
        reference_overrides={"mean_reward_total": "0.10"},
        candidate_overrides={"mean_reward_total": "-0.05"},
    )

    report = phase4d_policy_selection_report(
        candidate_summary_path=str(candidate),
        reference_summary_path=str(reference),
        training_summary_path=str(training),
    )

    assert report["summary"]["reward_acceptable"] == 0
    assert report["phase4d_status"] == "safe_but_not_better"


def test_phase4d_policy_selection_no_comfort_gain_safe_but_not_better(tmp_path):
    reference, candidate, training = _write_inputs(
        tmp_path,
        candidate_overrides={
            "warmup_excluded_comfort_comfort_score": "1.01",
            "warmup_excluded_comfort_rms_vertical_acc": "0.995",
        },
    )

    report = phase4d_policy_selection_report(
        candidate_summary_path=str(candidate),
        reference_summary_path=str(reference),
        training_summary_path=str(training),
    )

    assert report["summary"]["comfort_promising"] == 0
    assert report["phase4d_status"] == "safe_but_not_better"


def test_phase4d_policy_selection_stability_risk_tradeoff(tmp_path):
    reference, candidate, training = _write_inputs(
        tmp_path,
        candidate_overrides={
            "warmup_excluded_stability_peak_abs_roll": "1.06",
        },
    )

    report = phase4d_policy_selection_report(
        candidate_summary_path=str(candidate),
        reference_summary_path=str(reference),
        training_summary_path=str(training),
    )

    assert report["summary"]["stability_risk"] == 1
    assert report["phase4d_status"] == "comfort_promising_with_tradeoff"


def test_phase4d_policy_selection_action_overuse_warning(tmp_path):
    reference, candidate, training = _write_inputs(
        tmp_path,
        candidate_overrides={
            "rl_mean_abs_action": "0.71",
            "mean_abs_action": "0.71",
        },
    )

    report = phase4d_policy_selection_report(
        candidate_summary_path=str(candidate),
        reference_summary_path=str(reference),
        training_summary_path=str(training),
    )

    assert report["summary"]["action_overuse_warning"] == 1
    assert "action_overuse" in report["policy_tags"]
    assert "action_overuse" in report["summary"]["warnings"]


def test_phase4d_policy_selection_never_outputs_paper_candidate(tmp_path):
    reference, candidate, training = _write_inputs(tmp_path)

    report = phase4d_policy_selection_report(
        candidate_summary_path=str(candidate),
        reference_summary_path=str(reference),
        training_summary_path=str(training),
    )

    assert report["phase4d_status"] != "paper_candidate"
    assert "paper_candidate" not in report["summary"].get("policy_tags", "")
    assert "paper_candidate" not in report


def _write_inputs(tmp_path, reference_overrides=None, candidate_overrides=None):
    reference_dir = tmp_path / "seed100_S8_rl_zero_residual_skyhook"
    candidate_dir = tmp_path / "seed100_S4_rl_residual_skyhook"
    train_dir = tmp_path / "train"
    reference_dir.mkdir()
    candidate_dir.mkdir()
    train_dir.mkdir()
    reference_path = reference_dir / "summary.json"
    candidate_path = candidate_dir / "summary.json"
    training_path = train_dir / "training_summary.json"
    reference = _reference_row()
    candidate = _candidate_row()
    if reference_overrides:
        reference.update(reference_overrides)
    if candidate_overrides:
        candidate.update(candidate_overrides)
    _write_json(reference_path, reference)
    _write_json(candidate_path, candidate)
    _write_json(training_path, {
        "status": "trained",
        "backend": "live",
        "total_timesteps": 8192,
        "rollout_rows": 8192,
    })
    return reference_path, candidate_path, training_path


def _reference_row():
    row = _base_row("S8_rl_zero_residual_skyhook")
    row.update({
        "rl_mean_abs_action": "0.0",
        "rl_mean_abs_residual_damper": "0.0",
        "mean_abs_action": "0.0",
        "mean_abs_residual_damper": "0.0",
        "effective_control_ratio": "0.0",
        "mean_reward_total": "0.10",
        "mean_reward_action": "-0.02",
        "warmup_excluded_comfort_comfort_score": "1.00",
        "warmup_excluded_comfort_rms_vertical_acc": "1.00",
        "warmup_excluded_comfort_rms_lateral_acc": "1.00",
        "warmup_excluded_stability_peak_abs_roll": "1.00",
        "warmup_excluded_stability_rms_roll": "1.00",
        "warmup_excluded_stability_peak_abs_lateral_acc": "1.00",
        "warmup_excluded_stability_rms_lateral_acc": "1.00",
    })
    return row


def _candidate_row():
    row = _base_row("S4_rl_residual_skyhook")
    row.update({
        "rl_mean_abs_action": "0.20",
        "rl_mean_abs_residual_damper": "0.02",
        "mean_abs_action": "0.20",
        "mean_abs_residual_damper": "0.02",
        "effective_control_ratio": "0.80",
        "mean_reward_total": "0.05",
        "mean_reward_action": "-0.05",
        "warmup_excluded_comfort_comfort_score": "1.03",
        "warmup_excluded_comfort_rms_vertical_acc": "0.98",
        "warmup_excluded_comfort_rms_lateral_acc": "0.98",
        "warmup_excluded_stability_peak_abs_roll": "1.00",
        "warmup_excluded_stability_rms_roll": "1.00",
        "warmup_excluded_stability_peak_abs_lateral_acc": "1.00",
        "warmup_excluded_stability_rms_lateral_acc": "1.00",
    })
    return row


def _base_row(scenario):
    return {
        "scenario": scenario,
        "score_route": "100",
        "score_composed": "100",
        "score_penalty": "1.0",
        "sidecar_command_verifies": "8",
        "rl_policy_available_ratio": "1.0",
        "rl_fallback_rows": "0",
        "rl_fallback_reasons": "",
        "fallback_ratio": "0.0",
        "low_speed_mask_ratio": "0.10",
        "hard_safety_gate_ratio": "0.0",
        "observation_clip_ratio": "0.0",
        "reward_rows": "100",
        "reward_row_ratio": "1.0",
        "diagnostic_rows": "100",
        "route_completion_proxy": "1.0",
        "route_progress_monotonic_fraction_max": "1.0",
        "collision_count": "0",
        "lane_invasion_count": "0",
        "red_light_count": "0",
        "blocked_vehicle_count": "0",
        "route_timeout_count": "0",
        "infractions": {
            "collisions_layout": 0.0,
            "collisions_pedestrian": 0.0,
            "collisions_vehicle": 0.0,
            "red_light": 0.0,
            "stop_infraction": 0.0,
            "outside_route_lanes": 0.0,
            "route_timeout": 0.0,
        },
    }


def _write_json(path, data):
    with open(path, "w") as json_file:
        json.dump(data, json_file)


def _csv_rows(path):
    with open(path, newline="") as csv_file:
        return [dict(row) for row in csv.DictReader(csv_file)]


def _json_report(path):
    with open(path) as json_file:
        return json.load(json_file)
