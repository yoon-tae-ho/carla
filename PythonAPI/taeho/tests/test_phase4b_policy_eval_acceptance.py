import csv
import json

from suspension_control.rl.phase4_acceptance import (
    phase4b_policy_eval_acceptance_rows,
    write_phase4b_policy_eval_acceptance_report,
)


def test_phase4b_policy_eval_acceptance_passes_clean_s8_s4_and_writes_files(tmp_path):
    summary_path = tmp_path / "suite_summary.csv"
    rows = [_reference_row(), _learned_row()]
    _write_suite_summary(summary_path, rows)

    csv_path, json_path, acceptance = write_phase4b_policy_eval_acceptance_report(
        str(tmp_path),
        suite_summary_path=str(summary_path))

    reference, learned = acceptance
    assert reference["scenario"] == "S8_rl_zero_residual_skyhook"
    assert reference["phase4b_role"] == "reference"
    assert reference["phase4b_status"] == "pass"
    assert reference["action_nonzero_ok"] == ""

    assert learned["scenario"] == "S4_rl_residual_skyhook"
    assert learned["phase4b_role"] == "learned"
    assert learned["phase4b_status"] == "pass"
    assert learned["failed_checks"] == ""
    assert learned["route_score_ok"] == 1
    assert learned["action_nonzero_ok"] == 1
    assert learned["residual_nonzero_ok"] == 1
    assert abs(learned["warmup_excluded_comfort_delta_vs_S8"] + 0.05) < 1.0e-9
    assert abs(learned["warmup_excluded_peak_roll_delta_vs_S8"] - 0.1) < 1.0e-9

    with open(csv_path) as csv_file:
        csv_rows = list(csv.DictReader(csv_file))
    with open(json_path) as json_file:
        json_rows = json.load(json_file)
    assert csv_rows[1]["phase4b_status"] == "pass"
    assert json_rows[1]["phase4b_status"] == "pass"


def test_phase4b_policy_eval_acceptance_flags_policy_unavailable():
    rows = phase4b_policy_eval_acceptance_rows([
        _reference_row(),
        _learned_row(
            rl_policy_available_ratio="0.0",
            rl_fallback_reasons="policy_unavailable"),
    ])

    learned = rows[1]
    assert learned["phase4b_status"] == "fail"
    assert "policy_available" in learned["failed_checks"]
    assert "no_policy_fallback" in learned["failed_checks"]
    assert "allowed_fallback_only" in learned["failed_checks"]


def test_phase4b_policy_eval_acceptance_flags_missing_policy_path():
    rows = phase4b_policy_eval_acceptance_rows([
        _reference_row(),
        _learned_row(rl_fallback_reasons="missing_path"),
    ])

    learned = rows[1]
    assert learned["phase4b_status"] == "fail"
    assert "no_policy_fallback" in learned["failed_checks"]


def test_phase4b_policy_eval_acceptance_allows_safety_gate_zero_only():
    rows = phase4b_policy_eval_acceptance_rows([
        _reference_row(),
        _learned_row(
            rl_fallback_reasons="safety_gate_zero",
            fallback_ratio="0.20",
            low_speed_mask_ratio="0.20"),
    ])

    learned = rows[1]
    assert learned["phase4b_status"] == "pass"
    assert learned["no_policy_fallback_ok"] == 1
    assert learned["allowed_fallback_only_ok"] == 1
    assert learned["fallback_ratio_ok"] == 1


def test_phase4b_policy_eval_acceptance_flags_hard_safety_gate():
    rows = phase4b_policy_eval_acceptance_rows([
        _reference_row(),
        _learned_row(hard_safety_gate_ratio="0.01"),
    ])

    learned = rows[1]
    assert learned["phase4b_status"] == "fail"
    assert "hard_safety" in learned["failed_checks"]


def test_phase4b_policy_eval_acceptance_missing_reference_warns_but_learned_can_pass():
    rows = phase4b_policy_eval_acceptance_rows([_learned_row()])

    reference, learned = rows
    assert reference["scenario"] == "S8_rl_zero_residual_skyhook"
    assert reference["phase4b_status"] == "warn"
    assert "reference_missing" in reference["warnings"]
    assert learned["phase4b_status"] == "pass"
    assert "reference_missing" in learned["warnings"]


def test_phase4b_policy_eval_acceptance_missing_learned_fails():
    rows = phase4b_policy_eval_acceptance_rows([_reference_row()])

    learned = rows[1]
    assert learned["scenario"] == "S4_rl_residual_skyhook"
    assert learned["phase4b_status"] == "fail"
    assert "learned_scenario_missing" in learned["failed_checks"]


def _reference_row(**overrides):
    row = _base_row("S8_rl_zero_residual_skyhook")
    row.update({
        "rl_mean_abs_action": "0.0",
        "rl_mean_abs_residual_damper": "0.0",
        "mean_abs_action": "0.0",
        "mean_abs_residual_damper": "0.0",
        "effective_control_ratio": "0.0",
        "warmup_excluded_comfort_comfort_score": "1.0",
        "warmup_excluded_stability_peak_abs_roll": "0.2",
        "mean_reward_total": "0.1",
    })
    row.update(overrides)
    return row


def _learned_row(**overrides):
    row = _base_row("S4_rl_residual_skyhook")
    row.update({
        "rl_mean_abs_action": "0.2",
        "rl_mean_abs_residual_damper": "0.02",
        "mean_abs_action": "0.2",
        "mean_abs_residual_damper": "0.02",
        "effective_control_ratio": "0.80",
        "warmup_excluded_comfort_comfort_score": "0.95",
        "warmup_excluded_stability_peak_abs_roll": "0.3",
        "mean_reward_total": "0.0",
    })
    row.update(overrides)
    return row


def _base_row(scenario):
    return {
        "scenario": scenario,
        "score_route": "100",
        "score_composed": "100",
        "score_penalty": "0",
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
    }


def _write_suite_summary(path, rows):
    fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
