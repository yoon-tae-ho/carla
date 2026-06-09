import csv
import json

from suspension_control.rl.phase4b_policy_eval_acceptance import (
    load_phase4b_summary_rows,
    write_phase4b_split_reference_acceptance,
)


def test_phase4b_split_reference_acceptance_compares_separate_json_files(tmp_path):
    reference_dir = tmp_path / "seed100_S8_rl_zero_residual_skyhook"
    candidate_dir = tmp_path / "seed100_S4_rl_residual_skyhook"
    reference_dir.mkdir()
    candidate_dir.mkdir()
    reference_summary = reference_dir / "summary.json"
    candidate_summary = candidate_dir / "summary.json"
    _write_json(reference_summary, _reference_row(scenario=""))
    _write_json(candidate_summary, _candidate_row(scenario=""))

    (
        acceptance_csv,
        acceptance_json,
        comparison_csv,
        comparison_json,
        acceptance,
        comparison,
    ) = write_phase4b_split_reference_acceptance(
        str(tmp_path / "out"),
        candidate_summary_path=str(candidate_summary),
        reference_summary_path=str(reference_summary),
    )

    assert acceptance[0]["scenario"] == "S8_rl_zero_residual_skyhook"
    assert acceptance[0]["phase4b_status"] == "pass"
    assert acceptance[1]["scenario"] == "S4_rl_residual_skyhook"
    assert acceptance[1]["phase4b_status"] == "pass"
    assert acceptance[1]["warnings"] == ""
    assert comparison[0]["phase4b_split_status"] == "pass"
    assert comparison[0]["score_route_delta"] == 0.0
    assert comparison[0]["mean_abs_residual_damper_delta"] == 0.02

    assert _csv_rows(acceptance_csv)[1]["phase4b_status"] == "pass"
    assert _json_rows(acceptance_json)[1]["phase4b_status"] == "pass"
    assert _csv_rows(comparison_csv)[0]["phase4b_split_status"] == "pass"
    assert _json_rows(comparison_json)[0]["phase4b_split_status"] == "pass"


def test_phase4b_split_reference_acceptance_loads_suite_summary_json(tmp_path):
    suite_summary = tmp_path / "suite_summary.json"
    _write_json(suite_summary, [_reference_row(), _candidate_row()])

    _, _, _, _, acceptance, comparison = write_phase4b_split_reference_acceptance(
        str(tmp_path / "out"),
        candidate_summary_path=str(suite_summary),
        reference_summary_path=str(suite_summary),
    )

    assert [row["phase4b_status"] for row in acceptance] == ["pass", "pass"]
    assert comparison[0]["phase4b_split_status"] == "pass"


def test_phase4b_split_reference_acceptance_missing_reference_warns(tmp_path):
    candidate_summary = tmp_path / "seed100_S4_rl_residual_skyhook" / "summary.json"
    candidate_summary.parent.mkdir()
    _write_json(candidate_summary, _candidate_row(scenario=""))

    _, _, _, _, acceptance, comparison = write_phase4b_split_reference_acceptance(
        str(tmp_path / "out"),
        candidate_summary_path=str(candidate_summary),
    )

    reference, candidate = acceptance
    assert reference["phase4b_status"] == "warn"
    assert "reference_missing" in reference["warnings"]
    assert candidate["phase4b_status"] == "pass"
    assert "reference_missing" in candidate["warnings"]
    assert comparison[0]["phase4b_split_status"] == "warn"
    assert "reference_missing" in comparison[0]["warnings"]


def test_phase4b_split_reference_acceptance_flags_policy_unavailable(tmp_path):
    reference_summary = tmp_path / "reference.json"
    candidate_summary = tmp_path / "candidate.json"
    _write_json(reference_summary, _reference_row())
    _write_json(candidate_summary, _candidate_row(
        rl_policy_available_ratio="0.0",
        rl_fallback_reasons="policy_unavailable",
    ))

    _, _, _, _, acceptance, comparison = write_phase4b_split_reference_acceptance(
        str(tmp_path / "out"),
        candidate_summary_path=str(candidate_summary),
        reference_summary_path=str(reference_summary),
    )

    assert acceptance[1]["phase4b_status"] == "fail"
    assert "policy_available" in acceptance[1]["failed_checks"]
    assert comparison[0]["phase4b_split_status"] == "fail"


def test_phase4b_split_reference_acceptance_allows_safety_gate_zero(tmp_path):
    reference_summary = tmp_path / "reference.json"
    candidate_summary = tmp_path / "candidate.json"
    _write_json(reference_summary, _reference_row())
    _write_json(candidate_summary, _candidate_row(
        rl_fallback_reasons="safety_gate_zero",
        fallback_ratio="0.20",
        low_speed_mask_ratio="0.20",
    ))

    _, _, _, _, acceptance, comparison = write_phase4b_split_reference_acceptance(
        str(tmp_path / "out"),
        candidate_summary_path=str(candidate_summary),
        reference_summary_path=str(reference_summary),
    )

    assert acceptance[1]["phase4b_status"] == "pass"
    assert acceptance[1]["allowed_fallback_only_ok"] == 1
    assert comparison[0]["phase4b_split_status"] == "pass"


def test_phase4b_split_reference_loader_parses_leaderboard_result_json(tmp_path):
    result_json = tmp_path / "seed100_S8_rl_zero_residual_skyhook" / "tfpp" / "result.json"
    result_json.parent.mkdir(parents=True)
    _write_json(result_json, _leaderboard_result())

    rows = load_phase4b_summary_rows(str(result_json))

    assert len(rows) == 1
    row = rows[0]
    assert row["scenario"] == "S8_rl_zero_residual_skyhook"
    assert row["score_route"] == 100.0
    assert row["score_composed"] == 100.0
    assert row["score_penalty"] == 1.0
    assert row["infractions"]["collisions_vehicle"] == 0.0


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


def _candidate_row(**overrides):
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
        },
    }


def _leaderboard_result():
    return {
        "_checkpoint": {
            "global_record": {
                "status": "Perfect",
                "infractions": {
                    "collisions_layout": 0.0,
                    "collisions_pedestrian": 0.0,
                    "collisions_vehicle": 0.0,
                    "red_light": 0.0,
                    "stop_infraction": 0.0,
                    "outside_route_lanes": 0,
                    "route_timeout": 0.0,
                    "route_dev": 0.0,
                    "vehicle_blocked": 0.0,
                },
                "scores_mean": {
                    "score_composed": 100.0,
                    "score_route": 100.0,
                    "score_penalty": 1.0,
                },
                "meta": {
                    "duration_game": 297.0,
                    "duration_system": 1400.0,
                },
            }
        }
    }


def _write_json(path, data):
    with open(path, "w") as json_file:
        json.dump(data, json_file)


def _csv_rows(path):
    with open(path, newline="") as csv_file:
        return [dict(row) for row in csv.DictReader(csv_file)]


def _json_rows(path):
    with open(path) as json_file:
        return json.load(json_file)
