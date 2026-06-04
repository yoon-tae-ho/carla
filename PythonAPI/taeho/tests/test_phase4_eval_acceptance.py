import csv
import json

from suspension_control.rl.phase4_acceptance import (
    phase4_eval_acceptance_rows,
    write_phase4_eval_acceptance_report,
)


def test_phase4_eval_acceptance_passes_clean_route_rows(tmp_path):
    rows = phase4_eval_acceptance_rows([{
        "scenario": "S9_rl_residual_skyhook",
        "score_route": "100",
        "score_composed": "100",
        "score_penalty": "0",
        "sidecar_command_verifies": "8",
        "rl_policy_available_ratio": "1.0",
        "rl_fallback_reasons": "",
        "reward_rows": "100",
        "collision_count": "0",
        "lane_invasion_count": "0",
        "red_light_count": "0",
        "phase3b_status": "warn",
    }])

    assert rows[0]["phase4_eval_status"] == "pass"
    assert rows[0]["warnings"] == ""


def test_phase4_eval_acceptance_flags_policy_fallback_and_writes_files(tmp_path):
    summary_path = tmp_path / "suite_summary.csv"
    with open(summary_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=[
            "scenario",
            "score_route",
            "score_composed",
            "score_penalty",
            "sidecar_command_verifies",
            "rl_policy_available_ratio",
            "rl_fallback_reasons",
            "reward_rows",
        ])
        writer.writeheader()
        writer.writerow({
            "scenario": "S9_rl_residual_skyhook",
            "score_route": "100",
            "score_composed": "100",
            "score_penalty": "0",
            "sidecar_command_verifies": "4",
            "rl_policy_available_ratio": "0.0",
            "rl_fallback_reasons": "missing_path",
            "reward_rows": "10",
        })

    csv_path, json_path, rows = write_phase4_eval_acceptance_report(
        str(tmp_path),
        suite_summary_path=str(summary_path))

    assert rows[0]["phase4_eval_status"] == "fail"
    assert "policy_available_ratio" in rows[0]["failed_checks"]
    assert "policy_fallback" in rows[0]["failed_checks"]
    with open(csv_path) as csv_file:
        csv_rows = list(csv.DictReader(csv_file))
    with open(json_path) as json_file:
        json_rows = json.load(json_file)
    assert csv_rows[0]["phase4_eval_status"] == "fail"
    assert json_rows[0]["phase4_eval_status"] == "fail"

