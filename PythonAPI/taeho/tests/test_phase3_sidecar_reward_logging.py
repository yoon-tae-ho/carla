from suspension_control.controllers.base import SuspensionCommand
from transfuser_suspension_control_suite import (
    DIAGNOSTIC_FIELDS,
    baseline_command_for_reward,
    route_progress_id_from_subset,
    summarize_diagnostics_csv,
)


def write_csv(path, fields, rows):
    with open(path, "w") as csv_file:
        csv_file.write(",".join(fields) + "\n")
        for row in rows:
            csv_file.write(",".join(str(row.get(field, "")) for field in fields) + "\n")


def test_phase3_summary_aggregates_reward_progress_and_task_fields(tmp_path):
    path = tmp_path / "controller_diagnostics.csv"
    fields = (
        "reward_total",
        "reward_comfort",
        "reward_stability",
        "reward_task",
        "reward_action",
        "reward_safety",
        "reward_cost_comfort",
        "reward_cost_stability",
        "reward_cost_task",
        "reward_cost_action",
        "reward_cost_safety",
        "reward_term_action_mag",
        "reward_term_action_rate",
        "reward_term_damper_rate",
        "reward_term_baseline_dev",
        "reward_term_low_speed_not_planned",
        "reward_term_progress_stall",
        "reward_term_abs_speed_error",
        "route_progress_available",
        "route_progress_fraction",
        "route_progress_monotonic_fraction",
        "route_completion_proxy",
        "route_deviation_m",
        "route_progress_rate_mps",
        "abs_speed_error",
        "low_speed_not_planned",
        "progress_stall",
        "negative_progress",
    )
    write_csv(path, fields, [
        {
            "reward_total": 0.5,
            "reward_comfort": -0.1,
            "reward_stability": -0.2,
            "reward_task": -0.3,
            "reward_action": -0.01,
            "reward_safety": 0.0,
            "reward_cost_comfort": 0.1,
            "reward_cost_stability": 0.2,
            "reward_cost_task": 0.3,
            "reward_cost_action": 0.04,
            "reward_cost_safety": 0.0,
            "reward_term_action_mag": 0.0,
            "reward_term_action_rate": 0.0,
            "reward_term_damper_rate": 0.01,
            "reward_term_baseline_dev": 0.02,
            "reward_term_low_speed_not_planned": 0.0,
            "reward_term_progress_stall": 0.0,
            "reward_term_abs_speed_error": 0.05,
            "route_progress_available": 1,
            "route_progress_fraction": 0.2,
            "route_progress_monotonic_fraction": 0.2,
            "route_completion_proxy": 0.2,
            "route_deviation_m": 0.5,
            "route_progress_rate_mps": 2.0,
            "abs_speed_error": 0.5,
            "low_speed_not_planned": 0,
            "progress_stall": 0,
            "negative_progress": 0,
        },
        {
            "reward_total": -0.5,
            "reward_comfort": -0.2,
            "reward_stability": -0.1,
            "reward_task": -0.8,
            "reward_action": -0.04,
            "reward_safety": -0.1,
            "reward_cost_comfort": 0.2,
            "reward_cost_stability": 0.1,
            "reward_cost_task": 0.8,
            "reward_cost_action": 0.16,
            "reward_cost_safety": 0.025,
            "reward_term_action_mag": 0.2,
            "reward_term_action_rate": 0.3,
            "reward_term_damper_rate": 0.04,
            "reward_term_baseline_dev": 0.08,
            "reward_term_low_speed_not_planned": 0.7,
            "reward_term_progress_stall": 0.5,
            "reward_term_abs_speed_error": 0.15,
            "route_progress_available": 1,
            "route_progress_fraction": 0.4,
            "route_progress_monotonic_fraction": 0.45,
            "route_completion_proxy": 0.4,
            "route_deviation_m": 1.5,
            "route_progress_rate_mps": -0.5,
            "abs_speed_error": 1.5,
            "low_speed_not_planned": 1,
            "progress_stall": 1,
            "negative_progress": 1,
        },
        {
            "route_progress_available": 0,
            "route_progress_fraction": 0.0,
            "route_progress_monotonic_fraction": 0.0,
        },
    ])

    summary = summarize_diagnostics_csv(str(path))

    assert summary["diagnostic_rows"] == 3
    assert summary["reward_rows"] == 2
    assert summary["reward_row_ratio"] == 2.0 / 3.0
    assert summary["mean_reward_total"] == 0.0
    assert summary["mean_reward_cost_task"] == 0.55
    assert summary["mean_reward_term_action_rate"] == 0.15
    assert summary["mean_reward_term_baseline_dev"] == 0.05
    assert summary["mean_reward_term_low_speed_not_planned"] == 0.35
    assert summary["mean_reward_term_progress_stall"] == 0.25
    assert summary["mean_reward_term_abs_speed_error"] == 0.1
    assert summary["route_progress_available_rows"] == 2
    assert summary["route_progress_available_ratio"] == 2.0 / 3.0
    assert summary["route_progress_fraction_max"] == 0.4
    assert summary["route_progress_monotonic_fraction_max"] == 0.45
    assert summary["route_completion_proxy"] == 0.4
    assert summary["route_deviation_m_mean"] == 1.0
    assert summary["route_deviation_m_max"] == 1.5
    assert summary["low_speed_not_planned_rows"] == 1
    assert summary["progress_stall_rows"] == 1
    assert summary["negative_progress_rows"] == 1
    assert summary["mean_route_progress_rate_mps"] == 0.75
    assert summary["mean_abs_speed_error"] == 1.0


def test_phase3_diagnostic_fields_include_reward_task_and_progress_columns():
    for field in (
        "reward_mode",
        "reward_term_abs_speed_error",
        "reward_term_progress_stall",
        "reward_route_deviation_used",
        "reward_route_deviation_enabled",
        "reward_route_deviation_valid",
        "route_progress_available",
        "route_progress_raw_m",
        "route_progress_delta_m",
        "route_progress_raw_negative_delta_m",
        "route_progress_negative_raw",
        "route_progress_raw_deviation_m",
        "route_progress_deviation_valid",
        "route_progress_monotonic_fraction",
        "route_progress_nearest_segment_index",
        "route_progress_tracker_status",
        "target_speed_source",
        "planned_stop",
        "low_speed_not_planned_severity",
        "route_progress_stall",
        "progress_stall",
        "progress_stall_count",
        "negative_progress",
    ):
        assert field in DIAGNOSTIC_FIELDS


def test_baseline_command_for_reward_uses_rl_baseline_diagnostics():
    command = SuspensionCommand.uniform(damper_scale=1.2)
    baseline, source = baseline_command_for_reward(command, {
        "rl_policy_available": 1,
        "rl_baseline_damper_fl": 0.9,
        "rl_baseline_damper_fr": 0.95,
        "rl_baseline_damper_rl": 1.0,
        "rl_baseline_damper_rr": 1.05,
    })

    assert source == "rl_baseline_diagnostics"
    assert [wheel.damper_scale for wheel in baseline.wheels] == [
        0.9,
        0.95,
        1.0,
        1.05,
    ]


def test_route_progress_id_from_subset_uses_first_route_id():
    assert route_progress_id_from_subset("00") == "00"
    assert route_progress_id_from_subset("0-2") == "0"
    assert route_progress_id_from_subset("01,02") == "01"
    assert route_progress_id_from_subset("") == ""
