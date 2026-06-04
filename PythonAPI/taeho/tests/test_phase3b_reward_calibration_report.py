from suspension_control.rl.reward_calibration import (
    RANDOM_ACTION_SCENARIO,
    REFERENCE_SCENARIO,
    phase3b_reward_calibration_rows,
)


def base_row(scenario):
    return {
        "seed": 100,
        "scenario": scenario,
        "score_route": 100.0,
        "score_composed": 100.0,
        "score_penalty": 1.0,
        "sidecar_command_verifies": 3,
        "rl_policy_available_ratio": 1.0,
        "reward_row_ratio": 1.0,
        "route_progress_available_ratio": 1.0,
        "route_progress_monotonic_fraction_max": 1.0,
        "route_progress_monotonic_negative_rows": 0,
        "route_progress_raw_negative_ratio": 0.0,
        "route_progress_stall_ratio": 0.0,
        "low_speed_not_planned_ratio": 0.0,
        "mean_reward_total": 0.0,
        "mean_reward_task": -0.2,
        "mean_reward_action": 0.0,
        "mean_reward_cost_task": 0.2,
        "mean_reward_term_action_mag": 0.0,
        "mean_reward_term_action_rate": 0.0,
        "mean_reward_term_baseline_dev": 0.0,
        "mean_reward_term_route_deviation": 0.0,
        "mean_reward_route_deviation_used_ratio": 0.0,
        "collision_count": 0,
        "lane_invasion_count": 0,
        "red_light_count": 0,
        "blocked_vehicle_count": 0,
        "route_timeout_count": 0,
    }


def valid_rows():
    s8 = base_row(REFERENCE_SCENARIO)
    s8.update({
        "mean_reward_total": 1.0,
        "mean_reward_term_action_mag": 0.0,
        "mean_reward_term_action_rate": 0.0,
        "mean_reward_term_baseline_dev": 0.0,
    })
    s12 = base_row("S12_rl_const_action_plus_0p25_skyhook")
    s12.update({
        "mean_reward_total": 0.7,
        "mean_reward_action": -0.08,
        "mean_reward_term_action_mag": 0.05,
        "mean_reward_term_action_rate": 0.01,
        "mean_reward_term_baseline_dev": 0.04,
    })
    s13 = base_row("S13_rl_const_action_minus_0p25_skyhook")
    s13.update({
        "mean_reward_total": 0.7,
        "mean_reward_action": -0.09,
        "mean_reward_term_action_mag": 0.05,
        "mean_reward_term_action_rate": 0.012,
        "mean_reward_term_baseline_dev": 0.045,
    })
    s14 = base_row(RANDOM_ACTION_SCENARIO)
    s14.update({
        "mean_reward_total": 0.6,
        "mean_reward_action": -0.1,
        "mean_reward_term_action_mag": 0.03,
        "mean_reward_term_action_rate": 0.08,
        "mean_reward_term_baseline_dev": 0.03,
    })
    return [s8, s12, s13, s14]


def by_scenario(rows, scenario):
    for row in rows:
        if row["scenario"] == scenario:
            return row
    raise AssertionError("missing scenario %s" % scenario)


def test_phase3b_acceptance_passes_rebalanced_sanity_rows():
    rows = phase3b_reward_calibration_rows(valid_rows())

    assert len(rows) == 4
    assert all(row["phase3b_status"] == "pass" for row in rows)
    assert by_scenario(rows, REFERENCE_SCENARIO)["zero_action_penalty_ok"] == 1
    assert by_scenario(
        rows,
        "S12_rl_const_action_plus_0p25_skyhook")["constant_action_penalty_ok"] == 1
    assert by_scenario(rows, RANDOM_ACTION_SCENARIO)[
        "random_action_rate_penalty_ok"] == 1


def test_phase3b_acceptance_flags_random_above_reference():
    rows = valid_rows()
    rows[3]["mean_reward_total"] = 1.05

    report = phase3b_reward_calibration_rows(rows)
    random_row = by_scenario(report, RANDOM_ACTION_SCENARIO)

    assert random_row["phase3b_status"] == "pass"
    assert random_row["random_total_reward_not_above_reference_ok"] == 0
    assert abs(random_row["reward_random_total_delta_vs_reference"] - 0.05) < 1.0e-12
    assert "random_total_reward_above_reference" in random_row["warnings"]
    assert "random_total_reward" not in random_row["failed_checks"]


def test_phase3b_fails_task_reward_scale_without_infraction():
    rows = valid_rows()
    rows[1]["mean_reward_task"] = -2.5

    report = phase3b_reward_calibration_rows(rows)
    s12 = by_scenario(report, "S12_rl_const_action_plus_0p25_skyhook")

    assert s12["phase3b_status"] == "fail"
    assert "task_reward_scale" in s12["failed_checks"]


def test_phase3b_fails_route_deviation_used_when_disabled():
    rows = valid_rows()
    rows[1]["mean_reward_route_deviation_used_ratio"] = 0.1

    report = phase3b_reward_calibration_rows(rows)
    s12 = by_scenario(report, "S12_rl_const_action_plus_0p25_skyhook")

    assert s12["phase3b_status"] == "fail"
    assert "route_deviation_not_used_when_disabled" in s12["failed_checks"]


def test_phase3b_fails_monotonic_negative_rows():
    rows = valid_rows()
    rows[2]["route_progress_monotonic_negative_rows"] = 1

    report = phase3b_reward_calibration_rows(rows)
    s13 = by_scenario(report, "S13_rl_const_action_minus_0p25_skyhook")

    assert s13["phase3b_status"] == "fail"
    assert "monotonic_progress" in s13["failed_checks"]
