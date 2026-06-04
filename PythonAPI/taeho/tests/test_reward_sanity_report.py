from suspension_control.rl.reward_sanity import (
    RANDOM_ACTION_SCENARIO,
    REFERENCE_SCENARIO,
    phase3_reward_sanity_rows,
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
        "mean_reward_total": 0.0,
        "mean_reward_action": 0.0,
        "mean_reward_task": 0.0,
        "mean_reward_cost_action": 0.0,
        "mean_reward_cost_task": 0.0,
        "mean_reward_term_action_mag": 0.0,
        "mean_reward_term_action_rate": 0.0,
        "mean_reward_term_baseline_dev": 0.0,
        "mean_reward_term_low_speed_not_planned": 0.0,
        "mean_reward_term_progress_stall": 0.0,
        "low_speed_not_planned_ratio": 0.0,
        "progress_stall_ratio": 0.0,
        "duration_game": 100.0,
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
        "mean_reward_cost_action": 0.0,
        "mean_reward_term_action_rate": 0.0,
        "mean_reward_term_baseline_dev": 0.0,
    })
    s12 = base_row("S12_rl_const_action_plus_0p25_skyhook")
    s12.update({
        "mean_reward_total": 0.7,
        "mean_reward_cost_action": 0.3,
        "mean_reward_term_action_mag": 0.05,
        "mean_reward_term_action_rate": 0.01,
        "mean_reward_term_baseline_dev": 0.04,
    })
    s13 = base_row("S13_rl_const_action_minus_0p25_skyhook")
    s13.update({
        "mean_reward_total": 0.7,
        "mean_reward_cost_action": 0.32,
        "mean_reward_term_action_mag": 0.05,
        "mean_reward_term_action_rate": 0.012,
        "mean_reward_term_baseline_dev": 0.045,
    })
    s14 = base_row(RANDOM_ACTION_SCENARIO)
    s14.update({
        "mean_reward_total": 0.6,
        "mean_reward_cost_action": 0.25,
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


def test_valid_phase3_reward_sanity_suite_passes():
    rows = phase3_reward_sanity_rows(valid_rows())

    assert len(rows) == 4
    assert all(row["phase3_status"] == "pass" for row in rows)
    assert by_scenario(rows, REFERENCE_SCENARIO)["zero_action_penalty_ok"] == 1
    assert by_scenario(
        rows,
        "S12_rl_const_action_plus_0p25_skyhook")["constant_action_penalty_ok"] == 1
    assert by_scenario(rows, RANDOM_ACTION_SCENARIO)[
        "random_action_rate_penalty_ok"] == 1


def test_missing_reward_fields_fail_phase3_sanity():
    rows = valid_rows()
    rows[1]["reward_row_ratio"] = ""

    report = phase3_reward_sanity_rows(rows)
    s12 = by_scenario(report, "S12_rl_const_action_plus_0p25_skyhook")

    assert s12["phase3_status"] == "fail"
    assert "reward_row_ratio" in s12["failed_checks"]


def test_random_action_rate_must_exceed_constants():
    rows = valid_rows()
    rows[3]["mean_reward_term_action_rate"] = 0.011

    report = phase3_reward_sanity_rows(rows)
    s14 = by_scenario(report, RANDOM_ACTION_SCENARIO)

    assert s14["phase3_status"] == "fail"
    assert "random_action_rate_penalty" in s14["failed_checks"]


def test_total_reward_warning_does_not_fail_by_itself():
    rows = valid_rows()
    rows[3]["mean_reward_total"] = 1.2

    report = phase3_reward_sanity_rows(rows)
    s14 = by_scenario(report, RANDOM_ACTION_SCENARIO)

    assert s14["phase3_status"] == "pass"
    assert "random_total_reward_above_reference" in s14["warnings"]


def test_slow_run_without_task_penalty_warns_only():
    rows = valid_rows()
    rows[1]["duration_game"] = 120.0
    rows[1]["mean_reward_cost_task"] = 0.0

    report = phase3_reward_sanity_rows(rows)
    s12 = by_scenario(report, "S12_rl_const_action_plus_0p25_skyhook")

    assert s12["phase3_status"] == "pass"
    assert "slow_run_not_task_penalized" in s12["warnings"]
