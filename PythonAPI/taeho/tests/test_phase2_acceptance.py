from transfuser_suspension_control_suite import (
    phase2_acceptance_rows,
    phase2c_acceptance_rows,
    selected_scenarios,
    summarize_diagnostics_csv,
)


def base_row(scenario, controller, uses_api=1):
    return {
        "seed": 100,
        "scenario": scenario,
        "label": scenario,
        "controller": controller,
        "uses_suspension_api": uses_api,
        "score_route": 100.0,
        "score_composed": 100.0,
        "score_penalty": 1.0,
        "sidecar_command_verifies": 2 if uses_api else 0,
        "collisions_layout": 0,
        "collisions_pedestrian": 0,
        "collisions_vehicle": 0,
        "red_light": 0,
        "stop_infraction": 0,
        "outside_route_lanes": 0,
        "route_dev": 0,
        "vehicle_blocked": 0,
        "route_timeout": 0,
        "scenario_timeouts": 0,
        "min_speed_infractions": 0,
    }


def zero_row(scenario, controller):
    row = base_row(scenario, controller)
    row.update({
        "rl_diagnostic_rows": 100,
        "rl_policy_available_rows": 100,
        "rl_policy_available_ratio": 1.0,
        "rl_fallback_rows": 0,
        "rl_fallback_reasons": "",
        "rl_mean_action": 0.0,
        "rl_mean_abs_action": 0.0,
        "rl_nonzero_action_rows": 0,
        "rl_mean_residual_damper": 0.0,
        "rl_mean_abs_residual_damper": 0.0,
        "rl_nonzero_residual_rows": 0,
    })
    return row


def canary_row(scenario, controller, mean_action, mean_abs_action):
    row = base_row(scenario, controller)
    row.update({
        "rl_diagnostic_rows": 100,
        "rl_policy_available_rows": 100,
        "rl_policy_available_ratio": 1.0,
        "rl_fallback_rows": 20,
        "rl_fallback_reasons": "safety_gate_zero",
        "rl_mean_action": mean_action,
        "rl_mean_abs_action": mean_abs_action,
        "rl_nonzero_action_rows": 100,
        "rl_mean_residual_damper": (
            0.001 if mean_action >= 0 else -0.001),
        "rl_mean_abs_residual_damper": 0.001,
        "rl_nonzero_residual_rows": 80,
    })
    return row


def test_phase2_scenario_keys_are_available():
    scenarios = selected_scenarios(
        "stock,identity,pid,skyhook,"
        "rl_zero_residual_pid,rl_zero_residual_skyhook,"
        "rl_const_plus_0p02_skyhook,"
        "rl_const_minus_0p02_skyhook,"
        "rl_random_small_skyhook,"
        "rl_const_action_plus_0p25_skyhook,"
        "rl_const_action_minus_0p25_skyhook,"
        "rl_random_action_0p10_skyhook")
    assert [scenario["controller"] for scenario in scenarios] == [
        "stock",
        "identity",
        "pid",
        "skyhook",
        "rl_zero_residual_pid",
        "rl_zero_residual_skyhook",
        "rl_const_plus_0p02_skyhook",
        "rl_const_minus_0p02_skyhook",
        "rl_random_small_skyhook",
        "rl_const_action_plus_0p25_skyhook",
        "rl_const_action_minus_0p25_skyhook",
        "rl_random_action_0p10_skyhook",
    ]


def test_phase2_scenario_code_selection_handles_s10_without_s1_collision():
    scenarios = selected_scenarios("S8,S9,S10,S11,S12,S13,S14")
    assert [scenario["controller"] for scenario in scenarios] == [
        "rl_zero_residual_skyhook",
        "rl_const_plus_0p02_skyhook",
        "rl_const_minus_0p02_skyhook",
        "rl_random_small_skyhook",
        "rl_const_action_plus_0p25_skyhook",
        "rl_const_action_minus_0p25_skyhook",
        "rl_random_action_0p10_skyhook",
    ]


def test_phase2_acceptance_passes_for_clean_dummy_zero_suite():
    rows = [
        base_row("S0_stock", "stock", uses_api=0),
        base_row("S1_identity", "identity"),
        base_row("S2_pid", "pid"),
        base_row("S3_skyhook", "skyhook"),
        zero_row("S7_rl_zero_residual_pid", "rl_zero_residual_pid"),
        zero_row("S8_rl_zero_residual_skyhook", "rl_zero_residual_skyhook"),
    ]

    acceptance = phase2_acceptance_rows(rows)

    assert all(row["phase2_status"] == "pass" for row in acceptance)
    zero_acceptance = [
        row for row in acceptance
        if row["phase2_role"] == "rl_zero_residual"
    ]
    assert len(zero_acceptance) == 2
    assert all(row["policy_available_ok"] == 1 for row in zero_acceptance)
    assert all(row["fallback_ok"] == 1 for row in zero_acceptance)
    assert all(row["zero_action_ok"] == 1 for row in zero_acceptance)
    assert all(row["zero_residual_ok"] == 1 for row in zero_acceptance)


def test_phase2c_acceptance_passes_for_canary_suite():
    rows = [
        zero_row("S8_rl_zero_residual_skyhook", "rl_zero_residual_skyhook"),
        canary_row(
            "S9_rl_const_plus_0p02_skyhook",
            "rl_const_plus_0p02_skyhook",
            0.02,
            0.02),
        canary_row(
            "S10_rl_const_minus_0p02_skyhook",
            "rl_const_minus_0p02_skyhook",
            -0.02,
            0.02),
        canary_row(
            "S11_rl_random_small_skyhook",
            "rl_random_small_skyhook",
            0.0004,
            0.010),
        canary_row(
            "S12_rl_const_action_plus_0p25_skyhook",
            "rl_const_action_plus_0p25_skyhook",
            0.25,
            0.25),
        canary_row(
            "S13_rl_const_action_minus_0p25_skyhook",
            "rl_const_action_minus_0p25_skyhook",
            -0.25,
            0.25),
        canary_row(
            "S14_rl_random_action_0p10_skyhook",
            "rl_random_action_0p10_skyhook",
            0.001,
            0.05),
    ]

    acceptance = phase2c_acceptance_rows(rows)

    assert len(acceptance) == 7
    assert all(row["phase2c_status"] == "pass" for row in acceptance)
    assert all(row["policy_available_ok"] == 1 for row in acceptance)
    assert all(row["fallback_reason_ok"] == 1 for row in acceptance)
    assert all(row["action_path_ok"] == 1 for row in acceptance)


def test_phase2_acceptance_fails_for_fallback_rows():
    rows = [
        base_row("S2_pid", "pid"),
        zero_row("S7_rl_zero_residual_pid", "rl_zero_residual_pid"),
    ]
    rows[-1]["rl_fallback_rows"] = 1

    acceptance = phase2_acceptance_rows(rows)
    zero_acceptance = acceptance[-1]

    assert zero_acceptance["phase2_status"] == "fail"
    assert "fallback_rows" in zero_acceptance["failed_checks"]


def test_phase3_only_zero_residual_does_not_fail_for_missing_plain_baseline():
    rows = [
        zero_row("S8_rl_zero_residual_skyhook", "rl_zero_residual_skyhook"),
    ]

    acceptance = phase2_acceptance_rows(rows)
    zero_acceptance = acceptance[0]

    assert zero_acceptance["phase2_status"] == "pass"
    assert zero_acceptance["paired_baseline_found"] == 0
    assert zero_acceptance["pair_score_match_ok"] == "not_applicable"
    assert "paired_baseline" not in zero_acceptance["failed_checks"]


def test_phase2_acceptance_remains_strict_when_paired_baseline_is_present():
    rows = [
        base_row("S3_skyhook", "skyhook"),
        zero_row("S8_rl_zero_residual_skyhook", "rl_zero_residual_skyhook"),
    ]
    rows[0]["score_penalty"] = 0.5

    acceptance = phase2_acceptance_rows(rows)
    zero_acceptance = acceptance[-1]

    assert zero_acceptance["paired_baseline_found"] == 1
    assert zero_acceptance["pair_score_match_ok"] == 0
    assert zero_acceptance["phase2_status"] == "fail"
    assert "paired_score" in zero_acceptance["failed_checks"]


def test_diagnostics_summary_counts_safety_gate_reasons(tmp_path):
    path = tmp_path / "controller_diagnostics.csv"
    fields = (
        "rl_policy_available",
        "rl_fallback_reason",
        "rl_mean_abs_action",
        "rl_mean_abs_residual_damper",
        "rl_observation_clip_count",
        "rl_safety_gain",
        "rl_safety_gate_active",
        "rl_safety_gate_reason",
        "rl_safety_gate_speed_limit",
        "rl_safety_gate_nonfinite_obs",
        "rl_safety_gate_action_invalid",
        "reward_total",
        "reward_comfort",
        "reward_stability",
        "reward_task",
        "reward_action",
        "reward_safety",
        "route_progress_fraction",
        "delta_progress",
        "target_speed",
        "target_speed_error",
        "route_deviation",
        "collision_count",
        "lane_invasion_count",
        "red_light_count",
        "blocked_vehicle",
        "route_timeout",
        "speed",
        "roll",
        "local_ay",
        "yaw_rate",
    )
    with open(path, "w") as csv_file:
        csv_file.write(",".join(fields) + "\n")
        csv_file.write(
            "1,,0,0,0,1,0,,0,0,0,1.0,-0.1,-0.1,-0.1,-0.1,-0.1,0.10,0.01,8.0,0.1,0.0,0,0,0,0,0,8.0,1.0,0.5,2.0\n")
        csv_file.write(
            "1,,0.02,0.001,0,0.5,1,roll_limit;yaw_rate_limit,0,0,0,0.8,-0.2,-0.1,-0.3,-0.1,-0.2,0.20,0.01,8.0,0.2,0.1,0,0,0,0,0,4.0,7.0,2.0,70.0\n")
        csv_file.write(
            "1,safety_gate_zero,0.02,0,0,0,1,speed_below_min,1,0,0,0.7,-0.2,-0.1,-0.4,-0.1,-0.2,0.21,0.01,8.0,0.2,0.1,0,0,0,0,0,0.1,0.1,0.1,0.1\n")
        csv_file.write(
            "1,invalid_observation,0,0,2,0,1,nonfinite_obs,0,1,0,0.6,-0.3,-0.1,-0.5,-0.1,-0.2,0.22,0.01,8.0,0.2,0.1,1,0,0,0,0,8.0,0.0,0.0,0.0\n")

    summary = summarize_diagnostics_csv(str(path))

    assert summary["safety_gate_active_rows"] == 3
    assert summary["safety_gate_active_ratio"] == 0.75
    assert summary["safety_gate_reason_counts"] == (
        "nonfinite_obs=1;roll_limit=1;speed_below_min=1;yaw_rate_limit=1")
    assert summary["low_speed_mask_rows"] == 1
    assert summary["low_speed_mask_ratio"] == 0.25
    assert summary["hard_safety_gate_rows"] == 1
    assert summary["hard_safety_gate_ratio"] == 0.25
    assert summary["soft_safety_gain_rows"] == 1
    assert summary["soft_safety_gain_ratio"] == 0.25
    assert summary["effective_control_ratio"] == 0.25
    assert summary["fallback_ratio"] == 0.5
    assert summary["observation_clip_ratio"] == 0.25
    assert summary["mean_abs_action"] == summary["rl_mean_abs_action"]
    assert summary["mean_abs_residual_damper"] == (
        summary["rl_mean_abs_residual_damper"])
    assert summary["mean_reward_total"] == 0.775
    assert summary["route_progress_fraction_max"] == 0.22
    assert summary["route_completion_proxy"] == 0
    assert summary["collision_count"] == 1.0
    assert summary["safety_gate_active_roll_max"] == 7.0
    assert summary["safety_gate_active_yaw_rate_max"] == 70.0
