from transfuser_suspension_control_suite import summarize_diagnostics_csv


FIELDS = (
    "reward_total",
    "reward_comfort",
    "reward_stability",
    "reward_task",
    "reward_action",
    "reward_safety",
    "reward_cost_task",
    "reward_term_abs_speed_error",
    "reward_term_low_speed_not_planned",
    "reward_term_progress_stall",
    "reward_term_route_deviation",
    "reward_route_deviation_used",
    "reward_route_deviation_enabled",
    "reward_route_deviation_valid",
    "route_progress_available",
    "route_progress_fraction",
    "route_progress_monotonic_fraction",
    "route_progress_delta_m",
    "route_progress_rate_mps",
    "route_progress_raw_m",
    "route_progress_raw_deviation_m",
    "route_progress_deviation_valid",
    "route_progress_negative_raw",
    "route_progress_raw_negative_delta_m",
    "route_delta_progress_m",
    "route_progress_stall",
    "progress_stall",
    "progress_stall_count",
    "low_speed_not_planned",
    "planned_stop",
    "route_deviation_m",
    "negative_progress",
)


def write_csv(path, rows):
    with open(path, "w") as csv_file:
        csv_file.write(",".join(FIELDS) + "\n")
        for row in rows:
            csv_file.write(",".join(str(row.get(field, "")) for field in FIELDS) + "\n")


def phase3b_rows():
    return [
        {
            "reward_total": 0.5,
            "reward_comfort": -0.1,
            "reward_stability": -0.2,
            "reward_task": -0.3,
            "reward_action": -0.01,
            "reward_safety": 0.0,
            "reward_cost_task": 0.3,
            "reward_term_abs_speed_error": 0.05,
            "reward_term_low_speed_not_planned": 0.0,
            "reward_term_progress_stall": 0.0,
            "reward_term_route_deviation": 0.0,
            "reward_route_deviation_used": 0,
            "reward_route_deviation_enabled": 0,
            "reward_route_deviation_valid": 1,
            "route_progress_available": 1,
            "route_progress_fraction": 0.2,
            "route_progress_monotonic_fraction": 0.2,
            "route_progress_delta_m": 0.1,
            "route_progress_rate_mps": 2.0,
            "route_progress_raw_m": 10.0,
            "route_progress_raw_deviation_m": 0.5,
            "route_progress_deviation_valid": 1,
            "route_progress_negative_raw": 0,
            "route_progress_raw_negative_delta_m": 0.0,
            "route_delta_progress_m": 0.1,
            "route_progress_stall": 0,
            "progress_stall": 0,
            "progress_stall_count": 0,
            "low_speed_not_planned": 0,
            "planned_stop": 0,
            "route_deviation_m": 0.5,
            "negative_progress": 0,
        },
        {
            "reward_total": -0.5,
            "reward_comfort": -0.2,
            "reward_stability": -0.1,
            "reward_task": -0.8,
            "reward_action": -0.04,
            "reward_safety": -0.1,
            "reward_cost_task": 0.8,
            "reward_term_abs_speed_error": 0.15,
            "reward_term_low_speed_not_planned": 0.5,
            "reward_term_progress_stall": 0.5,
            "reward_term_route_deviation": 0.0,
            "reward_route_deviation_used": 0,
            "reward_route_deviation_enabled": 0,
            "reward_route_deviation_valid": 1,
            "route_progress_available": 1,
            "route_progress_fraction": 0.4,
            "route_progress_monotonic_fraction": 0.45,
            "route_progress_delta_m": 0.0,
            "route_progress_rate_mps": 0.0,
            "route_progress_raw_m": 9.8,
            "route_progress_raw_deviation_m": 1.5,
            "route_progress_deviation_valid": 1,
            "route_progress_negative_raw": 1,
            "route_progress_raw_negative_delta_m": -0.2,
            "route_delta_progress_m": -0.2,
            "route_progress_stall": 1,
            "progress_stall": 1,
            "progress_stall_count": 3,
            "low_speed_not_planned": 1,
            "planned_stop": 0,
            "route_deviation_m": 1.5,
            "negative_progress": 1,
        },
        {
            "route_progress_available": 0,
            "route_progress_fraction": 0.0,
            "route_progress_monotonic_fraction": 0.45,
            "route_progress_delta_m": -0.01,
            "route_progress_deviation_valid": 0,
        },
    ]


def summarize(tmp_path):
    path = tmp_path / "controller_diagnostics.csv"
    write_csv(path, phase3b_rows())
    return summarize_diagnostics_csv(str(path))


def test_summary_includes_phase3b_reward_fields(tmp_path):
    summary = summarize(tmp_path)

    assert summary["reward_rows"] == 2
    assert summary["mean_reward_cost_task"] == 0.55
    assert summary["mean_reward_term_abs_speed_error"] == 0.1
    assert summary["mean_reward_term_low_speed_not_planned"] == 0.25
    assert summary["mean_reward_term_progress_stall"] == 0.25
    assert summary["mean_reward_term_route_deviation"] == 0.0


def test_summary_includes_phase3b_route_progress_fields(tmp_path):
    summary = summarize(tmp_path)

    assert summary["route_progress_available_rows"] == 2
    assert summary["route_progress_available_ratio"] == 2.0 / 3.0
    assert summary["route_progress_fraction_max"] == 0.4
    assert summary["route_progress_monotonic_fraction_max"] == 0.45
    assert summary["route_progress_stall_rows"] == 1
    assert summary["route_progress_stall_ratio"] == 1.0 / 3.0
    assert summary["progress_stall_count_max"] == 3.0
    assert summary["route_deviation_valid_ratio"] == 2.0 / 3.0


def test_route_deviation_used_ratio_zero_when_disabled(tmp_path):
    summary = summarize(tmp_path)

    assert summary["reward_route_deviation_used_rows"] == 0
    assert summary["mean_reward_route_deviation_used_ratio"] == 0.0


def test_monotonic_negative_rows_detected(tmp_path):
    summary = summarize(tmp_path)

    assert summary["route_progress_monotonic_negative_rows"] == 1
    assert summary["route_progress_monotonic_negative_ratio"] == 1.0 / 3.0


def test_raw_negative_progress_ratio_is_diagnostic(tmp_path):
    summary = summarize(tmp_path)

    assert summary["route_progress_raw_negative_rows"] == 1
    assert summary["route_progress_raw_negative_ratio"] == 1.0 / 3.0
    assert summary["route_progress_delta_negative_raw_ratio"] == 1.0 / 3.0
    assert summary["negative_progress_ratio"] == 1.0 / 3.0


def test_reward_row_ratio_uses_diagnostic_rows_denominator(tmp_path):
    summary = summarize(tmp_path)

    assert summary["diagnostic_rows"] == 3
    assert summary["reward_rows"] == 2
    assert summary["reward_row_ratio"] == 2.0 / 3.0


def test_summary_includes_authority_sensitivity_residual_fields(tmp_path):
    fields = (
        "rl_policy_available",
        "rl_residual_mode",
        "rl_action_scale",
        "rl_residual_gain",
        "rl_scripted_residual_kind",
        "rl_mean_action",
        "rl_mean_abs_action",
        "rl_mean_raw_residual_damper",
        "rl_mean_abs_raw_residual_damper",
        "rl_mean_scaled_residual_damper",
        "rl_mean_abs_scaled_residual_damper",
        "rl_mean_scale_clipped_residual_damper",
        "rl_mean_abs_scale_clipped_residual_damper",
        "rl_mean_residual_damper",
        "rl_mean_abs_residual_damper",
        "rl_mean_final_residual_damper",
        "rl_mean_abs_final_residual_damper",
        "rl_residual_scale_clip",
        "rl_damper_final_clamp",
        "rl_residual_saturation",
    )
    rows = [
        {
            "rl_policy_available": 1,
            "rl_residual_mode": "learned_policy",
            "rl_action_scale": 5.0,
            "rl_residual_gain": 5.0,
            "rl_mean_action": 0.2,
            "rl_mean_abs_action": 0.2,
            "rl_mean_raw_residual_damper": 0.02,
            "rl_mean_abs_raw_residual_damper": 0.02,
            "rl_mean_scaled_residual_damper": 0.10,
            "rl_mean_abs_scaled_residual_damper": 0.10,
            "rl_mean_scale_clipped_residual_damper": 0.08,
            "rl_mean_abs_scale_clipped_residual_damper": 0.08,
            "rl_mean_residual_damper": 0.08,
            "rl_mean_abs_residual_damper": 0.08,
            "rl_mean_final_residual_damper": 0.08,
            "rl_mean_abs_final_residual_damper": 0.08,
            "rl_residual_scale_clip": 1,
            "rl_damper_final_clamp": 0,
            "rl_residual_saturation": 1,
        },
        {
            "rl_policy_available": 1,
            "rl_residual_mode": "scripted",
            "rl_action_scale": 1.0,
            "rl_residual_gain": 1.0,
            "rl_scripted_residual_kind": "const_m0p05",
            "rl_mean_action": 0.0,
            "rl_mean_abs_action": 0.05,
            "rl_mean_raw_residual_damper": -0.05,
            "rl_mean_abs_raw_residual_damper": 0.05,
            "rl_mean_scaled_residual_damper": -0.05,
            "rl_mean_abs_scaled_residual_damper": 0.05,
            "rl_mean_scale_clipped_residual_damper": -0.05,
            "rl_mean_abs_scale_clipped_residual_damper": 0.05,
            "rl_mean_residual_damper": -0.05,
            "rl_mean_abs_residual_damper": 0.05,
            "rl_mean_final_residual_damper": -0.05,
            "rl_mean_abs_final_residual_damper": 0.05,
            "rl_residual_scale_clip": 0,
            "rl_damper_final_clamp": 1,
            "rl_residual_saturation": 1,
        },
    ]
    path = tmp_path / "controller_diagnostics.csv"
    with open(path, "w") as csv_file:
        csv_file.write(",".join(fields) + "\n")
        for row in rows:
            csv_file.write(",".join(str(row.get(field, "")) for field in fields) + "\n")

    summary = summarize_diagnostics_csv(str(path))

    assert summary["rl_residual_modes"] == "learned_policy;scripted"
    assert summary["rl_scripted_residual_kinds"] == "const_m0p05"
    assert summary["rl_action_scale_mean"] == 3.0
    assert summary["rl_residual_gain_mean"] == 3.0
    assert abs(summary["raw_mean_abs_action"] - 0.125) < 1e-12
    assert abs(summary["raw_mean_abs_residual_damper"] - 0.035) < 1e-12
    assert abs(summary["scaled_mean_abs_residual_damper"] - 0.075) < 1e-12
    assert abs(summary["final_mean_abs_residual_damper"] - 0.065) < 1e-12
    assert summary["residual_scale_clip_rows"] == 1
    assert summary["residual_scale_clip_ratio"] == 0.5
    assert summary["damper_final_clamp_rows"] == 1
    assert summary["damper_final_clamp_ratio"] == 0.5
    assert summary["residual_saturation_rows"] == 2
    assert summary["residual_saturation_ratio"] == 1.0
