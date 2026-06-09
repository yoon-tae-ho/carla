from suspension_control.rl.phase4_acceptance import (
    phase4b_policy_eval_acceptance_rows,
)


def test_phase4d_score_penalty_one_with_zero_infractions_passes():
    rows = phase4b_policy_eval_acceptance_rows([
        _reference_row(),
        _candidate_row(),
    ])

    assert rows[0]["infraction_ok"] == 1
    assert rows[1]["infraction_ok"] == 1
    assert rows[1]["phase4b_status"] == "pass"


def test_phase4d_empty_lists_and_missing_optional_infraction_fields_are_zero():
    candidate = _candidate_row(
        collision_count="",
        lane_invasion_count="0.0",
        blocked_vehicle_count=None,
        collisions_layout=[],
        collisions_pedestrian=[],
        collisions_vehicle=[],
        red_light=[],
        stop_infraction=[],
        outside_route_lanes=[],
        infractions={
            "collisions_layout": [],
            "collisions_pedestrian": 0,
            "collisions_vehicle": "0.0",
            "red_light": "",
            "stop_infraction": {},
        },
    )

    rows = phase4b_policy_eval_acceptance_rows([
        _reference_row(),
        candidate,
    ])

    assert rows[1]["infraction_ok"] == 1
    assert "infractions" not in rows[1]["failed_checks"]


def test_phase4d_positive_scalar_infraction_fails():
    rows = phase4b_policy_eval_acceptance_rows([
        _reference_row(),
        _candidate_row(collision_count="1"),
    ])

    assert rows[1]["infraction_ok"] == 0
    assert "infraction" in rows[1]["failed_checks"]


def test_phase4d_nonempty_infraction_list_fails():
    rows = phase4b_policy_eval_acceptance_rows([
        _reference_row(),
        _candidate_row(infractions={"collisions_vehicle": ["collision"]}),
    ])

    assert rows[1]["infraction_ok"] == 0
    assert "infraction" in rows[1]["failed_checks"]


def test_phase4d_nested_leaderboard_infraction_value_fails():
    rows = phase4b_policy_eval_acceptance_rows([
        _reference_row(),
        _candidate_row(infractions={"red_light": {"count": 1}}),
    ])

    assert rows[1]["infraction_ok"] == 0
    assert "infraction" in rows[1]["failed_checks"]


def _reference_row(**overrides):
    row = _base_row("S8_rl_zero_residual_skyhook")
    row.update({
        "rl_mean_abs_action": "0.0",
        "rl_mean_abs_residual_damper": "0.0",
        "mean_abs_action": "0.0",
        "mean_abs_residual_damper": "0.0",
        "effective_control_ratio": "0.0",
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
        "stop_infraction_count": "0",
        "infractions": {
            "collisions_layout": 0.0,
            "collisions_pedestrian": 0.0,
            "collisions_vehicle": 0.0,
            "red_light": 0.0,
            "stop_infraction": 0.0,
            "outside_route_lanes": 0,
            "route_dev": 0.0,
            "vehicle_blocked": 0.0,
            "scenario_timeouts": 0.0,
            "route_timeout": 0.0,
        },
    }
