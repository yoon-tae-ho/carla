import math

from suspension_control.controllers.base import SuspensionCommand, VehicleState
from suspension_control.rl.reward import (
    RewardTransition,
    SuspensionReward,
    SuspensionRewardConfig,
)


def command(damper=1.0):
    return SuspensionCommand.uniform(damper_scale=damper)


def transition(**kwargs):
    values = {
        "state": VehicleState(speed=8.0, az=0.1, local_ay=0.2, local_ax=0.1),
        "previous_state": VehicleState(speed=8.0, az=0.0, local_ay=0.0, local_ax=0.0),
        "action": (0.0, 0.0, 0.0, 0.0),
        "previous_action": (0.0, 0.0, 0.0, 0.0),
        "final_command": command(),
        "baseline_command": command(),
        "task_info": {},
        "diagnostics": {},
        "dt": 0.05,
    }
    values.update(kwargs)
    return RewardTransition(**values)


def test_reward_returns_total_and_groups():
    reward, diagnostics = SuspensionReward().compute(transition())
    assert math.isfinite(reward)
    for key in (
        "reward_total",
        "reward_comfort",
        "reward_stability",
        "reward_task",
        "reward_action",
        "reward_safety",
    ):
        assert key in diagnostics


def test_worse_vertical_acceleration_and_jerk_lower_reward():
    reward = SuspensionReward()
    mild, _ = reward.compute(transition())
    harsh, _ = reward.compute(transition(
        state=VehicleState(speed=8.0, az=10.0, local_ay=0.2, local_ax=0.1),
        previous_state=VehicleState(speed=8.0, az=-10.0, local_ay=0.0, local_ax=0.0)))
    assert harsh < mild


def test_route_deviation_disabled_yields_zero_route_deviation_term():
    reward = SuspensionReward()
    ok, ok_diag = reward.compute(transition(task_info={
        "route_deviation_m": 0.0,
        "route_deviation_valid": 1.0,
    }))
    bad, bad_diag = reward.compute(transition(task_info={
        "route_deviation_m": 5.0,
        "route_deviation_valid": 1.0,
    }))

    assert bad_diag["reward_term_route_deviation"] == 0.0
    assert bad_diag["reward_route_deviation_enabled"] == 0.0
    assert bad_diag["reward_route_deviation_used"] == 0.0
    assert math.isclose(bad_diag["reward_task"], ok_diag["reward_task"])
    assert math.isclose(bad, ok)


def test_route_deviation_enabled_requires_valid_deviation():
    reward = SuspensionReward(SuspensionRewardConfig(
        reward_route_deviation_enabled=True,
        t_route_deviation=1.0))

    _, valid_diag = reward.compute(transition(task_info={
        "route_deviation_m": 4.0,
        "route_deviation_valid": 1.0,
    }))
    _, invalid_diag = reward.compute(transition(task_info={
        "route_deviation_m": 4.0,
        "route_deviation_valid": 0.0,
    }))
    _, too_far_diag = reward.compute(transition(task_info={
        "route_deviation_m": 9.0,
        "route_deviation_valid": 1.0,
    }))

    assert valid_diag["reward_term_route_deviation"] > 0.0
    assert valid_diag["reward_route_deviation_used"] == 1.0
    assert invalid_diag["reward_term_route_deviation"] == 0.0
    assert invalid_diag["reward_route_deviation_used"] == 0.0
    assert too_far_diag["reward_term_route_deviation"] == 0.0
    assert too_far_diag["reward_route_deviation_valid"] == 0.0


def test_task_reward_scale_without_infraction_is_small():
    _, diagnostics = SuspensionReward().compute(transition(task_info={
        "target_speed": 8.0,
        "abs_speed_error": 100.0,
        "route_progress_delta_m": 0.0,
        "low_speed_not_planned": 1.0,
        "low_speed_not_planned_severity": 1.0,
        "progress_stall": 1.0,
        "route_deviation_m": 100.0,
        "route_deviation_valid": 1.0,
        "collision_count": 0.0,
        "lane_invasion_count": 0.0,
        "red_light_count": 0.0,
        "blocked_vehicle": 0.0,
        "route_timeout": 0.0,
    }))

    assert diagnostics["reward_cost_task"] <= 2.0
    assert diagnostics["reward_task"] >= -2.0
    assert diagnostics["reward_task_capped_without_infraction"] == 1.0


def test_action_rate_penalty_increases_for_sharp_change():
    reward = SuspensionReward()
    smooth, smooth_diag = reward.compute(transition(
        action=(0.1, 0.1, 0.1, 0.1),
        previous_action=(0.1, 0.1, 0.1, 0.1)))
    sharp, sharp_diag = reward.compute(transition(
        action=(1.0, -1.0, 1.0, -1.0),
        previous_action=(-1.0, 1.0, -1.0, 1.0)))
    assert sharp < smooth
    assert sharp_diag["reward_term_action_rate"] > smooth_diag["reward_term_action_rate"]


def test_constant_action_has_larger_action_penalty_than_zero():
    reward = SuspensionReward()

    _, zero_diag = reward.compute(transition())
    _, const_diag = reward.compute(transition(
        action=(0.2, 0.2, 0.2, 0.2),
        previous_action=(0.2, 0.2, 0.2, 0.2)))

    assert const_diag["reward_cost_action"] > zero_diag["reward_cost_action"]
    assert const_diag["reward_term_action_mag"] > zero_diag["reward_term_action_mag"]


def test_random_action_rate_penalty_exceeds_constant():
    reward = SuspensionReward()

    _, const_diag = reward.compute(transition(
        action=(0.2, 0.2, 0.2, 0.2),
        previous_action=(0.2, 0.2, 0.2, 0.2)))
    _, random_diag = reward.compute(transition(
        action=(0.8, -0.7, 0.6, -0.5),
        previous_action=(-0.8, 0.7, -0.6, 0.5)))

    assert random_diag["reward_term_action_rate"] > (
        const_diag["reward_term_action_rate"])


def test_missing_task_signals_do_not_create_nan():
    reward, diagnostics = SuspensionReward().compute(transition(task_info=None))
    assert math.isfinite(reward)
    assert all(math.isfinite(value) for value in diagnostics.values())
