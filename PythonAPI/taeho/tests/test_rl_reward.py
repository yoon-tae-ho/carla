import math

from suspension_control.controllers.base import SuspensionCommand, VehicleState
from suspension_control.rl.reward import RewardTransition, SuspensionReward


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


def test_larger_route_penalty_lowers_reward():
    reward = SuspensionReward()
    ok, _ = reward.compute(transition(task_info={"route_deviation": 0.0}))
    bad, _ = reward.compute(transition(task_info={"route_deviation": 5.0}))
    assert bad < ok


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


def test_missing_task_signals_do_not_create_nan():
    reward, diagnostics = SuspensionReward().compute(transition(task_info=None))
    assert math.isfinite(reward)
    assert all(math.isfinite(value) for value in diagnostics.values())
