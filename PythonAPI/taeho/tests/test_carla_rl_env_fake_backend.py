import csv
import math

from suspension_control.controllers.base import SuspensionCommand, VehicleState
from suspension_control.rl.carla_backend import (
    FakeCarlaSuspensionBackend,
    LIVE_BACKEND_UNAVAILABLE_MESSAGE,
    LiveCarlaSuspensionBackend,
)
from suspension_control.rl.carla_online_env import CarlaSuspensionEnv
from suspension_control.rl.reward import RewardTransition, SuspensionReward
from suspension_control.rl.rollout_logger import RolloutLogger
from suspension_control.rl.task_info import DEFAULT_TASK_INFO, normalize_task_info


def finite_values(values):
    return [float(value) for value in list(values)]


def make_env(warmup_seconds=3.0):
    backend = FakeCarlaSuspensionBackend(max_steps=240)
    return CarlaSuspensionEnv(
        backend=backend,
        baseline="skyhook",
        planning_provider="empty",
        warmup_seconds=warmup_seconds,
        verify_suspension=True,
        route_id="00",
        seed=100)


def test_fake_backend_env_reset_returns_finite_observation_shape():
    env = make_env()

    observation, info = env.reset()
    values = finite_values(observation)

    assert len(values) == len(env.observation_builder.spec().feature_names)
    assert all(math.isfinite(value) for value in values)
    assert info["task_info"]
    assert set(DEFAULT_TASK_INFO).issubset(set(info["task_info"]))
    assert info["warmup_skipped_steps"] >= 60
    assert info["observation_diagnostics"]["planning_available"] == 0.0


def test_fake_backend_env_step_returns_finite_reward_and_task_info():
    env = make_env()
    env.reset()

    observation, reward, terminated, truncated, info = env.step(
        [0.25, 0.25, 0.25, 0.25])

    assert all(math.isfinite(value) for value in finite_values(observation))
    assert math.isfinite(reward)
    assert terminated in (True, False)
    assert truncated in (True, False)
    assert info["task_info"]
    assert set(DEFAULT_TASK_INFO).issubset(set(info["task_info"]))
    assert "route_progress_fraction" in info["task_info"]
    assert info["projection_diagnostics"]["rl_safety_gain"] >= 0.0
    assert "rl_residual_damper_fl" in info["projection_diagnostics"]
    assert info["backend_info"]["backend_apply_suspension"] == 1


def test_env_verify_every_applies_periodically():
    class CountingVerifyBackend(FakeCarlaSuspensionBackend):
        def __init__(self):
            super().__init__(max_steps=240)
            self.verify_flags = []

        def apply_suspension(self, command, verify):
            self.verify_flags.append(bool(verify))
            return super().apply_suspension(command, verify)

    backend = CountingVerifyBackend()
    env = CarlaSuspensionEnv(
        backend=backend,
        baseline="skyhook",
        planning_provider="empty",
        warmup_seconds=0.0,
        verify_suspension=True,
        verify_every=3,
        route_id="00",
        seed=100)
    env.reset()

    env.step([0.0, 0.0, 0.0, 0.0])
    env.step([0.0, 0.0, 0.0, 0.0])
    env.step([0.0, 0.0, 0.0, 0.0])

    assert backend.verify_flags == [False, False, True]


def test_normalize_task_info_rejects_empty_when_required():
    try:
        normalize_task_info({}, require_non_empty=True)
    except RuntimeError as error:
        assert "task_info is empty" in str(error)
    else:
        raise AssertionError("empty task_info should fail when required")


def test_fake_backend_env_rejects_empty_task_info():
    class EmptyTaskBackend(FakeCarlaSuspensionBackend):
        def tick(self):
            result = super().tick()
            return type(result)(
                state=result.state,
                planning=result.planning,
                task_info={},
                terminated=result.terminated,
                truncated=result.truncated,
                info=result.info)

    env = CarlaSuspensionEnv(
        backend=EmptyTaskBackend(max_steps=240),
        warmup_seconds=0.0)
    env.reset()

    try:
        env.step([0.0, 0.0, 0.0, 0.0])
    except RuntimeError as error:
        assert "task_info is empty" in str(error)
    else:
        raise AssertionError("empty task_info should fail in online env")


def test_rollout_logger_writes_reward_and_projection_diagnostics(tmp_path):
    env = make_env(warmup_seconds=0.0)
    env.reset()
    action = [0.25, 0.25, 0.25, 0.25]
    _, _, terminated, truncated, info = env.step(action)

    with RolloutLogger(str(tmp_path)) as logger:
        logger.log_transition(
            episode_id="fake",
            step=0,
            state=env.state,
            planning=env.planning,
            baseline_command=info["baseline_command"],
            action=action,
            final_command=info["final_command"],
            reward_diagnostics=info["reward_diagnostics"],
            controller_diagnostics=info["projection_diagnostics"],
            extra=dict(
                info["task_info"],
                terminated=int(terminated),
                truncated=int(truncated)))

    with open(tmp_path / "rollout.csv") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == 1
    assert rows[0]["reward_total"] != ""
    assert rows[0]["rl_safety_gain"] != ""
    assert rows[0]["rl_raw_residual_damper_fl"] != ""
    assert rows[0]["raw_residual_damper_fl"] != ""
    assert rows[0]["final_damper_fl"] != ""
    assert rows[0]["planning_source"] == "empty"
    assert rows[0]["planning_stale"] == "0"
    assert rows[0]["route_progress_fraction"] != ""


def test_raw_negative_progress_is_diagnostic_when_monotonic_delta_matches():
    reward = SuspensionReward()
    state = VehicleState(speed=8.0, local_vx=8.0, dt=0.05)
    command = SuspensionCommand.identity()
    forward_value, forward_diag = reward.compute(RewardTransition(
        state=state,
        previous_state=state,
        action=[0.0, 0.0, 0.0, 0.0],
        previous_action=[0.0, 0.0, 0.0, 0.0],
        final_command=command,
        baseline_command=command,
        task_info={
            "route_progress_delta_m": 0.003,
            "route_delta_progress_m": 0.003,
            "target_speed": 8.0,
            "target_speed_error": 0.0,
        }))
    raw_negative_value, raw_negative_diag = reward.compute(RewardTransition(
        state=state,
        previous_state=state,
        action=[0.0, 0.0, 0.0, 0.0],
        previous_action=[0.0, 0.0, 0.0, 0.0],
        final_command=command,
        baseline_command=command,
        task_info={
            "route_progress_delta_m": 0.003,
            "route_delta_progress_m": -0.002,
            "route_progress_negative_raw": 1.0,
            "target_speed": 8.0,
            "target_speed_error": 0.0,
        }))

    assert raw_negative_diag["reward_term_negative_progress"] == 0.0
    assert raw_negative_diag["reward_task"] == forward_diag["reward_task"]
    assert raw_negative_value == forward_value


def test_reward_task_worsens_for_large_target_speed_error():
    reward = SuspensionReward()
    state = VehicleState(speed=2.0, local_vx=2.0, dt=0.05)
    command = SuspensionCommand.identity()
    near_value, near_diag = reward.compute(RewardTransition(
        state=state,
        previous_state=state,
        action=[0.0, 0.0, 0.0, 0.0],
        previous_action=[0.0, 0.0, 0.0, 0.0],
        final_command=command,
        baseline_command=command,
        task_info={
            "delta_progress": 0.003,
            "target_speed": 8.0,
            "target_speed_error": 0.1,
        }))
    slow_value, slow_diag = reward.compute(RewardTransition(
        state=state,
        previous_state=state,
        action=[0.0, 0.0, 0.0, 0.0],
        previous_action=[0.0, 0.0, 0.0, 0.0],
        final_command=command,
        baseline_command=command,
        task_info={
            "delta_progress": 0.003,
            "target_speed": 8.0,
            "target_speed_error": 5.0,
        }))

    assert slow_diag["reward_term_target_speed_error"] > (
        near_diag["reward_term_target_speed_error"])
    assert slow_diag["reward_task"] < near_diag["reward_task"]
    assert slow_value < near_value


def test_live_backend_skeleton_fails_clearly():
    backend = LiveCarlaSuspensionBackend(
        host="127.0.0.1",
        port=2000,
        carla_importer=lambda: (_ for _ in ()).throw(
            ImportError("carla missing")),
        required_route_modules=())
    try:
        backend.reset(seed=100, route_id="00")
    except RuntimeError as error:
        message = str(error)
        assert "LiveRouteProcessBackend dependencies are unavailable" in message
        assert "must not fall back to synthetic training" in message
        assert "carla import failed" in message
        assert LIVE_BACKEND_UNAVAILABLE_MESSAGE in message
    else:
        raise AssertionError("live backend should fail when dependencies are missing")
