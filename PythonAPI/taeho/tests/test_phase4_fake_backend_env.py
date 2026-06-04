import math

from suspension_control.rl.carla_route_backend import FakeCarlaRouteBackend
from suspension_control.rl.carla_route_env import CarlaRouteSuspensionEnv


def test_phase4_alias_env_reset_uses_fake_backend_and_skips_warmup():
    backend = FakeCarlaRouteBackend(max_steps=200)
    env = CarlaRouteSuspensionEnv(
        backend=backend,
        baseline="skyhook",
        planning_provider="empty",
        warmup_seconds=0.15,
        max_episode_steps=4,
        route_id="00",
        seed=100)

    observation, info = env.reset()

    assert len(list(observation)) == len(env.observation_builder.spec().feature_names)
    assert all(math.isfinite(float(value)) for value in list(observation))
    assert info["warmup_skipped_steps"] == 3
    assert info["backend_info"]["backend"] == "fake"
    assert info["backend_info"]["fake_backend_used"] == 1
    assert info["backend_info"]["real_backend_used"] == 0
    assert info["backend_info"]["route_process_alive"] == 0
    assert info["task_info"]["route_progress_available"] == 1.0


def test_phase4_env_max_episode_steps_truncates_without_route_failure():
    env = CarlaRouteSuspensionEnv(
        backend=FakeCarlaRouteBackend(max_steps=200),
        warmup_seconds=0.0,
        max_episode_steps=2,
        route_id="00",
        seed=100)
    env.reset()

    first = env.step([0.25, 0.25, 0.25, 0.25])
    second = env.step([0.25, 0.25, 0.25, 0.25])

    assert first[2] is False
    assert first[3] is False
    assert second[2] is False
    assert second[3] is True
    assert second[4]["truncated_reason"] == "max_episode_steps"
    assert second[4]["done_reason"] == ""
    assert second[4]["rollout_row"]["truncated"] == 1
    assert second[4]["rollout_row"]["terminated"] == 0


def test_phase4_rollout_row_contains_training_canary_backend_fields():
    env = CarlaRouteSuspensionEnv(
        backend=FakeCarlaRouteBackend(max_steps=200),
        warmup_seconds=0.0,
        max_episode_steps=4,
        route_id="00",
        seed=100)
    env.reset()

    _, reward, _, _, info = env.step([0.25, -0.25, 0.10, -0.10])
    row = info["rollout_row"]

    assert math.isfinite(float(reward))
    assert row["backend"] == "fake"
    assert row["fake_backend_used"] == 1
    assert row["real_backend_used"] == 0
    assert row["route_process_alive"] == 0
    assert row["route_progress_available"] == 1.0
    assert row["route_progress_delta_m"] > 0.0
    assert row["planning_available"] == 0.0
    assert row["planning_source"] == "empty"
    assert row["rl_raw_residual_damper_fl"] != ""
    assert row["raw_residual_damper_fl"] == row["rl_raw_residual_damper_fl"]
    assert row["final_damper_fl"] != ""

