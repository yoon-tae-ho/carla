import csv

from suspension_control.rl.carla_route_backend import FakeCarlaRouteBackend
from suspension_control.rl.carla_route_env import CarlaRouteSuspensionEnv
from suspension_control.rl.rollout_logger import RolloutLogger


def test_phase4_rollout_logger_writes_backend_and_truncation_fields(tmp_path):
    env = CarlaRouteSuspensionEnv(
        backend=FakeCarlaRouteBackend(max_steps=200),
        warmup_seconds=0.0,
        max_episode_steps=1,
        route_id="00",
        seed=100)
    env.reset()
    row = env.step([0.25, 0.25, 0.25, 0.25])[4]["rollout_row"]

    with RolloutLogger(str(tmp_path)) as logger:
        logger.log(row)

    with open(tmp_path / "rollout.csv") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert len(rows) == 1
    assert rows[0]["backend"] == "fake"
    assert rows[0]["fake_backend_used"] == "1"
    assert rows[0]["real_backend_used"] == "0"
    assert rows[0]["route_process_alive"] == "0"
    assert rows[0]["truncated"] == "1"
    assert rows[0]["truncated_reason"] == "max_episode_steps"
    assert rows[0]["reward_total"] != ""
    assert rows[0]["route_progress_delta_m"] != ""
    assert rows[0]["planning_source"] == "empty"
