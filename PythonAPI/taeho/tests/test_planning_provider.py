from pathlib import Path

from suspension_control.controllers.base import PlanningInfo, VehicleState
from suspension_control.runtime.planning_provider import (
    EmptyPlanningInfoProvider,
    JsonlPlanningInfoProvider,
)


def test_empty_provider_and_compatibility_fields():
    info = EmptyPlanningInfoProvider().get(1, VehicleState(frame=1), None)
    assert info.points == ()
    assert hasattr(info, "source")
    assert hasattr(info, "metadata")
    assert info.preview_summary()["planning_available"] == 0.0


def test_planning_info_preview_summary_keys():
    info = PlanningInfo(
        available=True,
        frame=10,
        horizon_dt=0.1,
        target_speed=(8.0, 9.0),
        curvature=(0.0, 0.05),
        brake=(0.0, 0.3))
    summary = info.preview_summary()
    assert summary["planning_available"] == 1.0
    assert summary["preview_target_speed_mean"] == 8.5
    assert summary["preview_curvature_max_abs"] == 0.05
    assert summary["time_to_hard_brake"] == 0.1


def test_jsonl_provider_reads_valid_lines_and_ignores_malformed(tmp_path):
    path = Path(tmp_path) / "planning.jsonl"
    path.write_text(
        "{bad json\n"
        "{\"frame\": 10, \"horizon_dt\": 0.1, "
        "\"trajectory_xy\": [[0, 0], [1, 0.1]], "
        "\"target_speed\": [8.0, 9.0], "
        "\"curvature\": [0.0, 0.05], "
        "\"brake\": [0.0, 0.3]}\n")
    provider = JsonlPlanningInfoProvider(str(path), max_frame_lag=5)
    info = provider.get(10, VehicleState(frame=10, speed=8.0), None)
    assert info.source == "jsonl"
    assert info.frame == 10
    assert len(info.points) == 2
    assert info.metadata["planning_jsonl_malformed_lines"] == 1
    assert info.preview_summary()["preview_brake_max"] == 0.3


def test_jsonl_provider_rejects_stale_frames(tmp_path):
    path = Path(tmp_path) / "planning.jsonl"
    path.write_text("{\"frame\": 4, \"target_speed\": [5.0]}\n")
    provider = JsonlPlanningInfoProvider(str(path), max_frame_lag=2)
    info = provider.get(10, VehicleState(frame=10), None)
    assert info.source == "jsonl_stale"
    assert info.metadata["stale_frame"] == 4
    assert info.preview_summary()["planning_available"] == 0.0
