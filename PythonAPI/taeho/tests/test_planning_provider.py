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


def test_jsonl_provider_reads_lead_planning_preview_schema(tmp_path):
    path = Path(tmp_path) / "lead_planning.jsonl"
    path.write_text(
        "{\"schema\": \"taeho.planning_preview.v1\", "
        "\"source\": \"lead_tfv6\", "
        "\"available\": true, "
        "\"step\": 123, "
        "\"frame\": 4567, "
        "\"timestamp\": 12.3, "
        "\"horizon_dt\": 0.1, "
        "\"trajectory_xy\": [[2.0, 0.1], [4.0, 0.2], [6.0, 0.4]], "
        "\"target_speed\": [7.8], "
        "\"curvature\": [0.0, 0.03, 0.0], "
        "\"steer\": [0.12], "
        "\"throttle\": [0.35], "
        "\"brake\": [0.0], "
        "\"metadata\": {\"valid_prediction\": true, \"ego_speed_mps\": 8.5}, "
        "\"extra\": {\"target_points_ego\": {\"current\": [5.0, 0.5]}}}\n")
    provider = JsonlPlanningInfoProvider(str(path), max_frame_lag=5)
    info = provider.get(4567, VehicleState(frame=4567, speed=8.0), None)
    assert info.source == "lead_tfv6"
    assert info.available is True
    assert info.frame == 4567
    assert info.trajectory_xy == ((2.0, 0.1), (4.0, 0.2), (6.0, 0.4))
    assert info.target_speed == (7.8,)
    assert info.metadata["valid_prediction"] is True
    assert info.extra["target_points_ego"]["current"] == [5.0, 0.5]


def test_jsonl_provider_preserves_top_level_preview_provenance(tmp_path):
    path = Path(tmp_path) / "lead_planning_v2.jsonl"
    path.write_text(
        "{\"schema\": \"taeho.planning_preview.v1\", "
        "\"source\": \"lead_tfv6\", "
        "\"available\": true, "
        "\"step\": 123, "
        "\"frame\": 4567, "
        "\"producer_frame\": 4567, "
        "\"producer_step\": 123, "
        "\"producer_timestamp\": 12.3, "
        "\"write_timestamp\": 12.4, "
        "\"exporter_schema_version\": \"lead_planning_preview_exporter_v2\", "
        "\"exporter_code_version\": \"unit\", "
        "\"trajectory_source\": \"pred_route\", "
        "\"trajectory_frame\": \"lead_model_output_xy_unverified\", "
        "\"trajectory_point_semantics\": \"unverified\", "
        "\"horizon_dt_source\": \"exporter_default_constant_0.1_s\", "
        "\"speed_semantics\": "
        "\"pred_target_speed_scalar_flattened_not_time_sequence\", "
        "\"control_semantics\": \"current_final_control_single_sample\", "
        "\"validity_reason\": \"valid_prediction_with_preview\", "
        "\"trajectory_xy\": [[2.0, 0.1], [4.0, 0.2]], "
        "\"target_speed\": [7.8], "
        "\"metadata\": {\"valid_prediction\": true}}\n")
    provider = JsonlPlanningInfoProvider(str(path), max_frame_lag=5)

    info = provider.get(4567, VehicleState(frame=4567, speed=8.0), None)

    assert info.metadata["exporter_schema_version"] == (
        "lead_planning_preview_exporter_v2")
    assert info.metadata["producer_frame"] == 4567
    assert info.metadata["trajectory_source"] == "pred_route"
    assert info.metadata["speed_semantics"] == (
        "pred_target_speed_scalar_flattened_not_time_sequence")
    assert info.metadata["control_semantics"] == (
        "current_final_control_single_sample")
    assert info.metadata["validity_reason"] == "valid_prediction_with_preview"


def test_jsonl_provider_waits_for_partial_final_line(tmp_path):
    path = Path(tmp_path) / "planning_partial.jsonl"
    path.write_text(
        "{\"frame\": 10, \"target_speed\": [8.0]}\n"
        "{\"frame\": 11")
    provider = JsonlPlanningInfoProvider(str(path), max_frame_lag=5)

    info = provider.get(10, VehicleState(frame=10, speed=8.0), None)
    assert info.frame == 10
    assert info.metadata["planning_jsonl_malformed_lines"] == 0

    with path.open("a") as jsonl_file:
        jsonl_file.write(", \"target_speed\": [9.0]}\n")

    info = provider.get(11, VehicleState(frame=11, speed=8.0), None)
    assert info.frame == 11
    assert info.target_speed == (9.0,)
    assert info.metadata["planning_jsonl_malformed_lines"] == 0


def test_jsonl_provider_rejects_stale_frames(tmp_path):
    path = Path(tmp_path) / "planning.jsonl"
    path.write_text("{\"frame\": 4, \"target_speed\": [5.0]}\n")
    provider = JsonlPlanningInfoProvider(str(path), max_frame_lag=2)
    info = provider.get(10, VehicleState(frame=10), None)
    assert info.source == "jsonl_stale"
    assert info.metadata["stale_frame"] == 4
    assert info.preview_summary()["planning_available"] == 0.0
