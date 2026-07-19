"""CARLA-free contract checks for the LEAD planning-preview exporter."""

from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from suspension_control.controllers.base import VehicleState
from suspension_control.runtime.planning_provider import JsonlPlanningInfoProvider


def _load_exporter_class():
    sim_root = Path(__file__).resolve().parents[4]
    exporter_path = (
        sim_root /
        "e2e_models" /
        "lead" /
        "lead" /
        "inference" /
        "planning_preview_exporter.py")
    spec = importlib.util.spec_from_file_location(
        "planning_preview_exporter_for_test",
        str(exporter_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.PlanningPreviewJsonlExporter


class _FakeTimestamp:
    elapsed_seconds = 12.345


class _FakeSnapshot:
    frame = 4242
    timestamp = _FakeTimestamp()


class _FakeWorld:

    def get_snapshot(self):
        return _FakeSnapshot()


class PlanningPreviewExporterTest(unittest.TestCase):

    def test_disabled_exporter_is_inert(self):
        exporter_cls = _load_exporter_class()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "disabled.jsonl"
            exporter = exporter_cls("")

            exporter.write_prediction(
                step=1,
                input_data={"speed": 8.0},
                prediction=SimpleNamespace(
                    pred_route=[[0.0, 0.0], [1.0, 0.0]]),
                control=SimpleNamespace(steer=0.1, throttle=0.2, brake=0.0),
                world=_FakeWorld())

            self.assertFalse(path.exists())

    def test_exporter_adds_provenance_without_mutating_prediction_or_control(self):
        exporter_cls = _load_exporter_class()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "planning_preview.jsonl"
            exporter = exporter_cls(str(path))
            input_data = {
                "speed": 8.5,
                "compass": 0.12,
                "target_point_previous": [3.0, 0.1],
                "target_point": [4.0, 0.2],
                "target_point_next": [5.0, 0.3],
            }
            prediction = SimpleNamespace(
                pred_route=[[0.0, 0.0], [1.0, 0.1], [2.0, 0.4]],
                pred_future_waypoints=[[0.5, 0.0], [1.5, 0.2]],
                pred_target_speed_scalar=[9.5],
                pred_bounding_box_vehicle_system=(),
                route_steer=0.11,
                waypoints_steer=0.12,
                target_speed_throttle=0.33,
                target_speed_brake=0.0,
            )
            control = SimpleNamespace(steer=0.14, throttle=0.31, brake=0.02)
            input_before = copy.deepcopy(input_data)
            prediction_before = copy.deepcopy(prediction.__dict__)
            control_before = copy.deepcopy(control.__dict__)

            exporter.write_prediction(
                step=17,
                input_data=input_data,
                prediction=prediction,
                control=control,
                world=_FakeWorld(),
                valid_prediction=True,
                compass_rad=0.12)
            exporter.close()

            self.assertEqual(input_before, input_data)
            self.assertEqual(prediction_before, prediction.__dict__)
            self.assertEqual(control_before, control.__dict__)

            rows = path.read_text().splitlines()
            self.assertEqual(1, len(rows))
            row = json.loads(rows[0])
            self.assertEqual("taeho.planning_preview.v1", row["schema"])
            self.assertEqual(
                "lead_planning_preview_exporter_v2",
                row["exporter_schema_version"])
            self.assertEqual(
                "planning_preview_exporter_step02_provenance_v1",
                row["exporter_code_version"])
            self.assertEqual("lead_tfv6", row["source"])
            self.assertTrue(row["available"])
            self.assertEqual(17, row["step"])
            self.assertEqual(17, row["producer_step"])
            self.assertEqual(4242, row["frame"])
            self.assertEqual(4242, row["producer_frame"])
            self.assertEqual(12.345, row["timestamp"])
            self.assertEqual(12.345, row["producer_timestamp"])
            self.assertIsInstance(row["write_timestamp"], float)
            self.assertEqual("pred_route", row["trajectory_source"])
            self.assertEqual(prediction_before["pred_route"], row["trajectory_xy"])
            self.assertEqual([9.5], row["target_speed"])
            self.assertEqual([control.steer], row["steer"])
            self.assertEqual([control.throttle], row["throttle"])
            self.assertEqual([control.brake], row["brake"])
            self.assertEqual(
                "lead_model_output_xy_unverified",
                row["trajectory_frame"])
            self.assertEqual(
                "lead_pred_route_xy_points_time_semantics_unverified",
                row["trajectory_point_semantics"])
            self.assertEqual(
                "exporter_default_constant_0.1_s",
                row["horizon_dt_source"])
            self.assertEqual(
                "pred_target_speed_scalar_flattened_not_time_sequence",
                row["speed_semantics"])
            self.assertEqual(
                "current_final_control_single_sample",
                row["control_semantics"])
            self.assertEqual(
                "valid_prediction_with_preview",
                row["validity_reason"])
            self.assertFalse(row["metadata"]["future_speed_sequence_available"])
            self.assertFalse(row["metadata"]["future_control_sequence_available"])
            self.assertEqual(
                prediction_before["pred_future_waypoints"],
                row["extra"]["future_waypoints_ego"])
            self.assertFalse(row["extra"]["future_speed_sequence_available"])
            self.assertFalse(row["extra"]["future_control_sequence_available"])

            provider = JsonlPlanningInfoProvider(str(path), max_frame_lag=5)
            info = provider.get(4242, VehicleState(frame=4242, speed=8.5), None)
            self.assertEqual("lead_tfv6", info.source)
            self.assertTrue(info.available)
            self.assertEqual(4242, info.frame)
            self.assertEqual(
                ((0.0, 0.0), (1.0, 0.1), (2.0, 0.4)),
                info.trajectory_xy)
            self.assertEqual((9.5,), info.target_speed)
            self.assertEqual(
                "lead_planning_preview_exporter_v2",
                info.metadata["exporter_schema_version"])
            self.assertEqual(
                "pred_target_speed_scalar_flattened_not_time_sequence",
                info.metadata["speed_semantics"])
            self.assertEqual(
                "current_final_control_single_sample",
                info.metadata["control_semantics"])
            self.assertFalse(info.extra["future_speed_sequence_available"])

    def test_provider_preserves_top_level_provenance_without_metadata_duplicate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "lead_planning_v2.jsonl"
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
                "\"exporter_schema_version\": "
                "\"lead_planning_preview_exporter_v2\", "
                "\"exporter_code_version\": \"unit\", "
                "\"trajectory_source\": \"pred_route\", "
                "\"trajectory_frame\": \"lead_model_output_xy_unverified\", "
                "\"trajectory_point_semantics\": \"unverified\", "
                "\"horizon_dt_source\": "
                "\"exporter_default_constant_0.1_s\", "
                "\"speed_semantics\": "
                "\"pred_target_speed_scalar_flattened_not_time_sequence\", "
                "\"control_semantics\": "
                "\"current_final_control_single_sample\", "
                "\"validity_reason\": \"valid_prediction_with_preview\", "
                "\"trajectory_xy\": [[2.0, 0.1], [4.0, 0.2]], "
                "\"target_speed\": [7.8], "
                "\"metadata\": {\"valid_prediction\": true}}\n")
            provider = JsonlPlanningInfoProvider(str(path), max_frame_lag=5)

            info = provider.get(4567, VehicleState(frame=4567, speed=8.0), None)

            self.assertEqual(
                "lead_planning_preview_exporter_v2",
                info.metadata["exporter_schema_version"])
            self.assertEqual(4567, info.metadata["producer_frame"])
            self.assertEqual("pred_route", info.metadata["trajectory_source"])
            self.assertEqual(
                "pred_target_speed_scalar_flattened_not_time_sequence",
                info.metadata["speed_semantics"])
            self.assertEqual(
                "current_final_control_single_sample",
                info.metadata["control_semantics"])
            self.assertEqual(
                "valid_prediction_with_preview",
                info.metadata["validity_reason"])


if __name__ == "__main__":
    unittest.main()
