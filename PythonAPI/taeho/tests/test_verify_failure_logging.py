"""CARLA-free regression tests for verify failure logging."""

from __future__ import annotations

import csv
import io
import json
import os
import sys
import types
import unittest


TAEHO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if TAEHO_ROOT not in sys.path:
    sys.path.insert(0, TAEHO_ROOT)

import transfuser_suspension_control_suite as suite
from suspension_control.controllers.base import (
    ControllerOutput,
    PlanningInfo,
    SuspensionCommand,
    VehicleState,
    WheelScale,
)
from suspension_control.runtime.carla_adapter import (
    ScaleReadbackMismatch,
    assert_scale_match,
)


class _FakeWheel:

    def __init__(self, spring=1000.0, damper=200.0):
        self.spring_strength = float(spring)
        self.spring_damper_rate = float(damper)
        self.max_compression = 0.1
        self.max_droop = 0.1
        self.sprung_mass = 250.0


class _FakeSuspensionControl:

    def __init__(self, dampers=(200.0, 200.0, 200.0, 200.0)):
        self.wheels = tuple(_FakeWheel(damper=value) for value in dampers)


class _FakeActor:

    id = 42
    type_id = "vehicle.fake"
    attributes = {"role_name": "hero"}

    def __init__(self, readback=None, deferred_readback=None):
        self._readback = readback or _FakeSuspensionControl()
        self._deferred_readback = deferred_readback
        self._pending_readback = None
        self._stale_reads_remaining = 0

    def get_suspension_physics_control(self):
        if self._stale_reads_remaining > 0:
            self._stale_reads_remaining -= 1
            return self._readback
        if self._pending_readback is not None:
            self._readback = self._deferred_readback or self._pending_readback
            self._pending_readback = None
        return self._readback

    def stage_applied_command(self, native, command, stale_reads=1):
        dampers = tuple(
            native_wheel.spring_damper_rate * command_wheel.damper_scale
            for native_wheel, command_wheel in zip(native.wheels, command.wheels))
        self._pending_readback = _FakeSuspensionControl(dampers=dampers)
        self._stale_reads_remaining = stale_reads


class _FakeController:

    def __init__(self, output):
        self.output = output

    def compute(self, context):
        return self.output


class _FakePlanningProvider:

    def get(self, frame, state, previous_state):
        return PlanningInfo.empty()


class _FakeRouteProgressTracker:

    def update(self, state, actor_id, dt):
        return {}


class _FakeTaskInfoBuilder:

    def build(self, **kwargs):
        return {}

    def reset(self, actor_id):
        return None


class _FakeReward:

    def compute(self, transition):
        return 0.0, {}


def _state(frame=777, step=7, elapsed_seconds=12.5):
    return VehicleState(
        step=step,
        frame=frame,
        elapsed_seconds=elapsed_seconds,
        dt=0.05,
        speed=8.0,
        roll=1.0,
        pitch=0.5,
        yaw=3.0,
        local_ay=0.25,
        yaw_rate=2.0,
        roll_rate=1.5,
        pitch_rate=0.2,
        local_vx=8.0,
        local_vy=0.1,
        vz=0.0,
        local_ax=0.0,
        az=0.0,
        throttle=0.2,
        brake=0.0,
        steer=0.01,
    )


def _residual_diagnostics():
    result = {
        "residual_mpc_suppression_reason": "none",
        "residual_mpc_excitation_active": 1,
        "residual_mpc_modal_mean": 0.01,
        "residual_mpc_modal_roll_front": -0.02,
        "residual_mpc_modal_roll_rear": 0.03,
        "residual_mpc_modal_pitch": -0.04,
    }
    for prefix, base in (
            ("residual_mpc_phase_a_shadow_damper", 1.0),
            ("residual_mpc_requested_residual", 0.01),
            ("residual_mpc_bounded_residual", 0.01),
            ("residual_mpc_rate_limited_residual", 0.01),
            ("residual_mpc_projected_residual", 0.01),
            ("residual_mpc_final_residual", 0.01),
            ("residual_mpc_final_damper", 1.01)):
        for index, label in enumerate(("fl", "fr", "rl", "rr")):
            result["%s_%s" % (prefix, label)] = base + 0.001 * index
    return result


def _output(command=None, diagnostics=None):
    return ControllerOutput(
        command=command or SuspensionCommand.uniform(damper_scale=1.0),
        diagnostics=diagnostics if diagnostics is not None else _residual_diagnostics(),
    )


def _sidecar(output):
    sidecar = suite.SuspensionExperimentSidecar.__new__(
        suite.SuspensionExperimentSidecar)
    sidecar.args = types.SimpleNamespace(
        default_dt=0.05,
        verify_every=1,
        readback_every=0,
        readback_tolerance=1.0e-4,
    )
    sidecar.scenario = {
        "name": "S33_planning_aware_v5_residual_id_probe",
        "label": "residual id probe",
        "controller": "fake",
        "uses_suspension_api": True,
        "needs_suspension_state": False,
    }
    sidecar.seed = 111
    sidecar.native_by_actor_id = {42: _FakeSuspensionControl()}
    sidecar.controller_by_actor_id = {42: _FakeController(output)}
    sidecar.previous_state_by_actor_id = {}
    sidecar.episode_by_actor_id = {42: 3}
    sidecar.step_by_actor_id = {42: 7}
    sidecar.apply_count_by_actor_id = {42: 0}
    sidecar.verify_count_by_actor_id = {42: 0}
    sidecar.command_sequence_by_actor_id = {42: 0}
    sidecar.pending_verify_by_actor_id = {}
    sidecar.last_frame_by_actor_id = {}
    sidecar.previous_action_by_actor_id = {}
    sidecar.previous_final_damper_by_actor_id = {}
    sidecar.planning_provider = _FakePlanningProvider()
    sidecar.route_progress_tracker = _FakeRouteProgressTracker()
    sidecar.task_info_builder = _FakeTaskInfoBuilder()
    sidecar.reward = _FakeReward()
    return sidecar


def _writers():
    diagnostic_buffer = io.StringIO()
    event_buffer = io.StringIO()
    diagnostic_writer = csv.DictWriter(
        diagnostic_buffer,
        fieldnames=suite.DIAGNOSTIC_FIELDS)
    event_writer = csv.DictWriter(
        event_buffer,
        fieldnames=suite.EVENT_FIELDS)
    diagnostic_writer.writeheader()
    event_writer.writeheader()
    return diagnostic_buffer, diagnostic_writer, event_buffer, event_writer


def _rows(buffer):
    buffer.seek(0)
    return list(csv.DictReader(buffer))


class VerifyFailureLoggingTests(unittest.TestCase):

    def test_adapter_mismatch_exception_carries_structured_payload(self):
        native = _FakeSuspensionControl()
        readback = _FakeSuspensionControl(dampers=(200.0, 196.0, 200.0, 200.0))

        with self.assertRaises(ScaleReadbackMismatch) as raised:
            assert_scale_match(
                SuspensionCommand.uniform(damper_scale=1.0),
                native,
                readback,
                tolerance=1.0e-4)

        payload = raised.exception.payload
        self.assertEqual("damper_readback_mismatch", payload["hard_error_type"])
        self.assertEqual("fr", payload["mismatch_wheel_label"])
        self.assertAlmostEqual(0.02, payload["mismatch_max_abs"])
        self.assertEqual("fr", payload["first_mismatch_wheel_label"])
        self.assertAlmostEqual(0.02, payload["first_mismatch_abs"])
        self.assertEqual("fr", payload["vector_mismatch_wheel_label"])
        self.assertAlmostEqual(0.02, payload["vector_mismatch_max_abs"])
        self.assertEqual((1.0, 0.98, 1.0, 1.0), payload["readback_damper_scales"])

    def test_deferred_verify_success_preserves_diagnostic_and_event_logging(self):
        command = SuspensionCommand(tuple(
            WheelScale(damper_scale=value)
            for value in (1.04, 0.96, 1.02, 0.98)))
        sidecar = _sidecar(_output(command=command))
        actor = _FakeActor()
        diagnostic_buffer, diagnostic_writer, event_buffer, event_writer = _writers()
        original_apply = suite.apply_suspension_command

        def fake_apply(actor_arg, native, command, **kwargs):
            self.assertFalse(kwargs.get("verify_readback"))
            actor_arg.stage_applied_command(native, command)
            return {"applied_control": object()}

        suite.apply_suspension_command = fake_apply
        try:
            sidecar.process_controller(
                diagnostic_writer,
                event_writer,
                actor,
                _state(frame=777, step=7, elapsed_seconds=12.5))
            sidecar.process_controller(
                diagnostic_writer,
                event_writer,
                actor,
                _state(frame=778, step=8, elapsed_seconds=12.55))
        finally:
            suite.apply_suspension_command = original_apply

        diagnostics = _rows(diagnostic_buffer)
        events = _rows(event_buffer)
        self.assertEqual(2, len(diagnostics))
        self.assertEqual("success", diagnostics[0]["diagnostic_status"])
        self.assertEqual("pass", diagnostics[0]["verify_status"])
        self.assertEqual("1", diagnostics[0]["apply_count"])
        self.assertEqual("0", diagnostics[0]["verify_count"])
        self.assertEqual("1", diagnostics[0]["command_sequence_id"])
        self.assertEqual("1", diagnostics[0]["verify_command_sequence_id"])
        self.assertEqual(
            "immediate_after_apply",
            diagnostics[0]["verify_readback_timing"])
        self.assertEqual(
            "mismatch_deferred_pending",
            diagnostics[0]["immediate_readback_status"])
        self.assertEqual("fl", diagnostics[0]["immediate_first_mismatch_wheel"])
        self.assertEqual("pass", diagnostics[1]["verify_status"])
        self.assertEqual("1", diagnostics[1]["verify_count"])
        self.assertEqual(
            "deferred_next_frame",
            diagnostics[1]["verify_readback_timing"])
        self.assertEqual(
            ["command_verify_pending", "command_verified", "command_verify_pending"],
            [row["event"] for row in events])
        self.assertEqual("verify_pending", events[0]["event_type"])
        self.assertEqual("immediate_after_apply", events[0]["readback_timing"])
        self.assertEqual("command_verified", events[1]["event_type"])
        self.assertEqual("deferred_next_frame", events[1]["readback_timing"])
        self.assertEqual("1", events[1]["verify_command_sequence_id"])

    def test_readback_mismatch_logs_verify_failure_event_and_row(self):
        command = SuspensionCommand.uniform(damper_scale=1.0)
        sidecar = _sidecar(_output(command=command))
        actor = _FakeActor(deferred_readback=_FakeSuspensionControl(
            dampers=(200.0, 196.0, 200.0, 200.0)))
        diagnostic_buffer, diagnostic_writer, event_buffer, event_writer = _writers()
        original_apply = suite.apply_suspension_command

        def fake_apply(actor_arg, native, command_arg, **kwargs):
            self.assertFalse(kwargs.get("verify_readback"))
            actor_arg.stage_applied_command(native, command_arg)
            return {"applied_control": object()}

        suite.apply_suspension_command = fake_apply
        try:
            sidecar.process_controller(
                diagnostic_writer,
                event_writer,
                actor,
                _state(frame=777, step=7, elapsed_seconds=12.5))
            sidecar.process_controller(
                diagnostic_writer,
                event_writer,
                actor,
                _state(frame=778, step=8, elapsed_seconds=12.55))
        finally:
            suite.apply_suspension_command = original_apply

        diagnostics = _rows(diagnostic_buffer)
        events = _rows(event_buffer)
        self.assertEqual(2, len(diagnostics))
        self.assertEqual("success", diagnostics[0]["diagnostic_status"])
        self.assertEqual("pass", diagnostics[0]["verify_status"])
        self.assertEqual("verify_failure", diagnostics[1]["diagnostic_status"])
        self.assertEqual("hard_error", diagnostics[1]["verify_status"])
        self.assertEqual(
            "damper_readback_mismatch",
            diagnostics[1]["verify_failure_type"])
        self.assertEqual("fr", diagnostics[1]["readback_mismatch_wheel"])
        self.assertEqual("fr", diagnostics[1]["first_mismatch_wheel"])
        self.assertEqual("fr", diagnostics[1]["vector_mismatch_max_wheel"])
        runtime_events = [row for row in events if row["event"] == "runtime_error"]
        self.assertEqual(1, len(runtime_events))
        failure = runtime_events[0]
        self.assertEqual("verify_failure", failure["event_type"])
        self.assertEqual("hard_error", failure["severity"])
        self.assertEqual("778", failure["frame"])
        self.assertEqual("12.55", failure["elapsed_seconds"])
        self.assertEqual("42", failure["actor_id"])
        self.assertEqual("3", failure["episode_index"])
        self.assertEqual("1", failure["verify_command_sequence_id"])
        self.assertEqual("777", failure["verify_apply_frame"])
        self.assertEqual("778", failure["verify_deferred_frame"])
        self.assertEqual("deferred_next_frame", failure["readback_timing"])
        self.assertEqual([1.0, 1.0, 1.0, 1.0], json.loads(
            failure["expected_damper_scale_json"]))
        self.assertEqual([1.0, 0.98, 1.0, 1.0], json.loads(
            failure["readback_damper_scale_json"]))
        for actual, expected in zip(
                json.loads(failure["readback_damper_delta_json"]),
                [0.0, -0.02, 0.0, 0.0]):
            self.assertAlmostEqual(expected, actual)
        self.assertEqual("0.0001", failure["readback_tolerance"])
        self.assertEqual("0", failure["diagnostic_row_emission_skipped"])
        self.assertEqual("fr", failure["first_mismatch_wheel"])
        self.assertEqual("fr", failure["vector_mismatch_max_wheel"])

    def test_actor_removed_event_is_distinct_from_verify_failure(self):
        sidecar = _sidecar(_output())
        event_buffer = io.StringIO()
        event_writer = csv.DictWriter(event_buffer, fieldnames=suite.EVENT_FIELDS)
        event_writer.writeheader()

        sidecar.remove_missing_actors(event_writer, active_actor_ids=[], frame=888)

        events = _rows(event_buffer)
        self.assertEqual(1, len(events))
        self.assertEqual("actor_removed", events[0]["event"])
        self.assertEqual("actor_removed", events[0]["event_type"])
        self.assertEqual("warning", events[0]["severity"])
        self.assertEqual("1", events[0]["actor_removed"])
        self.assertEqual("42", events[0]["actor_id"])
        self.assertEqual(
            "actor_removed",
            events[0]["diagnostic_row_emission_skipped_reason"])


if __name__ == "__main__":
    unittest.main()
