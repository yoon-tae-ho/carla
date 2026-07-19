"""CARLA-free regression tests for suspension command/readback timing."""

from __future__ import annotations

import unittest

from suspension_control.controllers.base import SuspensionCommand, WheelScale
from suspension_control.runtime.carla_adapter import (
    ScaleReadbackMismatch,
    apply_suspension_command,
    assert_scale_match,
    compare_suspension_scales,
    make_scaled_suspension_control,
    read_suspension_scale_summary,
)


class _FakeCarlaModule:

    class WheelSuspensionPhysicsControl:

        def __init__(
            self,
            spring_strength,
            spring_damper_rate,
            max_compression,
            max_droop,
            sprung_mass,
        ):
            self.spring_strength = float(spring_strength)
            self.spring_damper_rate = float(spring_damper_rate)
            self.max_compression = float(max_compression)
            self.max_droop = float(max_droop)
            self.sprung_mass = float(sprung_mass)

    class SuspensionPhysicsControl:

        def __init__(self):
            self.wheels = ()


class _FakeWheel:

    def __init__(self, spring, damper):
        self.spring_strength = float(spring)
        self.spring_damper_rate = float(damper)
        self.max_compression = 0.1
        self.max_droop = 0.1
        self.sprung_mass = 250.0


class _FakeSuspensionControl:

    def __init__(
        self,
        springs=(1000.0, 2000.0, 3000.0, 4000.0),
        dampers=(100.0, 200.0, 300.0, 400.0),
    ):
        self.wheels = tuple(
            _FakeWheel(spring, damper)
            for spring, damper in zip(springs, dampers))


class _OneReadStaleVehicle:

    def __init__(self, visible_control):
        self._visible_control = visible_control
        self._pending_control = None
        self._stale_reads_remaining = 0

    def apply_suspension_physics_control(self, control):
        self._pending_control = control
        self._stale_reads_remaining = 1

    def get_suspension_physics_control(self):
        if self._stale_reads_remaining > 0:
            self._stale_reads_remaining -= 1
            return self._visible_control
        if self._pending_control is not None:
            self._visible_control = self._pending_control
            self._pending_control = None
        return self._visible_control


class _PersistentWrongVehicle:

    def __init__(self, wrong_control):
        self._wrong_control = wrong_control

    def apply_suspension_physics_control(self, control):
        self._applied_control = control

    def get_suspension_physics_control(self):
        return self._wrong_control


def _command(dampers, springs=(1.0, 1.0, 1.0, 1.0)):
    return SuspensionCommand(tuple(
        WheelScale(spring_scale=spring, damper_scale=damper)
        for spring, damper in zip(springs, dampers)))


def _scaled_control(native, command):
    return make_scaled_suspension_control(
        native,
        command,
        carla_module=_FakeCarlaModule)


class CarlaAdapterReadbackTimingTest(unittest.TestCase):

    def test_stale_immediate_readback_can_raise_but_deferred_passes(self):
        native = _FakeSuspensionControl()
        previous_command = _command((0.90, 0.95, 1.00, 1.05))
        current_command = _command((1.08, 0.92, 1.14, 0.88))
        previous_control = _scaled_control(native, previous_command)
        vehicle = _OneReadStaleVehicle(previous_control)

        with self.assertRaises(ScaleReadbackMismatch) as raised:
            apply_suspension_command(
                vehicle,
                native,
                current_command,
                verify_readback=True,
                readback_tolerance=1.0e-4,
                carla_module=_FakeCarlaModule)

        immediate_payload = raised.exception.payload
        self.assertFalse(immediate_payload["matched"])
        self.assertEqual(
            (0.90, 0.95, 1.00, 1.05),
            immediate_payload["readback_damper_scales"])
        self.assertEqual("damper", immediate_payload["mismatch_field"])

        deferred_readback = vehicle.get_suspension_physics_control()
        deferred = compare_suspension_scales(
            current_command,
            native,
            deferred_readback,
            tolerance=1.0e-4)
        self.assertTrue(deferred["matched"])
        self.assertEqual(
            current_command.as_scale_lists()["damper_scales"],
            deferred["readback_damper_scales"])

    def test_real_deferred_mismatch_still_raises_hard_error(self):
        native = _FakeSuspensionControl()
        current_command = _command((1.02, 1.04, 0.98, 0.96))
        wrong_control = _scaled_control(native, _command((1.02, 0.80, 0.98, 0.96)))
        vehicle = _PersistentWrongVehicle(wrong_control)

        apply_suspension_command(
            vehicle,
            native,
            current_command,
            verify_readback=False,
            carla_module=_FakeCarlaModule)
        deferred_readback = vehicle.get_suspension_physics_control()

        with self.assertRaises(ScaleReadbackMismatch) as raised:
            assert_scale_match(
                current_command,
                native,
                deferred_readback,
                tolerance=1.0e-4)
        self.assertEqual(
            "damper_readback_mismatch",
            raised.exception.payload["hard_error_type"])
        self.assertEqual("fr", raised.exception.payload["mismatch_wheel_label"])

    def test_native_scale_basis_and_wheel_order_round_trip(self):
        native = _FakeSuspensionControl(
            springs=(1000.0, 2000.0, 3000.0, 4000.0),
            dampers=(100.0, 300.0, 700.0, 1100.0))
        command = _command(
            dampers=(0.91, 1.03, 1.17, 0.86),
            springs=(1.00, 1.01, 0.99, 1.02))
        vehicle = _OneReadStaleVehicle(_scaled_control(native, command))

        result = apply_suspension_command(
            vehicle,
            native,
            command,
            verify_readback=True,
            readback_tolerance=1.0e-4,
            carla_module=_FakeCarlaModule)
        applied = result["applied_control"]
        self.assertEqual(
            (91.0, 309.0, 819.0, 946.0),
            tuple(wheel.spring_damper_rate for wheel in applied.wheels))

        summary = read_suspension_scale_summary(native, result["readback"])
        self.assertEqual((0.91, 1.03, 1.17, 0.86), summary["damper_scales"])
        self.assertEqual((1.00, 1.01, 0.99, 1.02), summary["spring_scales"])

    def test_mismatch_logging_separates_first_mismatch_from_vector_max(self):
        native = _FakeSuspensionControl()
        expected = SuspensionCommand.uniform(damper_scale=1.0)
        actual = _scaled_control(native, _command((0.995, 1.020, 0.990, 1.010)))

        comparison = compare_suspension_scales(
            expected,
            native,
            actual,
            tolerance=1.0e-4)

        self.assertFalse(comparison["matched"])
        self.assertEqual("fl", comparison["first_mismatch_wheel_label"])
        self.assertAlmostEqual(0.005, comparison["first_mismatch_abs"])
        self.assertEqual("fl", comparison["mismatch_wheel_label"])
        self.assertAlmostEqual(0.005, comparison["mismatch_max_abs"])
        self.assertEqual("fr", comparison["vector_mismatch_wheel_label"])
        self.assertAlmostEqual(0.020, comparison["vector_mismatch_max_abs"])


if __name__ == "__main__":
    unittest.main()
