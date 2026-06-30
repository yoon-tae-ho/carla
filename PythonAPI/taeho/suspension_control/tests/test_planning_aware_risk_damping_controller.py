"""Unit tests for the planning-aware risk damping controller."""

from __future__ import annotations

import math
import unittest

from suspension_control.controllers.base import (
    ControllerContext,
    PlanningInfo,
    VehicleState,
)
from suspension_control.controllers.planning_aware_risk_damping import (
    PlanningAwareRiskDampingConfig,
    PlanningAwareRiskDampingController,
)


def _state(**overrides):
    values = {
        "frame": 100,
        "elapsed_seconds": 5.0,
        "dt": 0.05,
        "speed": 12.0,
        "local_ay": 0.0,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
        "roll_rate": 0.0,
        "pitch_rate": 0.0,
        "yaw_rate": 0.0,
        "throttle": 0.0,
        "brake": 0.0,
        "steer": 0.0,
    }
    values.update(overrides)
    return VehicleState(**values)


def _context(planning=None, state=None, dt=0.05):
    return ControllerContext(
        state=state if state is not None else _state(dt=dt),
        planning=planning if planning is not None else PlanningInfo.empty(),
        dt=dt)


def _valid_planning(**overrides):
    values = {
        "available": True,
        "frame": 100,
        "horizon_dt": 0.1,
        "target_speed": (12.0, 12.0, 12.0, 12.0),
        "curvature": (0.0, 0.0, 0.0, 0.0),
        "predicted_ay": (0.0, 0.0, 0.0, 0.0),
        "predicted_ax": (0.0, 0.0, 0.0, 0.0),
        "brake": (0.0, 0.0, 0.0, 0.0),
        "steer": (0.0, 0.0, 0.0, 0.0),
    }
    values.update(overrides)
    return PlanningInfo(**values)


def _high_ay_planning(**overrides):
    values = {
        "available": True,
        "frame": 100,
        "horizon_dt": 0.1,
        "target_speed": (12.0, 12.0, 12.0, 12.0),
        "curvature": (0.05, 0.06, 0.06, 0.05),
        "predicted_ay": (5.2, 5.4, 5.3, 5.1),
        "predicted_ax": (0.0, 0.0, 0.0, 0.0),
        "brake": (0.0, 0.0, 0.0, 0.0),
        "steer": (0.2, 0.2, 0.2, 0.2),
    }
    values.update(overrides)
    return PlanningInfo(**values)


def _damper_scales(output):
    return tuple(wheel.damper_scale for wheel in output.command.wheels)


def _spring_scales(output):
    return tuple(wheel.spring_scale for wheel in output.command.wheels)


def _assert_identity(testcase, output):
    testcase.assertEqual((1.0, 1.0, 1.0, 1.0), _spring_scales(output))
    testcase.assertEqual((1.0, 1.0, 1.0, 1.0), _damper_scales(output))


class PlanningAwareRiskDampingControllerTest(unittest.TestCase):

    def test_missing_planning_returns_identity(self):
        controller = PlanningAwareRiskDampingController()

        output = controller.compute(_context(planning=None))

        _assert_identity(self, output)
        self.assertEqual(1, output.diagnostics["fallback_active"])
        self.assertEqual("planning_unavailable", output.diagnostics["fallback_reason"])

        output = controller.compute(ControllerContext(
            state=_state(),
            planning=None,
            dt=0.05))

        _assert_identity(self, output)
        self.assertEqual("no_planning", output.diagnostics["fallback_reason"])

    def test_unavailable_planning_returns_identity(self):
        controller = PlanningAwareRiskDampingController()

        output = controller.compute(_context(PlanningInfo.empty()))

        _assert_identity(self, output)
        self.assertEqual("planning_unavailable", output.diagnostics["fallback_reason"])

    def test_stale_planning_returns_identity(self):
        controller = PlanningAwareRiskDampingController()
        planning = _valid_planning(frame=90)

        output = controller.compute(_context(planning))

        _assert_identity(self, output)
        self.assertEqual("stale_planning", output.diagnostics["fallback_reason"])

    def test_bad_horizon_dt_returns_identity(self):
        controller = PlanningAwareRiskDampingController()
        planning = _valid_planning(horizon_dt=0.5)

        output = controller.compute(_context(planning))

        _assert_identity(self, output)
        self.assertEqual("bad_horizon_dt", output.diagnostics["fallback_reason"])

    def test_invalid_prediction_metadata_returns_identity(self):
        controller = PlanningAwareRiskDampingController()
        planning = _valid_planning(metadata={"valid_prediction": False})

        output = controller.compute(_context(planning))

        _assert_identity(self, output)
        self.assertEqual(
            "metadata_invalid_prediction",
            output.diagnostics["fallback_reason"])

    def test_negative_target_speed_returns_identity(self):
        controller = PlanningAwareRiskDampingController()
        planning = _valid_planning(target_speed=(12.0, -1.0, 12.0))

        output = controller.compute(_context(planning))

        _assert_identity(self, output)
        self.assertEqual("negative_target_speed", output.diagnostics["fallback_reason"])

    def test_nonfinite_state_returns_identity(self):
        controller = PlanningAwareRiskDampingController()

        output = controller.compute(_context(
            _valid_planning(),
            state=_state(speed=float("nan"))))

        _assert_identity(self, output)
        self.assertEqual("nonfinite_state", output.diagnostics["fallback_reason"])

    def test_invalid_predicted_ay_falls_back_to_v2kappa(self):
        controller = PlanningAwareRiskDampingController(
            PlanningAwareRiskDampingConfig(
                tau_rise=0.0,
                max_scale_rate_per_s=10.0,
                enable_event_hold=False))
        planning = _valid_planning(
            predicted_ay=(float("nan"), float("nan"), float("nan")),
            curvature=(0.06, 0.06, 0.06),
            target_speed=(12.0,))

        output = controller.compute(_context(planning))

        self.assertEqual("v2kappa", output.diagnostics["predicted_ay_source"])
        self.assertGreater(output.diagnostics["predicted_ay_abs_max"], 5.0)
        self.assertGreater(_damper_scales(output)[0], 1.0)

    def test_missing_curvature_falls_back_to_trajectory(self):
        controller = PlanningAwareRiskDampingController(
            PlanningAwareRiskDampingConfig(
                tau_rise=0.0,
                max_scale_rate_per_s=10.0,
                enable_event_hold=False))
        planning = PlanningInfo(
            available=True,
            frame=100,
            horizon_dt=0.1,
            target_speed=(10.0,),
            trajectory_xy=((0.0, 0.0), (5.0, 0.0), (5.0, 5.0)),
            predicted_ax=(0.0, 0.0, 0.0),
            brake=(0.0, 0.0, 0.0))

        output = controller.compute(_context(planning))

        self.assertEqual("trajectory", output.diagnostics["curvature_source"])
        self.assertEqual("v2kappa", output.diagnostics["predicted_ay_source"])
        self.assertGreater(_damper_scales(output)[0], 1.0)

    def test_no_command_relevant_signal_falls_back_to_identity(self):
        controller = PlanningAwareRiskDampingController()
        planning = PlanningInfo(available=True, frame=100, horizon_dt=0.1)

        output = controller.compute(_context(planning))

        _assert_identity(self, output)
        self.assertEqual(
            "no_valid_preview_signal",
            output.diagnostics["fallback_reason"])

    def test_high_predicted_ay_increases_bounded_uniform_damper(self):
        controller = PlanningAwareRiskDampingController(
            PlanningAwareRiskDampingConfig(
                tau_rise=0.0,
                max_scale_rate_per_s=10.0,
                enable_event_hold=False))

        output = controller.compute(_context(_high_ay_planning()))
        dampers = _damper_scales(output)

        self.assertGreater(dampers[0], 1.0)
        self.assertLessEqual(dampers[0], 1.035)
        self.assertEqual(dampers, (dampers[0], dampers[0], dampers[0], dampers[0]))
        self.assertEqual((1.0, 1.0, 1.0, 1.0), _spring_scales(output))
        self.assertEqual(0, output.diagnostics["fallback_active"])
        self.assertEqual("none", output.diagnostics["fallback_reason"])

    def test_low_risk_straight_planning_keeps_identity(self):
        controller = PlanningAwareRiskDampingController(
            PlanningAwareRiskDampingConfig(
                tau_rise=0.0,
                max_scale_rate_per_s=10.0,
                enable_event_hold=False))

        output = controller.compute(_context(_valid_planning()))

        _assert_identity(self, output)
        self.assertEqual(0.0, output.diagnostics["risk_raw"])

    def test_rate_limiter_caps_per_tick_change(self):
        controller = PlanningAwareRiskDampingController(
            PlanningAwareRiskDampingConfig(
                tau_rise=0.0,
                max_scale_rate_per_s=0.08,
                enable_event_hold=False))

        output = controller.compute(_context(_high_ay_planning(), dt=0.05))
        damper = _damper_scales(output)[0]

        self.assertTrue(math.isclose(damper, 1.004, rel_tol=0.0, abs_tol=1e-9))
        self.assertEqual(1, output.diagnostics["rate_limit_active_uniform"])

    def test_event_hold_keeps_previous_sent_scale(self):
        controller = PlanningAwareRiskDampingController(
            PlanningAwareRiskDampingConfig(
                tau_rise=0.0,
                max_scale_rate_per_s=10.0,
                event_delta_threshold=0.01,
                event_force_refresh_s=1.0,
                enable_event_hold=True))

        first = controller.compute(_context(
            _high_ay_planning(predicted_ay=(5.5, 5.5, 5.5)),
            state=_state(elapsed_seconds=5.0)))
        second = controller.compute(_context(
            _high_ay_planning(predicted_ay=(5.1, 5.1, 5.1)),
            state=_state(elapsed_seconds=5.05)))

        self.assertEqual(0, first.diagnostics["event_held"])
        self.assertEqual(1, second.diagnostics["event_held"])
        self.assertEqual(_damper_scales(first), _damper_scales(second))

    def test_shadow_mode_returns_identity_but_logs_would_command(self):
        controller = PlanningAwareRiskDampingController(
            PlanningAwareRiskDampingConfig(
                shadow_mode=True,
                tau_rise=0.0,
                max_scale_rate_per_s=10.0,
                enable_event_hold=False))

        output = controller.compute(_context(_high_ay_planning()))

        _assert_identity(self, output)
        self.assertGreater(output.diagnostics["uniform_damper_would"], 1.0)
        self.assertEqual(1.0, output.diagnostics["uniform_damper_cmd"])
        self.assertEqual(1, output.diagnostics["shadow_mode"])

    def test_fallback_resets_smoothed_internal_risk(self):
        controller = PlanningAwareRiskDampingController(
            PlanningAwareRiskDampingConfig(
                tau_rise=0.0,
                max_scale_rate_per_s=10.0,
                enable_event_hold=False))

        active = controller.compute(_context(_high_ay_planning()))
        self.assertGreater(active.diagnostics["risk_smooth"], 0.0)

        fallback = controller.compute(_context(PlanningInfo.empty()))

        _assert_identity(self, fallback)
        self.assertEqual(0.0, controller.risk_smooth)
        self.assertEqual(0.0, fallback.diagnostics["risk_smooth"])

    def test_numeric_diagnostics_are_finite(self):
        controller = PlanningAwareRiskDampingController(
            PlanningAwareRiskDampingConfig(
                tau_rise=0.0,
                max_scale_rate_per_s=10.0,
                enable_event_hold=False))

        output = controller.compute(_context(_high_ay_planning()))

        for key, value in output.diagnostics.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                self.assertTrue(math.isfinite(float(value)), key)

    def test_same_sequence_after_reset_is_deterministic(self):
        controller = PlanningAwareRiskDampingController(
            PlanningAwareRiskDampingConfig(max_scale_rate_per_s=0.08))
        contexts = (
            _context(_valid_planning(), state=_state(elapsed_seconds=5.00)),
            _context(_high_ay_planning(), state=_state(elapsed_seconds=5.05)),
            _context(_high_ay_planning(), state=_state(elapsed_seconds=5.10)),
        )

        first = []
        for context in contexts:
            output = controller.compute(context)
            first.append((_damper_scales(output), dict(output.diagnostics)))

        controller.reset()
        second = []
        for context in contexts:
            output = controller.compute(context)
            second.append((_damper_scales(output), dict(output.diagnostics)))

        first_dampers = [item[0] for item in first]
        second_dampers = [item[0] for item in second]
        first_diagnostics = [item[1] for item in first]
        second_diagnostics = [item[1] for item in second]
        self.assertEqual(first_dampers, second_dampers)
        self.assertEqual(first_diagnostics, second_diagnostics)

    def test_front_rear_mode_keeps_left_right_pairs(self):
        controller = PlanningAwareRiskDampingController(
            PlanningAwareRiskDampingConfig(
                output_mode="front_rear",
                damper_max=1.06,
                max_uniform_extra=0.045,
                tau_rise=0.0,
                max_scale_rate_per_s=10.0,
                enable_event_hold=False))

        output = controller.compute(_context(_high_ay_planning(
            predicted_ax=(-4.0, -4.0, -4.0),
            brake=(0.8, 0.8, 0.8))))
        dampers = _damper_scales(output)

        self.assertEqual(dampers[0], dampers[1])
        self.assertEqual(dampers[2], dampers[3])
        self.assertEqual((1.0, 1.0, 1.0, 1.0), _spring_scales(output))


if __name__ == "__main__":
    unittest.main()
