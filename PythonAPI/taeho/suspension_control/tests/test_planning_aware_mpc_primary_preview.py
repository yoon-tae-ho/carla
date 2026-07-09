"""CARLA-free checks for planning-aware v3 preview and guard helpers."""

from __future__ import annotations

import math
import unittest

from suspension_control.controllers.base import (
    ControllerContext,
    PlanningInfo,
    VehicleState,
)
from suspension_control.controllers.planning_aware_mpc_primary import (
    ComfortGuardState,
    PlanningAwareV3MpcPrimaryConfig,
    build_planning_preview,
)


def _state(
    frame=100,
    speed=10.0,
    elapsed_seconds=5.0,
    dt=0.05,
    local_ax=0.0,
    local_ay=0.0,
    yaw_rate=0.0,
):
    return VehicleState(
        frame=frame,
        speed=speed,
        elapsed_seconds=elapsed_seconds,
        dt=dt,
        local_ax=local_ax,
        local_ay=local_ay,
        yaw_rate=yaw_rate)


def _context(state=None, planning=None, dt=0.05):
    return ControllerContext(
        state=state if state is not None else _state(dt=dt),
        planning=planning if planning is not None else PlanningInfo.empty(),
        dt=dt)


def _planning(**overrides):
    values = {
        "available": True,
        "source": "unit",
        "frame": 100,
        "horizon_dt": 0.1,
        "target_speed": (10.0, 10.0, 10.0),
        "curvature": (0.0, 0.0, 0.0),
        "metadata": {"valid_prediction": True},
    }
    values.update(overrides)
    return PlanningInfo(**values)


def _assert_finite_sequence(test_case, values):
    for value in values:
        test_case.assertTrue(math.isfinite(value))


class PlanningAwareV3PreviewBuilderTest(unittest.TestCase):

    def test_empty_preview_reports_unavailable_reason(self):
        preview = build_planning_preview(_context(planning=PlanningInfo.empty()))

        self.assertFalse(preview.valid)
        self.assertEqual("planning_unavailable", preview.fallback_reason)
        self.assertEqual(
            0,
            preview.diagnostics["planning_aware_v3_valid"])

    def test_stale_preview_reports_stale_reason(self):
        preview = build_planning_preview(_context(
            state=_state(frame=100),
            planning=_planning(frame=94)))

        self.assertFalse(preview.valid)
        self.assertEqual("stale_planning", preview.fallback_reason)
        self.assertEqual(6, preview.planning_age_frames)

    def test_valid_direct_predicted_ay_uses_planning_source(self):
        preview = build_planning_preview(_context(planning=_planning(
            predicted_ay=(1.0, 2.0, 3.0),
            curvature=(0.2, 0.2, 0.2))))

        self.assertTrue(preview.valid)
        self.assertEqual("planning", preview.predicted_ay_source)
        self.assertAlmostEqual(1.0, preview.predicted_ay[0])
        self.assertAlmostEqual(1.5, preview.predicted_ay[1])

    def test_missing_predicted_ay_uses_v2kappa_source(self):
        preview = build_planning_preview(_context(planning=_planning(
            target_speed=(8.0, 8.0, 8.0),
            curvature=(0.05, 0.05, 0.05))))

        self.assertTrue(preview.valid)
        self.assertEqual("v2kappa", preview.predicted_ay_source)
        self.assertAlmostEqual(3.2, preview.predicted_ay[0])

    def test_target_speed_sequence_is_resampled(self):
        preview = build_planning_preview(_context(
            state=_state(speed=10.0),
            planning=_planning(target_speed=(10.0, 8.0, 6.0))))

        self.assertTrue(preview.valid)
        self.assertEqual("target_speed_sequence", preview.speed_profile_source)
        self.assertAlmostEqual(10.0, preview.speed_profile[0])
        self.assertAlmostEqual(9.0, preview.speed_profile[1])
        self.assertAlmostEqual(8.0, preview.speed_profile[2])

    def test_length_one_target_speed_builds_ramp_not_flat_repeat(self):
        preview = build_planning_preview(_context(
            state=_state(speed=4.0),
            planning=_planning(target_speed=(10.0,), curvature=(0.01,))))

        self.assertTrue(preview.valid)
        self.assertEqual("target_speed_ramp", preview.speed_profile_source)
        self.assertAlmostEqual(4.0, preview.speed_profile[0])
        self.assertGreater(preview.speed_profile[5], preview.speed_profile[0])
        self.assertLess(preview.speed_profile[5], 10.0)
        self.assertNotEqual(
            tuple(10.0 for _ in preview.speed_profile),
            preview.speed_profile)

    def test_brake_with_length_one_target_speed_synthesizes_deceleration(self):
        preview = build_planning_preview(_context(
            state=_state(speed=10.0),
            planning=_planning(
                target_speed=(10.0,),
                curvature=(0.0,),
                brake=(0.8,))))

        self.assertTrue(preview.valid)
        self.assertEqual("synthetic_control", preview.predicted_ax_source)
        self.assertLess(min(preview.predicted_ax), -1.0)
        self.assertLess(preview.speed_profile[4], preview.speed_profile[0])

    def test_direct_predicted_ax_wins_over_synthetic_ax(self):
        preview = build_planning_preview(_context(planning=_planning(
            predicted_ax=(1.5, 1.5),
            brake=(1.0, 1.0))))

        self.assertTrue(preview.valid)
        self.assertEqual("planning", preview.predicted_ax_source)
        self.assertAlmostEqual(1.5, preview.predicted_ax[0])

    def test_missing_predicted_ax_uses_speed_finite_difference_when_possible(self):
        preview = build_planning_preview(_context(
            state=_state(speed=10.0),
            planning=_planning(target_speed=(10.0, 8.0, 8.0))))

        self.assertTrue(preview.valid)
        self.assertEqual("speed_finite_difference", preview.predicted_ax_source)
        self.assertLess(min(preview.predicted_ax), -1.0)

    def test_synthetic_ax_works_from_controls_without_target_speed(self):
        preview = build_planning_preview(_context(
            state=_state(speed=12.0),
            planning=_planning(
                target_speed=(),
                curvature=(),
                brake=(0.5,),
                throttle=(0.0,))))

        self.assertTrue(preview.valid)
        self.assertEqual("synthetic_control", preview.predicted_ax_source)
        self.assertLess(preview.predicted_ax[0], -2.0)

    def test_yaw_rate_and_yaw_acc_peaks_are_finite(self):
        preview = build_planning_preview(_context(planning=_planning(
            target_speed=(12.0, 12.0, 12.0),
            curvature=(0.05, -0.05, 0.02))))
        diagnostics = preview.diagnostics

        self.assertTrue(preview.valid)
        _assert_finite_sequence(self, preview.yaw_rate_ref)
        _assert_finite_sequence(self, preview.yaw_acc_ref)
        self.assertTrue(math.isfinite(diagnostics["predicted_yaw_rate_peak"]))
        self.assertTrue(math.isfinite(diagnostics["predicted_yaw_acc_peak"]))

    def test_direct_nonfinite_input_produces_invalid_finite_preview(self):
        preview = build_planning_preview(_context(planning=_planning(
            predicted_ay=(float("nan"),),
            curvature=(0.01,))))
        diagnostics = preview.diagnostics

        self.assertFalse(preview.valid)
        self.assertEqual("nonfinite_preview_value", preview.fallback_reason)
        for key, value in diagnostics.items():
            if isinstance(value, float):
                self.assertTrue(math.isfinite(value), key)


class ComfortGuardStateTest(unittest.TestCase):

    def test_comfort_guard_margins_are_finite(self):
        guard = ComfortGuardState()
        result = None
        for index in range(4):
            result = guard.update(_context(state=_state(
                elapsed_seconds=5.0 + index * 0.05,
                local_ax=0.1,
                local_ay=0.2,
                yaw_rate=2.0)))

        self.assertIsNotNone(result)
        diagnostics = result.diagnostics
        for key, value in diagnostics.items():
            if isinstance(value, float):
                self.assertTrue(math.isfinite(value), key)

    def test_comfort_guard_activates_when_jerk_and_yaw_margins_are_small(self):
        guard = ComfortGuardState()
        config = PlanningAwareV3MpcPrimaryConfig(comfort_filter_tau_s=0.0)
        guard.update(_context(state=_state(
            elapsed_seconds=5.0,
            local_ax=0.0,
            local_ay=0.0,
            yaw_rate=0.0)))
        result = guard.update(_context(state=_state(
            elapsed_seconds=5.05,
            local_ax=10.0,
            local_ay=0.0,
            yaw_rate=120.0)), config)

        self.assertTrue(result.active)
        self.assertLess(result.lon_jerk_margin, 0.0)
        self.assertLess(result.yaw_rate_margin, 0.0)
        self.assertGreater(result.slew_penalty_multiplier, 1.0)
        self.assertGreater(result.mean_damper_penalty_multiplier, 1.0)


if __name__ == "__main__":
    unittest.main()
