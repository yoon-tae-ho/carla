"""CARLA-free checks for the planning-aware v4 Phase A controller."""

from __future__ import annotations

import math
import unittest

from suspension_control.controllers.base import (
    ControllerContext,
    PlanningInfo,
    VehicleState,
)
from suspension_control.controllers.planning_aware_mpc_phaseA import (
    PLANNING_AWARE_V4_MPC_PHASEA_VERSION,
    PlanningAwareV4MpcPhaseAConfig,
    PlanningAwareV4MpcPhaseAController,
)
from suspension_control.controllers.planning_aware_mpc_primary import (
    build_planning_preview,
)
from suspension_control.controllers.skyhook_roll_v3 import (
    SkyhookRollV3CanonicalModalController,
)


_LABELS = ("fl", "fr", "rl", "rr")


class _FakeWheel:

    def __init__(
        self,
        index,
        velocity,
        field_valid=True,
        velocity_valid=True,
        contact_valid=True,
        wheel_in_air=False,
    ):
        self.wheel_index_raw = index
        self.wheel_name_canonical = ("FL", "FR", "RL", "RR")[index]
        self.raw_suspension_offset_m = 0.01
        self.suspension_compression_m = 0.01
        self.suspension_travel_m = 0.01
        self.suspension_velocity_mps = velocity
        self.normalized_travel = 0.5
        self.contact_valid = contact_valid
        self.wheel_in_air = wheel_in_air
        self.field_valid = field_valid
        self.velocity_valid = velocity_valid


class _FakeSuspensionState:

    def __init__(
        self,
        velocities=(-0.5, 0.5, -0.5, 0.5),
        state_valid=True,
        velocity_valid=True,
        failure_reason="",
        **wheel_kwargs
    ):
        self.state_valid = state_valid
        self.velocity_valid = velocity_valid
        self.compression_convention_validated = True
        self.state_source = "fake"
        self.failure_reason = failure_reason
        self.wheels = tuple(
            _FakeWheel(index, velocity, **wheel_kwargs)
            for index, velocity in enumerate(velocities))


class _FakeNativeWheel:
    spring_strength = 35000.0
    spring_damper_rate = 4500.0


class _FakeNativeSuspension:
    wheels = (_FakeNativeWheel(),) * 4


def _state(
    frame=100,
    speed=12.0,
    elapsed_seconds=5.0,
    dt=0.05,
    vz=0.0,
    roll=0.0,
    pitch=0.0,
    roll_rate=0.0,
    pitch_rate=0.0,
    yaw_rate=0.0,
    local_ax=0.0,
    local_ay=0.0,
):
    return VehicleState(
        frame=frame,
        speed=speed,
        elapsed_seconds=elapsed_seconds,
        dt=dt,
        vz=vz,
        roll=roll,
        pitch=pitch,
        roll_rate=roll_rate,
        pitch_rate=pitch_rate,
        yaw_rate=yaw_rate,
        local_ax=local_ax,
        local_ay=local_ay)


def _context(
    state=None,
    planning=None,
    suspension_state=None,
    suspension_state_valid=True,
    invalid_reason="",
    native_dampers=(4500.0, 4500.0, 4500.0, 4500.0),
    dt=0.05,
):
    state_value = state if state is not None else _state(dt=dt)
    return ControllerContext(
        state=state_value,
        planning=planning if planning is not None else PlanningInfo.empty(),
        native_suspension=_FakeNativeSuspension(),
        current_suspension=_FakeNativeSuspension(),
        suspension_state=(
            suspension_state
            if suspension_state is not None else _FakeSuspensionState()),
        suspension_state_valid=suspension_state_valid,
        suspension_state_invalid_reason=invalid_reason,
        native_spring_strength_by_wheel=(35000.0,) * 4,
        native_damper_rate_by_wheel=native_dampers,
        dt=dt)


def _planning(**overrides):
    values = {
        "available": True,
        "source": "unit",
        "frame": 100,
        "horizon_dt": 0.1,
        "target_speed": (12.0, 12.0, 12.0, 12.0),
        "curvature": (0.0, 0.0, 0.0, 0.0),
        "predicted_ax": (0.0, 0.0, 0.0, 0.0),
        "predicted_ay": (0.0, 0.0, 0.0, 0.0),
        "brake": (0.0, 0.0, 0.0, 0.0),
        "throttle": (0.0, 0.0, 0.0, 0.0),
        "metadata": {"valid_prediction": True},
    }
    values.update(overrides)
    return PlanningInfo(**values)


def _lateral_planning(ay=5.0):
    return _planning(
        curvature=(0.03, 0.03, 0.03, 0.03),
        predicted_ay=(ay, ay, ay, ay),
        predicted_ax=(0.0, 0.0, 0.0, 0.0))


def _dampers(output):
    return tuple(wheel.damper_scale for wheel in output.command.wheels)


def _springs(output):
    return tuple(wheel.spring_scale for wheel in output.command.wheels)


def _assert_finite_command(test_case, output):
    for wheel in output.command.wheels:
        test_case.assertTrue(math.isfinite(wheel.spring_scale))
        test_case.assertTrue(math.isfinite(wheel.damper_scale))
        test_case.assertGreater(wheel.spring_scale, 0.0)
        test_case.assertGreater(wheel.damper_scale, 0.0)


def _raw_optimizer(controller, context):
    shadow_output = controller.skyhook_roll_v3_shadow.compute(context)
    shadow_dampers = controller._command_dampers(
        shadow_output.command,
        4,
        default=1.0)
    preview = build_planning_preview(context, controller.config)
    comfort = controller.comfort_guard.update(context, controller.config)
    return controller._generate_v3_raw_mpc_grid_candidates(
        context=context,
        preview=preview,
        comfort=comfort,
        shadow_dampers=shadow_dampers,
        top_k=controller._projection_top_k())


class PlanningAwareV4MpcPhaseAControllerTest(unittest.TestCase):

    def test_direct_constructibility_and_authority_version(self):
        config = PlanningAwareV4MpcPhaseAConfig.from_mapping({})
        controller = PlanningAwareV4MpcPhaseAController(config)

        self.assertEqual(
            PLANNING_AWARE_V4_MPC_PHASEA_VERSION,
            config.controller_version)
        self.assertEqual(
            PLANNING_AWARE_V4_MPC_PHASEA_VERSION,
            controller.config.controller_version)
        self.assertEqual(
            "planning_aware_v4_mpc_phaseA_authority",
            controller.name)
        self.assertTrue(controller.requires_suspension_state)
        self.assertEqual("authority", controller.config.base_profile)
        self.assertEqual(2.0, controller.config.max_rate_up_scale_per_s)
        self.assertEqual(1.6, controller.config.max_rate_down_scale_per_s)

    def test_first_tick_previous_candidate_uses_identity_and_can_win(self):
        output = PlanningAwareV4MpcPhaseAController().compute(
            _context(planning=_planning()))

        self.assertEqual("previous_hold", output.diagnostics[
            "v4_selected_candidate_kind"])
        self.assertEqual(1, output.diagnostics["v4_selected_from_previous"])
        self.assertEqual(1, output.diagnostics["v4_projection_changed_winner"])
        for label in _LABELS:
            self.assertAlmostEqual(
                1.0,
                output.diagnostics["v4_selected_raw_damper_%s" % label])
            self.assertAlmostEqual(
                1.0,
                output.diagnostics["v4_selected_final_damper_%s" % label])

    def test_previous_and_neutral_candidate_raw_dampers_are_explicit(self):
        controller = PlanningAwareV4MpcPhaseAController()
        controller.previous_final_damper_scales = [1.04, 1.03, 1.02, 1.01]
        context = _context(planning=_planning())
        optimizer = _raw_optimizer(controller, context)
        candidates = controller._build_projection_rerank_candidates(optimizer, 4)
        previous = controller._first_candidate_of_kind(candidates, "previous_hold")
        neutral = controller._first_candidate_of_kind(candidates, "neutral_return")

        self.assertIsNotNone(previous)
        self.assertIsNotNone(neutral)
        self.assertEqual(
            tuple(controller.previous_final_damper_scales),
            previous.raw_dampers)
        self.assertEqual((1.0, 1.0, 1.0, 1.0), neutral.raw_dampers)

    def test_valid_planning_rerank_set_includes_raw_previous_and_neutral(self):
        output = PlanningAwareV4MpcPhaseAController().compute(
            _context(planning=_planning()))

        self.assertEqual(1, output.diagnostics["v4_previous_candidate_included"])
        self.assertEqual(1, output.diagnostics["v4_neutral_candidate_included"])
        self.assertEqual("mpc_grid", output.diagnostics["v4_raw_winner_kind"])
        self.assertGreaterEqual(
            output.diagnostics["v4_rerank_candidate_count"],
            3)
        self.assertTrue(math.isfinite(output.diagnostics[
            "v4_raw_winner_projected_cost"]))

    def test_neutral_can_win_when_previous_is_far_and_preview_is_calm(self):
        config = PlanningAwareV4MpcPhaseAConfig(W_slew=0.0)
        controller = PlanningAwareV4MpcPhaseAController(config)
        controller.previous_final_damper_scales = [1.08, 1.08, 1.08, 1.08]
        output = controller.compute(_context(planning=_planning()))

        self.assertEqual("neutral_return", output.diagnostics[
            "v4_selected_candidate_kind"])
        self.assertEqual(1, output.diagnostics["v4_selected_from_neutral"])
        for damper in _dampers(output):
            self.assertLessEqual(damper, 1.08)

    def test_mpc_grid_can_still_win_under_strong_lateral_preview(self):
        state = _state(roll_rate=8.0, local_ay=5.0)
        output = PlanningAwareV4MpcPhaseAController().compute(
            _context(state=state, planning=_lateral_planning(ay=5.0)))
        dampers = _dampers(output)

        self.assertEqual("mpc_grid", output.diagnostics[
            "v4_selected_candidate_kind"])
        self.assertEqual(1, output.diagnostics["v4_selected_from_mpc_grid"])
        self.assertGreater(dampers[0], dampers[1])
        self.assertGreater(dampers[2], dampers[3])

    def test_final_command_is_finite_bounded_spring_frozen_and_rate_limited(self):
        controller = PlanningAwareV4MpcPhaseAController()
        controller.previous_final_damper_scales = [1.02, 0.98, 1.01, 0.99]
        previous = tuple(controller.previous_final_damper_scales)
        context = _context(
            state=_state(roll_rate=8.0, local_ay=5.0),
            planning=_lateral_planning(ay=5.0))
        output = controller.compute(context)

        _assert_finite_command(self, output)
        self.assertEqual((1.0, 1.0, 1.0, 1.0), _springs(output))
        max_up = controller.config.max_rate_up_scale_per_s * context.dt
        max_down = controller.config.max_rate_down_scale_per_s * context.dt
        for index, damper in enumerate(_dampers(output)):
            self.assertGreaterEqual(damper, controller.config.min_damper_scale)
            self.assertLessEqual(damper, controller.config.max_damper_scale)
            self.assertLessEqual(damper - previous[index], max_up + 1.0e-12)
            self.assertLessEqual(previous[index] - damper, max_down + 1.0e-12)

    def test_stale_and_nonfinite_planning_fallbacks_remain_v3_compatible(self):
        stale_context = _context(
            state=_state(frame=100),
            planning=_planning(frame=94))
        expected = SkyhookRollV3CanonicalModalController().compute(stale_context)
        stale = PlanningAwareV4MpcPhaseAController().compute(stale_context)

        self.assertTrue(stale.command.is_close(expected.command))
        self.assertEqual(
            "skyhook_roll_v3_fallback",
            stale.diagnostics["planning_aware_v3_mode"])
        self.assertEqual(
            "stale_planning",
            stale.diagnostics["planning_aware_v4_fallback_reason"])

        nonfinite = PlanningAwareV4MpcPhaseAController().compute(_context(
            planning=_planning(predicted_ay=(float("nan"),), curvature=(0.01,))))
        _assert_finite_command(self, nonfinite)
        self.assertIn(
            "nonfinite_preview_value",
            nonfinite.diagnostics["planning_aware_v4_fallback_reason"])

    def test_compute_time_diagnostics_exist_and_default_debug_sleep_is_zero(self):
        output = PlanningAwareV4MpcPhaseAController().compute(
            _context(planning=_planning()))

        for key in (
                "controller_compute_ms",
                "controller_compute_ms_raw_optimizer",
                "controller_compute_ms_projection_rerank",
                "controller_compute_ms_projection",
                "controller_compute_ms_final_guard",
                "debug_compute_sleep_ms"):
            self.assertIn(key, output.diagnostics)
            self.assertTrue(math.isfinite(output.diagnostics[key]), key)
            self.assertGreaterEqual(output.diagnostics[key], 0.0)
        self.assertEqual(0.0, output.diagnostics["debug_compute_sleep_ms"])


if __name__ == "__main__":
    unittest.main()
