"""CARLA-free checks for the planning-aware v3 MPC-primary controller."""

from __future__ import annotations

import math
import unittest

from suspension_control.controllers.base import (
    ControllerContext,
    PlanningInfo,
    VehicleState,
)
from suspension_control.controllers.planning_aware_mpc_primary import (
    PLANNING_AWARE_V3_MPC_PRIMARY_VERSION,
    PLANNING_AWARE_V3_REQUIRED_DIAGNOSTIC_FIELDS,
    PlanningAwareV3MpcPrimaryConfig,
    PlanningAwareV3MpcPrimaryController,
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


class PlanningAwareV3MpcPrimaryControllerTest(unittest.TestCase):

    def test_direct_constructibility_and_version(self):
        config = PlanningAwareV3MpcPrimaryConfig.from_mapping({})
        controller = PlanningAwareV3MpcPrimaryController(config)

        self.assertEqual(
            PLANNING_AWARE_V3_MPC_PRIMARY_VERSION,
            config.controller_version)
        self.assertEqual(
            PLANNING_AWARE_V3_MPC_PRIMARY_VERSION,
            controller.config.controller_version)
        self.assertEqual("planning_aware_v3_mpc_primary", controller.name)
        self.assertTrue(controller.requires_suspension_state)

    def test_empty_preview_falls_back_to_skyhook_roll_v3(self):
        context = _context(planning=PlanningInfo.empty())
        expected = SkyhookRollV3CanonicalModalController().compute(context)
        actual = PlanningAwareV3MpcPrimaryController().compute(context)

        self.assertTrue(actual.command.is_close(expected.command))
        self.assertEqual(
            "skyhook_roll_v3_fallback",
            actual.diagnostics["planning_aware_v3_mode"])
        self.assertEqual(
            "planning_unavailable",
            actual.diagnostics["planning_aware_v3_fallback_reason"])
        self.assertEqual(_dampers(expected), tuple(
            actual.diagnostics["skyhook_roll_v3_shadow_damper_%s" % label]
            for label in _LABELS))

    def test_stale_preview_falls_back_to_skyhook_roll_v3(self):
        planning = _planning(frame=94)
        context = _context(state=_state(frame=100), planning=planning)
        expected = SkyhookRollV3CanonicalModalController().compute(context)
        actual = PlanningAwareV3MpcPrimaryController().compute(context)

        self.assertTrue(actual.command.is_close(expected.command))
        self.assertEqual(
            "stale_planning",
            actual.diagnostics["planning_aware_v3_fallback_reason"])

    def test_fallback_invalid_shadow_returns_identity_safe_command(self):
        context = _context(
            planning=PlanningInfo.empty(),
            suspension_state_valid=False,
            invalid_reason="strict_state_missing")
        output = PlanningAwareV3MpcPrimaryController().compute(context)

        _assert_finite_command(self, output)
        self.assertEqual((1.0, 1.0, 1.0, 1.0), _dampers(output))
        self.assertEqual((1.0, 1.0, 1.0, 1.0), _springs(output))
        self.assertEqual(
            "identity_fallback",
            output.diagnostics["planning_aware_v3_mode"])
        self.assertIn(
            "skyhook_roll_v3_invalid:strict_state_missing",
            output.diagnostics["planning_aware_v3_fallback_reason"])

    def test_zero_calm_preview_returns_near_neutral_springs_identity(self):
        output = PlanningAwareV3MpcPrimaryController().compute(
            _context(planning=_planning()))

        _assert_finite_command(self, output)
        self.assertEqual((1.0, 1.0, 1.0, 1.0), _springs(output))
        for damper in _dampers(output):
            self.assertAlmostEqual(1.0, damper, delta=0.02)
        self.assertEqual("mpc_primary", output.diagnostics[
            "planning_aware_v3_mode"])

    def test_lateral_preview_produces_roll_differential(self):
        state = _state(roll_rate=8.0, local_ay=5.0)
        output = PlanningAwareV3MpcPrimaryController().compute(
            _context(state=state, planning=_lateral_planning(ay=5.0)))
        dampers = _dampers(output)

        _assert_finite_command(self, output)
        self.assertGreater(dampers[0], dampers[1])
        self.assertGreater(dampers[2], dampers[3])
        self.assertGreater(output.diagnostics["front_roll_contrast"], 0.0)
        self.assertGreater(output.diagnostics["rear_roll_contrast"], 0.0)

    def test_braking_preview_synthetic_ax_records_decel_cost(self):
        planning = _planning(
            target_speed=(4.0,),
            predicted_ax=(),
            predicted_ay=(0.0,),
            curvature=(0.0,),
            brake=(0.8,))
        output = PlanningAwareV3MpcPrimaryController().compute(
            _context(state=_state(speed=12.0), planning=planning))

        self.assertEqual(
            "synthetic_control",
            output.diagnostics["predicted_ax_source"])
        self.assertLess(output.diagnostics["predicted_ax_min"], -1.0)
        self.assertGreater(output.diagnostics["mpc_b2d_threshold_cost"], 0.0)

    def test_rate_constraint_at_005_and_010(self):
        for dt in (0.05, 0.10):
            with self.subTest(dt=dt):
                config = PlanningAwareV3MpcPrimaryConfig(default_dt=dt)
                controller = PlanningAwareV3MpcPrimaryController(config)
                state = _state(dt=dt, roll_rate=8.0, local_ay=5.0)
                output = controller.compute(_context(
                    state=state,
                    planning=_lateral_planning(ay=5.0),
                    dt=dt))
                limit = config.max_rate_up_scale_per_s * dt + 1.0e-12

                self.assertEqual(1, output.diagnostics[
                    "mpc_rate_constraint_active"])
                for damper in _dampers(output):
                    self.assertLessEqual(abs(damper - 1.0), limit)

    def test_nonfinite_direct_planning_input_never_outputs_nonfinite(self):
        planning = _planning(predicted_ay=(float("nan"),), curvature=(0.01,))
        output = PlanningAwareV3MpcPrimaryController().compute(
            _context(planning=planning))

        _assert_finite_command(self, output)
        self.assertIn(
            "nonfinite_preview_value",
            output.diagnostics["planning_aware_v3_fallback_reason"])

    def test_comfort_guard_small_jerk_margin_increases_slew_penalty(self):
        config = PlanningAwareV3MpcPrimaryConfig(comfort_filter_tau_s=0.0)
        controller = PlanningAwareV3MpcPrimaryController(config)
        controller.compute(_context(
            state=_state(elapsed_seconds=5.0, local_ax=0.0),
            planning=_planning()))
        output = controller.compute(_context(
            state=_state(elapsed_seconds=5.05, local_ax=10.0),
            planning=_planning()))

        self.assertEqual(1, output.diagnostics["comfort_guard_active"])
        self.assertLess(
            output.diagnostics["comfort_guard_lon_jerk_margin"],
            0.0)
        self.assertGreater(
            output.diagnostics["mpc_slew_weight_scheduled"],
            config.W_slew)

    def test_ltr_proxy_is_finite_and_soft_guard_can_activate(self):
        output = PlanningAwareV3MpcPrimaryController().compute(
            _context(planning=_lateral_planning(ay=7.0)))

        self.assertTrue(math.isfinite(output.diagnostics[
            "mpc_ltr_proxy_peak"]))
        self.assertEqual(1, output.diagnostics["mpc_ltr_soft_guard_active"])
        self.assertEqual(0, output.diagnostics["mpc_ltr_hard_guard_active"])

    def test_wheel_order_is_fl_fr_rl_rr(self):
        output = PlanningAwareV3MpcPrimaryController().compute(
            _context(planning=_lateral_planning(ay=5.0)))
        dampers = _dampers(output)

        self.assertEqual(dampers[0], output.diagnostics["final_damper_fl"])
        self.assertEqual(dampers[1], output.diagnostics["final_damper_fr"])
        self.assertEqual(dampers[2], output.diagnostics["final_damper_rl"])
        self.assertEqual(dampers[3], output.diagnostics["final_damper_rr"])

    def test_all_required_diagnostics_fields_exist(self):
        output = PlanningAwareV3MpcPrimaryController().compute(
            _context(planning=_lateral_planning(ay=5.0)))

        for key in PLANNING_AWARE_V3_REQUIRED_DIAGNOSTIC_FIELDS:
            self.assertIn(key, output.diagnostics)

    def test_skyhook_prior_false_leaves_prior_penalty_inactive(self):
        output = PlanningAwareV3MpcPrimaryController().compute(
            _context(planning=_lateral_planning(ay=5.0)))

        self.assertEqual(0.0, output.diagnostics["skyhook_prior_penalty"])

    def test_skyhook_prior_true_records_nonnegative_prior_penalty(self):
        config = PlanningAwareV3MpcPrimaryConfig(
            use_skyhook_prior=True,
            W_prior=0.05)
        output = PlanningAwareV3MpcPrimaryController(config).compute(
            _context(planning=_lateral_planning(ay=5.0)))

        self.assertGreaterEqual(
            output.diagnostics["skyhook_prior_penalty"],
            0.0)


if __name__ == "__main__":
    unittest.main()
