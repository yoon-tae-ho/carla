"""CARLA-free checks for skyhook_roll_v3 canonical modal roll-rate damping."""

from __future__ import annotations

from dataclasses import fields
import math
import unittest

from suspension_control.controllers.base import (
    ControllerContext,
    PlanningInfo,
    VehicleState,
)
from suspension_control.controllers.skyhook import (
    SkyhookConfig,
    SkyhookController,
)
from suspension_control.controllers.skyhook_roll_v3 import (
    SKYHOOK_ROLL_V3_CANONICAL_MODAL_VERSION,
    SkyhookRollV3CanonicalModalConfig,
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
        velocities=(-0.5, -0.5, -0.5, -0.5),
        velocity_valid=True,
        **wheel_kwargs
    ):
        self.state_valid = True
        self.velocity_valid = velocity_valid
        self.compression_convention_validated = True
        self.state_source = "fake"
        self.failure_reason = ""
        self.wheels = tuple(
            _FakeWheel(index, velocity, **wheel_kwargs)
            for index, velocity in enumerate(velocities))


class _FakeNativeWheel:
    spring_strength = 35000.0
    spring_damper_rate = 4500.0


class _FakeNativeSuspension:
    wheels = (_FakeNativeWheel(),) * 4


def _context(
    state=None,
    previous_state=None,
    planning=None,
    suspension_state=None,
    suspension_state_valid=True,
    invalid_reason="",
    native_dampers=(4500.0, 4500.0, 4500.0, 4500.0),
    dt=0.05,
):
    return ControllerContext(
        state=state if state is not None else VehicleState(vz=1.0, dt=dt),
        previous_state=previous_state,
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


def _config(**overrides):
    values = {
        "max_damper_delta_per_step": 10.0,
        "sprung_velocity_deadband": 0.0,
        "rel_velocity_deadband": 0.0,
        "product_deadband": 0.0,
    }
    values.update(overrides)
    return SkyhookRollV3CanonicalModalConfig(**values)


def _skyhook_config_from_v3(config):
    allowed = {item.name for item in fields(SkyhookConfig)}
    values = {
        name: getattr(config, name)
        for name in allowed
        if hasattr(config, name)
    }
    return SkyhookConfig(**values)


def _springs(output):
    return tuple(wheel.spring_scale for wheel in output.command.wheels)


def _dampers(output):
    return tuple(wheel.damper_scale for wheel in output.command.wheels)


def _wheel_values(diagnostics, prefix):
    return tuple(diagnostics["%s_%s" % (prefix, label)] for label in _LABELS)


def _rich_planning():
    return PlanningInfo(
        available=True,
        source="unit",
        frame=40,
        horizon_dt=0.1,
        target_speed=(25.0, 10.0, 5.0),
        curvature=(0.10, -0.15, 0.08),
        steer=(0.8, -0.6, 0.4),
        throttle=(1.0, 0.0, 0.0),
        brake=(0.0, 0.9, 0.7),
        predicted_ax=(-3.0, -4.0, -2.0),
        predicted_ay=(7.0, -8.0, 5.0),
        metadata={"route": "ignored"})


class SkyhookRollV3CanonicalModalControllerTest(unittest.TestCase):

    def test_direct_constructibility_and_version(self):
        config = SkyhookRollV3CanonicalModalConfig()
        controller = SkyhookRollV3CanonicalModalController(config)

        self.assertEqual(
            SKYHOOK_ROLL_V3_CANONICAL_MODAL_VERSION,
            config.controller_version)
        self.assertEqual(
            SKYHOOK_ROLL_V3_CANONICAL_MODAL_VERSION,
            controller.config.controller_version)
        self.assertEqual("skyhook_roll_v3", controller.name)

    def test_config_accepts_aliases_and_ignores_legacy_roll_knobs(self):
        base = SkyhookRollV3CanonicalModalConfig.from_mapping({
            "half_track": 0.9,
            "half_wheelbase": 1.6,
            "relative_velocity_deadband": 0.02,
        })
        legacy = SkyhookRollV3CanonicalModalConfig.from_mapping({
            "half_track": 0.9,
            "half_wheelbase": 1.6,
            "relative_velocity_deadband": 0.02,
            "roll_damper_enabled": False,
            "roll_c_scale": 99.0,
            "max_roll_extra_scale": 99.0,
            "roll_softening_override_limit": 99.0,
            "roll_ay_on": 99.0,
            "roll_ay_full": 100.0,
            "outer_side_bias": 99.0,
            "nominal_front_roll_distribution": 0.99,
        })
        state = VehicleState(vz=0.7, roll_rate=18.0, local_ay=6.0)
        suspension_state = _FakeSuspensionState(
            velocities=(-0.5, 0.4, -0.3, 0.2))

        base_output = SkyhookRollV3CanonicalModalController(base).compute(
            _context(state=state, suspension_state=suspension_state))
        legacy_output = SkyhookRollV3CanonicalModalController(legacy).compute(
            _context(state=state, suspension_state=suspension_state))

        self.assertEqual(0.9, legacy.half_track_m)
        self.assertEqual(1.6, legacy.half_wheelbase_m)
        self.assertEqual(0.02, legacy.rel_velocity_deadband)
        self.assertEqual(_dampers(base_output), _dampers(legacy_output))
        self.assertEqual(_springs(base_output), _springs(legacy_output))

    def test_spring_scale_is_always_identity_even_with_spring_knobs(self):
        controller = SkyhookRollV3CanonicalModalController(_config(
            base_spring_scale=2.0,
            min_spring_scale=0.5,
            max_spring_scale=3.0,
            max_spring_delta_per_step=10.0,
            roll_spring_enabled=True,
            enable_roll_spring_control=True,
            roll_modal_c_scale=2.0))

        output = controller.compute(_context(
            state=VehicleState(vz=0.6, roll_rate=25.0),
            suspension_state=_FakeSuspensionState(
                velocities=(-0.5, 0.5, -0.5, 0.5))))

        self.assertEqual((1.0, 1.0, 1.0, 1.0), _springs(output))
        self.assertEqual(0, output.diagnostics["spring_used_in_command"])
        for label in _LABELS:
            self.assertEqual(
                1.0,
                output.diagnostics["final_spring_scale_%s" % label])

    def test_disabled_zero_scale_and_zero_roll_rate_equal_skyhook(self):
        cases = (
            ("disabled", {"roll_modal_enabled": False},
             VehicleState(vz=0.6, roll_rate=18.0, pitch_rate=4.0)),
            ("zero_scale", {"roll_modal_c_scale": 0.0},
             VehicleState(vz=0.6, roll_rate=18.0, pitch_rate=4.0)),
            ("zero_roll_rate", {},
             VehicleState(vz=0.6, roll_rate=0.0, pitch_rate=4.0)),
        )
        suspension_state = _FakeSuspensionState(
            velocities=(-0.5, 0.4, -0.3, 0.2))
        native_dampers = (5000.0, 4500.0, 4000.0, 3500.0)

        for name, overrides, state in cases:
            with self.subTest(name=name):
                config = _config(**overrides)
                v3 = SkyhookRollV3CanonicalModalController(config)
                skyhook = SkyhookController(_skyhook_config_from_v3(config))
                context = _context(
                    state=state,
                    suspension_state=suspension_state,
                    native_dampers=native_dampers)

                v3_output = v3.compute(context)
                skyhook_output = skyhook.compute(context)

                self.assertEqual(_springs(skyhook_output), _springs(v3_output))
                self.assertEqual(_dampers(skyhook_output), _dampers(v3_output))
                self.assertEqual((0.0, 0.0, 0.0, 0.0), _wheel_values(
                    v3_output.diagnostics,
                    "F_roll_modal"))

    def test_forbidden_state_and_preview_inputs_do_not_change_commands(self):
        config = _config(
            enable_yaw_distribution=True,
            use_planning_preview=True,
            use_lateral_accel_gate=True,
            use_roll_angle_feedback=True,
            use_outer_side_bias=True,
            use_front_rear_roll_distribution=True)
        base_state = VehicleState(
            vz=0.5,
            roll=0.0,
            roll_rate=22.0,
            pitch_rate=3.0,
            yaw=0.0,
            yaw_rate=0.0,
            local_ay=0.0,
            steer=0.0)
        changed_state = VehicleState(
            vz=0.5,
            roll=45.0,
            roll_rate=22.0,
            pitch_rate=3.0,
            yaw=160.0,
            yaw_rate=90.0,
            local_ay=8.0,
            steer=-0.9)
        suspension_state = _FakeSuspensionState(
            velocities=(-0.4, 0.3, -0.2, 0.1))

        base_output = SkyhookRollV3CanonicalModalController(config).compute(
            _context(
                state=base_state,
                planning=PlanningInfo.empty(),
                suspension_state=suspension_state))
        changed_output = SkyhookRollV3CanonicalModalController(config).compute(
            _context(
                state=changed_state,
                planning=_rich_planning(),
                suspension_state=suspension_state))

        self.assertEqual(_springs(base_output), _springs(changed_output))
        self.assertEqual(_dampers(base_output), _dampers(changed_output))
        for flag in (
                "roll_angle_used_in_command",
                "local_ay_used_in_command",
                "yaw_used_in_command",
                "planning_preview_used_in_command",
                "spring_used_in_command"):
            self.assertEqual(0, changed_output.diagnostics[flag])

    def test_roll_modal_forces_have_geometry_min_norm_moments(self):
        config = _config(roll_modal_c_scale=0.20)
        positive = SkyhookRollV3CanonicalModalController(config).compute(
            _context(
                state=VehicleState(vz=0.0, roll_rate=20.0),
                suspension_state=_FakeSuspensionState(
                    velocities=(-0.5, 0.5, -0.5, 0.5))))
        negative = SkyhookRollV3CanonicalModalController(config).compute(
            _context(
                state=VehicleState(vz=0.0, roll_rate=-20.0),
                suspension_state=_FakeSuspensionState(
                    velocities=(-0.5, 0.5, -0.5, 0.5))))

        f_pos = _wheel_values(positive.diagnostics, "F_roll_modal")
        f_neg = _wheel_values(negative.diagnostics, "F_roll_modal")
        x_values = _wheel_values(positive.diagnostics, "x")
        y_values = _wheel_values(positive.diagnostics, "y")

        for left, right in zip(f_pos, f_neg):
            self.assertAlmostEqual(left, -right)
        self.assertAlmostEqual(0.0, sum(f_pos), places=9)
        self.assertAlmostEqual(
            0.0,
            sum(x_i * force for x_i, force in zip(x_values, f_pos)),
            places=9)
        self.assertAlmostEqual(
            positive.diagnostics["Q_roll_des"],
            sum(y_i * force for y_i, force in zip(y_values, f_pos)),
            places=9)
        self.assertAlmostEqual(abs(f_pos[0]), abs(f_pos[2]))
        self.assertAlmostEqual(abs(f_pos[1]), abs(f_pos[3]))
        self.assertAlmostEqual(f_pos[0], -f_pos[1])
        self.assertEqual(
            "geometry_min_norm",
            positive.diagnostics["roll_modal_distribution"])
        self.assertEqual(1, positive.diagnostics["roll_modal_valid"])

    def test_total_projection_branches_and_product_are_reported(self):
        cases = (
            (
                "neutral_sprung_deadband",
                _config(
                    roll_modal_enabled=False,
                    sprung_velocity_deadband=0.025,
                    rel_velocity_deadband=0.015),
                VehicleState(vz=0.001),
                _FakeSuspensionState(velocities=(-0.5,) * 4),
            ),
            (
                "neutral_rel_deadband",
                _config(
                    roll_modal_enabled=False,
                    sprung_velocity_deadband=0.025,
                    rel_velocity_deadband=0.015),
                VehicleState(vz=1.0),
                _FakeSuspensionState(velocities=(-0.001,) * 4),
            ),
            (
                "projected_feasible",
                _config(roll_modal_enabled=False),
                VehicleState(vz=1.0),
                _FakeSuspensionState(velocities=(-0.5,) * 4),
            ),
            (
                "soft_infeasible",
                _config(roll_modal_enabled=False),
                VehicleState(vz=1.0),
                _FakeSuspensionState(velocities=(0.5,) * 4),
            ),
        )

        for expected_branch, config, state, suspension_state in cases:
            with self.subTest(expected_branch=expected_branch):
                output = SkyhookRollV3CanonicalModalController(config).compute(
                    _context(state=state, suspension_state=suspension_state))

                self.assertEqual(
                    expected_branch,
                    output.diagnostics["total_target_branch_fl"])
                self.assertAlmostEqual(
                    output.diagnostics["v_eff_total_fl"] *
                    output.diagnostics["v_rel_extension_mps_fl"],
                    output.diagnostics["total_product_fl"])
                self.assertEqual(
                    output.diagnostics["total_target_damper_fl"],
                    output.diagnostics["final_target_damper_fl"])

    def test_total_target_can_drop_below_base_target_without_floor(self):
        controller = SkyhookRollV3CanonicalModalController(_config(
            roll_modal_c_scale=1.0))

        output = controller.compute(_context(
            state=VehicleState(vz=1.0, roll_rate=30.0),
            suspension_state=_FakeSuspensionState(velocities=(-0.5,) * 4)))

        self.assertGreater(
            output.diagnostics["base_target_damper_fl"],
            output.diagnostics["total_target_damper_fl"])
        self.assertEqual(
            output.diagnostics["total_target_damper_fl"],
            output.diagnostics["final_target_damper_fl"])
        self.assertEqual(
            output.diagnostics["total_target_damper_fl"],
            _dampers(output)[0])

    def test_invalid_state_fallback_returns_identity_and_resets_history(self):
        controller = SkyhookRollV3CanonicalModalController(_config(
            max_damper_delta_per_step=0.02,
            roll_modal_enabled=False))
        controller.compute(_context(
            state=VehicleState(vz=1.0),
            suspension_state=_FakeSuspensionState(velocities=(-0.5,) * 4)))
        self.assertEqual((1.02, 1.02, 1.02, 1.02), tuple(
            controller.previous_damper_scales))

        output = controller.compute(_context(
            suspension_state=_FakeSuspensionState(field_valid=False)))

        self.assertEqual((1.0, 1.0, 1.0, 1.0), _springs(output))
        self.assertEqual((1.0, 1.0, 1.0, 1.0), _dampers(output))
        self.assertEqual((1.0, 1.0, 1.0, 1.0), tuple(
            controller.previous_damper_scales))
        self.assertEqual(1, output.diagnostics["identity_fallback_this_tick"])
        self.assertEqual(
            "wheel_0_field_valid_false",
            output.diagnostics["identity_fallback_reason"])
        self.assertEqual(
            "identity_fallback",
            output.diagnostics["total_target_branch_fl"])

    def test_rate_limiter_uses_inherited_fixed_step_delta(self):
        controller = SkyhookRollV3CanonicalModalController(_config(
            max_damper_delta_per_step=0.02,
            roll_modal_enabled=False))

        output = controller.compute(_context(
            state=VehicleState(vz=1.0),
            suspension_state=_FakeSuspensionState(velocities=(-0.5,) * 4)))

        self.assertEqual((1.02, 1.02, 1.02, 1.02), _dampers(output))
        self.assertEqual(1, output.diagnostics["rate_limited_damper_fl"])
        self.assertEqual(
            0.02,
            output.diagnostics["max_damper_delta_per_step_used"])

    def test_required_diagnostics_are_present(self):
        output = SkyhookRollV3CanonicalModalController(_config()).compute(
            _context(
                state=VehicleState(
                    frame=20,
                    vz=0.8,
                    roll_rate=25.0,
                    pitch_rate=2.0),
                previous_state=VehicleState(frame=19),
                suspension_state=_FakeSuspensionState(
                    velocities=(-0.5, 0.4, -0.3, 0.2))))
        diagnostics = output.diagnostics
        scalar_fields = (
            "controller",
            "controller_name",
            "controller_version",
            "skyhook_roll_v3_mode",
            "roll_modal_enabled",
            "roll_modal_c_scale",
            "roll_modal_distribution",
            "roll_rate_rad_s",
            "roll_rate_deg_s",
            "C_phi",
            "Q_roll_des",
            "roll_distribution_denom",
            "roll_modal_valid",
            "roll_residual_heave_sum",
            "roll_residual_pitch_sum",
            "roll_residual_roll_moment",
            "roll_angle_used_in_command",
            "local_ay_used_in_command",
            "yaw_used_in_command",
            "planning_preview_used_in_command",
            "spring_used_in_command",
            "dt",
            "expected_dt",
            "frame_delta",
            "dt_gap_warning",
        )
        per_wheel_fields = (
            "x",
            "y",
            "C_native",
            "C_sky",
            "v_s_base",
            "v_rel_extension_mps",
            "F_sky_ideal",
            "F_roll_modal",
            "F_total_ideal",
            "v_eff_total",
            "base_target_damper",
            "total_raw_target_damper",
            "total_target_branch",
            "total_product",
            "total_required_scale_unclipped",
            "total_target_damper",
            "final_spring_scale",
            "final_damper_scale",
            "rate_limited_damper",
        )

        for field_name in scalar_fields:
            self.assertIn(field_name, diagnostics)
        for label in _LABELS:
            for field_name in per_wheel_fields:
                self.assertIn("%s_%s" % (field_name, label), diagnostics)
        self.assertEqual(
            SKYHOOK_ROLL_V3_CANONICAL_MODAL_VERSION,
            diagnostics["controller_version"])
        self.assertEqual("skyhook_roll_v3", diagnostics["controller_name"])
        for flag in (
                "roll_angle_used_in_command",
                "local_ay_used_in_command",
                "yaw_used_in_command",
                "planning_preview_used_in_command",
                "spring_used_in_command"):
            self.assertEqual(0, diagnostics[flag])


if __name__ == "__main__":
    unittest.main()
