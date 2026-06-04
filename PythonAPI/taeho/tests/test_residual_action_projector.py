from suspension_control.controllers.base import (
    ControllerContext,
    ControllerOutput,
    SuspensionCommand,
    VehicleState,
)
from suspension_control.controllers.rl_residual import (
    ResidualRLConfig,
    ResidualRLController,
)
from suspension_control.rl.action_projection import (
    ResidualActionProjector,
    ResidualActionProjectorConfig,
    ResidualProjectionResult,
)


class StaticPolicy:
    is_available = True
    status = "static"
    builtin_id = ""
    alias_deprecated = False

    def __init__(self, action):
        self.action = list(action)

    def predict(self, observation, deterministic=True):
        return list(self.action)


class ConstantBaseline:
    def __init__(self, command):
        self.command = command

    def reset(self, native_suspension=None):
        pass

    def compute(self, context):
        return ControllerOutput(
            command=self.command,
            diagnostics={"controller": "constant_baseline"})


def state(speed=8.0, local_ay=0.8, roll=1.2, pitch=0.4, yaw_rate=0.1):
    return VehicleState(
        step=1,
        frame=1,
        elapsed_seconds=0.05,
        dt=0.05,
        speed=speed,
        local_vx=speed,
        local_ay=local_ay,
        az=0.3,
        roll=roll,
        pitch=pitch,
        roll_rate=0.6,
        pitch_rate=0.2,
        yaw_rate=yaw_rate)


def context(vehicle_state):
    return ControllerContext(
        state=vehicle_state,
        previous_state=None,
        step=vehicle_state.step,
        dt=vehicle_state.dt)


def dampers(command):
    return [wheel.damper_scale for wheel in command.wheels]


def assert_close_tuple(values, expected, tolerance=1.0e-12):
    assert len(values) == len(expected)
    for value, target in zip(values, expected):
        assert abs(value - target) <= tolerance


def test_zero_action_returns_baseline_exactly_even_at_low_speed():
    baseline = SuspensionCommand.uniform(spring_scale=1.03, damper_scale=1.12)
    projector = ResidualActionProjector()

    result = projector.project(
        baseline,
        [0.0, 0.0, 0.0, 0.0],
        state(speed=0.0))

    assert result.command.is_close(baseline, abs_tol=0.0, rel_tol=0.0)
    assert result.safety_gain == 0.0
    assert result.safety_gate_active == 1
    assert result.safety_gate_reason == "speed_below_min"
    assert result.fallback_reason == ""
    assert result.final_residual_damper_per_wheel == (0.0, 0.0, 0.0, 0.0)


def test_action_scale_semantics_before_rate_limit_and_clamp():
    baseline = SuspensionCommand.uniform(damper_scale=1.0)
    projector = ResidualActionProjector(ResidualActionProjectorConfig(
        max_damper_residual_scale=0.08,
        max_damper_delta_per_step=1.0))

    result = projector.project(
        baseline,
        [0.25, 0.25, 0.25, 0.25],
        state())

    assert result.raw_residual_damper_per_wheel == (0.02, 0.02, 0.02, 0.02)
    assert result.safety_scaled_residual_damper_per_wheel == (
        0.02,
        0.02,
        0.02,
        0.02)
    assert isinstance(result, ResidualProjectionResult)
    assert result.action == result.sanitized_action
    assert result.raw_residuals == result.raw_residual_damper_per_wheel
    assert result.safety_scaled_residuals == (
        result.safety_scaled_residual_damper_per_wheel)
    assert result.rate_limited_residuals == (
        result.rate_limited_residual_damper_per_wheel)
    assert result.final_residuals == result.final_residual_damper_per_wheel
    assert result.baseline_dampers == result.baseline_damper_per_wheel
    assert result.final_dampers == result.final_damper_per_wheel
    assert result.safety_diagnostics["rl_safety_gain"] == 1.0
    assert_close_tuple(
        result.final_residual_damper_per_wheel,
        (0.02, 0.02, 0.02, 0.02))
    assert_close_tuple(dampers(result.command), [1.02, 1.02, 1.02, 1.02])


def test_speed_below_threshold_masks_nonzero_residual():
    baseline = SuspensionCommand.uniform(damper_scale=1.0)
    projector = ResidualActionProjector()

    result = projector.project(
        baseline,
        [0.25, 0.25, 0.25, 0.25],
        state(speed=0.0))

    assert result.command.is_close(baseline, abs_tol=0.0, rel_tol=0.0)
    assert result.safety_gain == 0.0
    assert result.safety_gate_reason == "speed_below_min"
    assert result.fallback_reason == "safety_gate_zero"
    assert result.final_residual_damper_per_wheel == (0.0, 0.0, 0.0, 0.0)


def test_obs_diag_and_previous_final_dampers_are_supported():
    baseline = SuspensionCommand.uniform(damper_scale=1.0)
    projector = ResidualActionProjector(ResidualActionProjectorConfig(
        max_damper_residual_scale=0.08,
        max_damper_delta_per_step=0.04))

    invalid_result = projector.project(
        baseline,
        [0.25, 0.25, 0.25, 0.25],
        state(),
        obs_diag={"observation_valid": 0})
    assert invalid_result.command.is_close(baseline, abs_tol=0.0, rel_tol=0.0)
    assert invalid_result.fallback_reason == "safety_gate_zero"
    assert invalid_result.safety_gate_reason == "nonfinite_obs"

    result = projector.project(
        baseline,
        [1.0, 1.0, 1.0, 1.0],
        state(),
        previous_final_dampers=[0.90, 0.90, 0.90, 0.90])
    assert_close_tuple(dampers(result.command), [0.94, 0.94, 0.94, 0.94])


def test_high_lateral_acc_reduces_gain_and_marks_reason():
    baseline = SuspensionCommand.uniform(damper_scale=1.0)
    projector = ResidualActionProjector(ResidualActionProjectorConfig(
        max_damper_residual_scale=0.08,
        max_damper_delta_per_step=1.0,
        max_abs_lateral_acc_for_full_policy=7.0))

    result = projector.project(
        baseline,
        [0.25, 0.25, 0.25, 0.25],
        state(local_ay=14.0))

    assert result.safety_gate_reason == "lateral_acc_limit"
    assert result.safety_gain == 0.5
    assert result.safety_scaled_residual_damper_per_wheel == (
        0.01,
        0.01,
        0.01,
        0.01)
    assert_close_tuple(
        result.final_residual_damper_per_wheel,
        (0.01, 0.01, 0.01, 0.01))


def test_controller_and_projector_command_parity():
    baseline = SuspensionCommand.uniform(damper_scale=1.0)
    action = [0.25, -0.25, 0.5, -0.5]
    vehicle_state = state()
    cfg = ResidualRLConfig(
        allow_untrained_policy=True,
        max_damper_residual_scale=0.08,
        max_damper_delta_per_step=1.0)
    controller = ResidualRLController(
        cfg,
        baseline_controller=ConstantBaseline(baseline),
        policy=StaticPolicy(action))
    projector = ResidualActionProjector(ResidualActionProjectorConfig(
        max_damper_residual_scale=cfg.max_damper_residual_scale,
        max_damper_delta_per_step=cfg.max_damper_delta_per_step))

    output = controller.compute(context(vehicle_state))
    projected = projector.project(baseline, action, vehicle_state)

    assert output.command.is_close(projected.command)
    assert output.diagnostics["rl_mean_abs_residual_damper"] == (
        projected.diagnostics["rl_mean_abs_residual_damper"])
