import math

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
from suspension_control.controllers.skyhook import SkyhookController


class StaticPolicy:
    is_available = True
    status = "static"

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


def state(speed=8.0, frame=1):
    return VehicleState(
        step=frame,
        frame=frame,
        elapsed_seconds=0.05 * frame,
        dt=0.05,
        speed=speed,
        local_vx=speed,
        local_ay=0.8,
        az=0.3,
        roll=1.2,
        pitch=0.4,
        roll_rate=0.6,
        pitch_rate=0.2,
        yaw_rate=0.1,
        throttle=0.2,
        brake=0.0,
        steer=0.03)


def context(vehicle_state=None, previous_state=None):
    vehicle_state = vehicle_state or state()
    return ControllerContext(
        state=vehicle_state,
        previous_state=previous_state,
        step=vehicle_state.step,
        dt=vehicle_state.dt)


def dampers(command):
    return [wheel.damper_scale for wheel in command.wheels]


def test_default_config_parses_unknown_keys_ignored():
    config = ResidualRLConfig.from_mapping({
        "baseline": "pid",
        "max_damper_residual_scale": 0.04,
        "unknown": "ignored",
    })
    assert config.baseline == "pid"
    assert config.max_damper_residual_scale == 0.04


def test_missing_policy_returns_skyhook_baseline_exactly():
    ctx = context()
    baseline = SkyhookController().compute(ctx).command
    output = ResidualRLController(
        ResidualRLConfig(allow_untrained_policy=False)).compute(ctx)
    assert output.command.is_close(baseline)
    assert output.diagnostics["rl_fallback_reason"] == "policy_unavailable"


def test_dummy_zero_policy_returns_skyhook_baseline_exactly():
    ctx = context()
    baseline = SkyhookController().compute(ctx).command
    output = ResidualRLController(
        ResidualRLConfig(allow_untrained_policy=True)).compute(ctx)
    assert output.command.is_close(baseline)
    assert output.diagnostics["rl_fallback_reason"] == "zero_residual"


def test_exact_fallback_bypasses_projection_and_clamping():
    baseline = SuspensionCommand.uniform(damper_scale=1.42)
    controller = ResidualRLController(
        ResidualRLConfig(
            allow_untrained_policy=False,
            max_damper_scale=1.30),
        baseline_controller=ConstantBaseline(baseline))
    output = controller.compute(context())
    assert output.command.is_close(baseline)
    assert dampers(output.command) == [1.42, 1.42, 1.42, 1.42]


def test_fake_policy_produces_bounded_residual_and_frozen_springs():
    ctx = context()
    controller = ResidualRLController(
        ResidualRLConfig(allow_untrained_policy=True),
        policy=StaticPolicy([1.0, -1.0, 0.5, -0.5]))
    output = controller.compute(ctx)
    baseline = SkyhookController().compute(ctx).command
    assert not output.command.is_close(baseline)
    for wheel in output.command.wheels:
        assert 0.80 <= wheel.damper_scale <= 1.30
        assert wheel.spring_scale == 1.0
    assert output.diagnostics["rl_fallback_reason"] == ""


def test_non_finite_action_triggers_baseline_fallback():
    ctx = context()
    baseline = SkyhookController().compute(ctx).command
    controller = ResidualRLController(
        ResidualRLConfig(allow_untrained_policy=True),
        policy=StaticPolicy([math.nan, 0.0, 0.0, 0.0]))
    output = controller.compute(ctx)
    assert output.command.is_close(baseline)
    assert output.diagnostics["rl_fallback_reason"] == "policy_action_non_finite"


def test_rate_limit_enforced_across_compute_calls():
    cfg = ResidualRLConfig(
        allow_untrained_policy=True,
        max_damper_residual_scale=0.08,
        max_damper_delta_per_step=0.02)
    controller = ResidualRLController(cfg, policy=StaticPolicy([1.0, 1.0, 1.0, 1.0]))
    first = controller.compute(context(state(frame=1))).command
    second = controller.compute(
        context(state(frame=2), previous_state=state(frame=1))).command
    for left, right in zip(dampers(first), dampers(second)):
        assert abs(right - left) <= cfg.max_damper_delta_per_step + 1.0e-9
