import math

from suspension_control.controllers.base import (
    ControllerContext,
    PlanningInfo,
    VehicleState,
)
from suspension_control.controllers.skyhook import SkyhookController
from suspension_control.rl.observations import ObservationBuilder


def state(**overrides):
    values = {
        "step": 1,
        "frame": 1,
        "elapsed_seconds": 0.05,
        "dt": 0.05,
        "speed": 8.0,
        "local_vx": 8.0,
        "local_vy": 0.1,
        "vz": 0.0,
        "local_ax": 0.2,
        "local_ay": 0.8,
        "az": 0.3,
        "roll": 1.2,
        "pitch": 0.4,
        "yaw_rate": 0.1,
        "roll_rate": 0.6,
        "pitch_rate": 0.2,
    }
    values.update(overrides)
    return VehicleState(**values)


def build_observation(planning=None, vehicle_state=None):
    vehicle_state = vehicle_state or state()
    context = ControllerContext(
        state=vehicle_state,
        planning=planning or PlanningInfo.empty(),
        step=vehicle_state.step,
        dt=vehicle_state.dt)
    baseline = SkyhookController().compute(context)
    builder = ObservationBuilder()
    return builder, builder.build(context, baseline, None, None)


def test_observation_length_matches_feature_names():
    builder, (observation, diagnostics) = build_observation()
    assert len(observation) == len(builder.feature_names)
    assert len(observation) == len(builder.spec().feature_names)
    assert diagnostics["observation_size"] == len(builder.feature_names)


def test_missing_planning_produces_zero_planning_features():
    _, (_, diagnostics) = build_observation()
    assert diagnostics["planning_available"] == 0.0
    assert diagnostics["preview_curvature_max_abs"] == 0.0
    assert diagnostics["preview_brake_max"] == 0.0


def test_synthetic_planning_preview_populates_features():
    planning = PlanningInfo(
        available=True,
        source="synthetic",
        frame=1,
        horizon_dt=0.1,
        target_speed=(8.0, 9.0),
        curvature=(0.0, 0.05),
        brake=(0.0, 0.3),
        predicted_ay=(0.0, 3.0))
    _, (_, diagnostics) = build_observation(planning=planning)
    assert diagnostics["planning_available"] == 1.0
    assert diagnostics["preview_curvature_max_abs"] == 0.05
    assert diagnostics["preview_brake_max"] == 0.3
    assert diagnostics["time_to_hard_brake"] == 0.1


def test_normalized_values_are_finite_and_clipped():
    extreme_state = state(speed=1000.0, local_ay=500.0, az=-500.0, roll=300.0)
    _, (observation, diagnostics) = build_observation(vehicle_state=extreme_state)
    assert all(math.isfinite(value) for value in observation)
    assert all(-5.0 <= value <= 5.0 for value in observation)
    assert diagnostics["observation_clip_count"] > 0
