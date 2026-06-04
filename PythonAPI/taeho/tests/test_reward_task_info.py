from suspension_control.controllers.base import (
    PlanningInfo,
    SuspensionCommand,
    VehicleState,
)
from suspension_control.rl.reward import RewardTransition, SuspensionReward
from suspension_control.rl.task_info import (
    TaskInfoBuilder,
    TaskInfoBuilderConfig,
    normalize_task_info,
)


def command():
    return SuspensionCommand.identity()


def route_progress(**overrides):
    values = {
        "route_progress_available": 1,
        "route_progress_fraction": 0.2,
        "route_progress_monotonic_fraction": 0.2,
        "route_progress_m": 10.0,
        "route_progress_delta_m": 0.0,
        "route_delta_progress_m": 0.0,
        "route_progress_rate_mps": 0.0,
        "route_deviation_m": 0.0,
        "route_progress_deviation_valid": 1,
        "route_distance_to_end_m": 90.0,
    }
    values.update(overrides)
    return values


def reward_transition(task_info):
    state = VehicleState(speed=0.1, local_vx=0.1, brake=0.0, dt=0.05)
    return RewardTransition(
        state=state,
        previous_state=state,
        action=[0.0, 0.0, 0.0, 0.0],
        previous_action=[0.0, 0.0, 0.0, 0.0],
        final_command=command(),
        baseline_command=command(),
        task_info=task_info,
        dt=0.05)


def test_task_info_builder_detects_low_speed_not_planned():
    builder = TaskInfoBuilder()

    info = builder.build(
        state=VehicleState(speed=0.1, brake=0.0),
        route_progress=route_progress())
    planned_brake = builder.build(
        state=VehicleState(speed=0.1, brake=0.5),
        route_progress=route_progress(),
        actor_id="braking")

    assert info["target_speed"] == 8.0
    assert info["target_speed_source"] == "nominal"
    assert info["low_speed_not_planned"] == 1.0
    assert info["low_speed_not_planned_severity"] > 0.0
    assert info["planned_stop"] == 0.0
    assert info["abs_speed_error"] > 0.0
    assert planned_brake["low_speed_not_planned"] == 0.0
    assert planned_brake["planned_stop"] == 1.0


def test_task_info_builder_uses_planning_target_speed():
    builder = TaskInfoBuilder()
    planning = PlanningInfo(available=True, source="jsonl", target_speed=(4.0,))

    info = builder.build(
        state=VehicleState(speed=3.5, brake=0.0),
        route_progress=route_progress(route_progress_rate_mps=3.0),
        planning=planning)

    assert info["target_speed"] == 4.0
    assert info["target_speed_source"] == "planning"
    assert info["speed_error"] == 0.5
    assert info["low_speed_not_planned"] == 0.0


def test_stall_detection_uses_monotonic_progress():
    builder = TaskInfoBuilder(TaskInfoBuilderConfig(progress_stall_steps=3))

    info = {}
    for _ in range(3):
        info = builder.build(
            state=VehicleState(speed=2.0, brake=0.0),
            route_progress=route_progress(
                route_progress_delta_m=0.0,
                route_delta_progress_m=-2.0),
            actor_id="stalling")

    assert info["progress_stall"] == 1.0
    assert info["route_progress_stall"] == 1.0
    assert info["progress_stall_count"] == 3.0

    moving = builder.build(
        state=VehicleState(speed=2.0, brake=0.0),
        route_progress=route_progress(
            route_progress_delta_m=1.0,
            route_delta_progress_m=-2.0),
        actor_id="stalling")
    assert moving["progress_stall"] == 0.0
    assert moving["progress_stall_count"] == 0.0


def test_task_info_builder_detects_negative_progress():
    builder = TaskInfoBuilder()

    info = builder.build(
        state=VehicleState(speed=2.0, brake=0.0),
        route_progress=route_progress(
            route_progress_delta_m=0.0,
            route_delta_progress_m=-0.5))

    assert info["negative_progress"] == 1.0
    assert info["delta_progress"] == 0.0
    assert info["route_progress_delta_m"] == 0.0
    assert info["route_delta_progress_m"] == -0.5


def test_planned_stop_does_not_trigger_low_speed_not_planned():
    builder = TaskInfoBuilder()
    planning = PlanningInfo(
        available=True,
        source="jsonl",
        target_speed=(0.0,),
        brake=(0.4,))

    info = builder.build(
        state=VehicleState(speed=0.1, brake=0.0),
        route_progress=route_progress(),
        planning=planning,
        actor_id="planned_stop")

    assert info["planned_stop"] == 1.0
    assert info["low_speed_not_planned"] == 0.0
    assert info["progress_stall"] == 0.0


def test_normalize_task_info_preserves_new_aliases():
    info = normalize_task_info(
        {
            "route_progress_delta_m": 0.0,
            "route_delta_progress_m": -0.2,
            "route_progress_rate_mps": -4.0,
            "route_progress_monotonic_fraction": 0.4,
            "route_progress_deviation_valid": 1,
            "abs_speed_error": 3.0,
            "target_speed_source": "nominal",
        },
        require_non_empty=False)

    assert info["delta_progress"] == 0.0
    assert info["route_progress_delta_m"] == 0.0
    assert info["route_delta_progress_m"] == -0.2
    assert info["negative_progress"] == 1.0
    assert info["route_progress_monotonic_fraction"] == 0.4
    assert info["route_deviation_valid"] == 1
    assert info["abs_speed_error"] == 3.0
    assert info["target_speed_source"] == "nominal"


def test_reward_task_terms_nonzero_for_phase3_task_signals():
    task_info = {
        "target_speed": 8.0,
        "abs_speed_error": 7.9,
        "delta_progress": 0.0,
        "route_progress_delta_m": 0.0,
        "route_delta_progress_m": -0.5,
        "negative_progress": 1.0,
        "low_speed_not_planned": 1.0,
        "low_speed_not_planned_severity": 0.8,
        "progress_stall": 1.0,
    }

    _, diagnostics = SuspensionReward().compute(reward_transition(task_info))

    assert diagnostics["reward_term_low_speed_not_planned"] > 0.0
    assert diagnostics["reward_term_progress_stall"] > 0.0
    assert diagnostics["reward_term_negative_progress"] == 0.0
    assert diagnostics["reward_term_abs_speed_error"] > 0.0
    assert diagnostics["reward_term_target_speed_error"] == (
        diagnostics["reward_term_abs_speed_error"])
