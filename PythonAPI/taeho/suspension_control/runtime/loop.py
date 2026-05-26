"""Reusable CARLA control loop for suspension controllers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from ..controllers.base import (
    ControllerContext,
    ControllerOutput,
    PlanningInfo,
    SuspensionCommand,
    SuspensionController,
    VehicleState,
)
from .carla_adapter import (
    apply_suspension_command,
    read_suspension_scale_summary,
    read_vehicle_state,
    validate_suspension_control,
)


@dataclass(frozen=True)
class LoopStepResult:
    step: int
    before_state: VehicleState
    after_state: VehicleState
    output: ControllerOutput
    command_applied: bool
    readback_summary: Mapping[str, Any] = field(default_factory=dict)


class SuspensionControlLoop:
    """Small synchronous loop around a `SuspensionController`.

    The loop computes a suspension command from the state available before a
    world tick, applies the command, optionally applies vehicle throttle/brake/
    steer control, then ticks the CARLA world.
    """

    def __init__(
        self,
        world: Any,
        vehicle: Any,
        controller: SuspensionController,
        apply_policy: str = "every_tick",
        command_tolerance: float = 1.0e-6,
        readback_tolerance: float = 1.0e-4,
        verify_readback: bool = False,
        default_dt: float = 0.05,
        tick_world: bool = True,
    ):
        self.world = world
        self.vehicle = vehicle
        self.controller = controller
        self.apply_policy = apply_policy
        self.command_tolerance = command_tolerance
        self.readback_tolerance = readback_tolerance
        self.verify_readback = verify_readback
        self.default_dt = default_dt
        self.tick_world = tick_world

        self.native_suspension = None
        self.previous_state = None
        self.last_command = None

    def reset(self) -> None:
        self.native_suspension = self.vehicle.get_suspension_physics_control()
        validate_suspension_control(self.native_suspension)
        self.controller.reset(self.native_suspension)
        self.previous_state = None
        self.last_command = None

    def step(
        self,
        vehicle_control: Any = None,
        planning: Optional[PlanningInfo] = None,
        step_index: int = 0,
    ) -> LoopStepResult:
        if self.native_suspension is None:
            self.reset()

        before_state = read_vehicle_state(
            self.world,
            self.vehicle,
            step=step_index,
            previous_state=self.previous_state,
            control=vehicle_control)
        dt = before_state.dt if before_state.dt > 0.0 else self.default_dt
        current_suspension = self.vehicle.get_suspension_physics_control()
        context = ControllerContext(
            state=before_state,
            previous_state=self.previous_state,
            planning=planning or PlanningInfo.empty(),
            native_suspension=self.native_suspension,
            current_suspension=current_suspension,
            step=step_index,
            dt=dt)
        output = self.controller.compute(context)
        output.command.validate(expected_wheels=len(self.native_suspension.wheels))

        command_applied = self._should_apply(output.command)
        if command_applied:
            apply_suspension_command(
                self.vehicle,
                self.native_suspension,
                output.command,
                verify_readback=self.verify_readback,
                readback_tolerance=self.readback_tolerance)
            self.last_command = output.command

        if vehicle_control is not None:
            self.vehicle.apply_control(vehicle_control)
        if self.tick_world:
            self.world.tick()

        after_state = read_vehicle_state(
            self.world,
            self.vehicle,
            step=step_index,
            previous_state=before_state,
            control=vehicle_control)
        self.previous_state = after_state

        readback_summary = read_suspension_scale_summary(
            self.native_suspension,
            self.vehicle.get_suspension_physics_control())

        return LoopStepResult(
            step=step_index,
            before_state=before_state,
            after_state=after_state,
            output=output,
            command_applied=command_applied,
            readback_summary=readback_summary)

    def restore_native(self, tick_world: bool = True) -> None:
        if self.native_suspension is None:
            return
        self.vehicle.apply_suspension_physics_control(self.native_suspension)
        if tick_world:
            self.world.tick()

    def _should_apply(self, command: SuspensionCommand) -> bool:
        if self.apply_policy == "never":
            return False
        if self.apply_policy == "every_tick":
            return True
        if self.apply_policy == "on_change":
            return not command.is_close(
                self.last_command,
                rel_tol=self.command_tolerance,
                abs_tol=self.command_tolerance)
        raise ValueError("unknown apply_policy %s" % self.apply_policy)
