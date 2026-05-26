"""Placeholder for the future planning-preview suspension controller."""

from .base import ControllerContext, ControllerOutput, SuspensionCommand, SuspensionController


class PlanningPreviewSuspensionController(SuspensionController):
    """Skeleton for a proactive controller using planning information.

    The current placeholder deliberately returns identity suspension so it can
    be inserted into the runtime loop without changing vehicle behavior. Future
    work can replace `_preview_risk` and `compute` with model-predictive or
    learned preview logic.
    """

    name = "planning_preview_placeholder"

    def __init__(self, wheel_count=4):
        self.wheel_count = int(wheel_count)

    def compute(self, context: ControllerContext) -> ControllerOutput:
        preview_risk = self._preview_risk(context)
        return ControllerOutput(
            command=SuspensionCommand.identity(self.wheel_count),
            diagnostics={
                "controller": self.name,
                "placeholder": True,
                "planning_points": len(context.planning.points),
                "preview_risk": preview_risk,
            })

    def _preview_risk(self, context: ControllerContext) -> float:
        # Future hook: derive roll/pitch/comfort risk from planned path,
        # planned speeds, curvature, braking, and route horizon.
        return 0.0


YourController = PlanningPreviewSuspensionController
