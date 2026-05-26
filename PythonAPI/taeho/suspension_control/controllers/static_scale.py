"""Static spring/damper scale baseline."""

from .base import ControllerContext, ControllerOutput, SuspensionCommand, SuspensionController


class StaticScaleController(SuspensionController):
    """Return a fixed uniform spring/damper scale every step."""

    name = "static_scale"

    def __init__(self, spring_scale=1.0, damper_scale=1.0, wheel_count=4):
        self.spring_scale = float(spring_scale)
        self.damper_scale = float(damper_scale)
        self.wheel_count = int(wheel_count)
        self.command = SuspensionCommand.uniform(
            spring_scale=self.spring_scale,
            damper_scale=self.damper_scale,
            wheel_count=self.wheel_count).validate(expected_wheels=self.wheel_count)

    def compute(self, context: ControllerContext) -> ControllerOutput:
        return ControllerOutput(
            command=self.command,
            diagnostics={
                "controller": self.name,
                "spring_scale": self.spring_scale,
                "damper_scale": self.damper_scale,
            })
