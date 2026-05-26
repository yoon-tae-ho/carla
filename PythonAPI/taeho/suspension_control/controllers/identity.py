"""Identity suspension baseline."""

from .base import ControllerContext, ControllerOutput, SuspensionCommand, SuspensionController


class IdentityController(SuspensionController):
    """Apply native spring and damper values through the runtime API."""

    name = "identity"

    def __init__(self, wheel_count=4):
        self.wheel_count = wheel_count

    def compute(self, context: ControllerContext) -> ControllerOutput:
        return ControllerOutput(
            command=SuspensionCommand.identity(self.wheel_count),
            diagnostics={"controller": self.name})
