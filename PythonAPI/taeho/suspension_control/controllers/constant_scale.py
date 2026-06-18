"""Constant spring/damper scale baseline for Step 07 experiments."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping

from .base import (
    ControllerContext,
    ControllerOutput,
    SuspensionCommand,
    SuspensionController,
    finite_float,
)


@dataclass(frozen=True)
class ConstantScaleConfig:
    """Configuration for a uniform constant-scale suspension baseline."""

    spring_scale: float = 1.0
    damper_scale: float = 1.03
    wheel_count: int = 4

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ConstantScaleConfig":
        allowed = {item.name for item in fields(cls)}
        kwargs = {key: value for key, value in values.items() if key in allowed}
        return cls(**kwargs)


class ConstantScaleController(SuspensionController):
    """Return a fixed uniform spring/damper command every control step."""

    name = "constant_scale"

    def __init__(self, config: ConstantScaleConfig | None = None):
        self.config = config or ConstantScaleConfig()
        spring_scale = finite_float(self.config.spring_scale, "spring_scale")
        damper_scale = finite_float(self.config.damper_scale, "damper_scale")
        wheel_count = int(self.config.wheel_count)
        self.command = SuspensionCommand.uniform(
            spring_scale=spring_scale,
            damper_scale=damper_scale,
            wheel_count=wheel_count).validate(expected_wheels=wheel_count)
        self.spring_scale = spring_scale
        self.damper_scale = damper_scale
        self.wheel_count = wheel_count

    def compute(self, context: ControllerContext) -> ControllerOutput:
        return ControllerOutput(
            command=self.command,
            diagnostics={
                "controller": self.name,
                "spring_scale": self.spring_scale,
                "damper_scale": self.damper_scale,
            })
