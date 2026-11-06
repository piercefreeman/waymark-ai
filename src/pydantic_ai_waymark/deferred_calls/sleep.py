"""Durable timers a tool asks for by raising ``DurableSleep``."""

import math

from pydantic_ai.exceptions import CallDeferred

from ..types import ToolValue


class DurableSleep(CallDeferred):
    """Ask the enclosing Waymark workflow to wait before returning this tool result."""

    def __init__(self, seconds: float, result: ToolValue = "Sleep completed.") -> None:
        seconds = float(seconds)
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("sleep duration must be finite and non-negative")
        self.seconds = seconds
        self.result = result
        super().__init__()

    def __reduce__(self) -> tuple[type, tuple[float, ToolValue]]:
        return type(self), (self.seconds, self.result)
