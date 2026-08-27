import math
from typing import Never

from pydantic import ValidationError
from pydantic_ai import RunContext
from pydantic_ai.capabilities import (
    AbstractCapability,
    ValidatedToolArgs,
    WrapToolExecuteHandler,
)
from pydantic_ai.capabilities.abstract import CapabilityOrdering
from pydantic_ai.exceptions import CallDeferred, UserError
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import DeferredToolRequests, ToolDefinition

from .types import AgentDeps, ToolValue, WaymarkToolPolicy


class DurableSleep(CallDeferred):
    """Ask the enclosing Waymark workflow to wait before returning this tool result."""

    def __init__(self, seconds: float, result: ToolValue = "Sleep completed.") -> None:
        seconds = float(seconds)
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("sleep duration must be finite and non-negative")
        self.seconds = seconds
        self.result = result
        super().__init__(
            metadata={
                "waymark": {
                    "kind": "sleep",
                    "seconds": seconds,
                    "result": result,
                }
            }
        )

    def __reduce__(self) -> tuple[type, tuple[float, ToolValue]]:
        return type(self), (self.seconds, self.result)


def _tool_policy(tool_def: ToolDefinition) -> WaymarkToolPolicy:
    metadata = tool_def.metadata or {}
    policy = metadata.get("waymark", {})
    try:
        return WaymarkToolPolicy.model_validate(policy)
    except ValidationError as error:
        message = f"Tool {tool_def.name!r} has invalid 'waymark' metadata: {error}"
        raise UserError(message) from error


class PendingToolCallsError(Exception):
    def __init__(self, requests: DeferredToolRequests) -> None:
        self.requests = requests


class WaymarkToolBoundary(AbstractCapability[AgentDeps]):
    """Stop a graph node after validation, before user tool code runs."""

    @classmethod
    def get_serialization_name(cls) -> None:
        return None

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position="outermost")

    async def wrap_tool_execute(
        self,
        ctx: RunContext[AgentDeps],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        handler: WrapToolExecuteHandler,
    ) -> Never:
        del ctx, call, handler
        policy = _tool_policy(tool_def)
        metadata: dict[str, object] = {
            "waymark": policy,
            "waymark_sequential": tool_def.sequential,
        }
        raise CallDeferred(metadata=metadata)

    async def handle_deferred_tool_calls(
        self,
        ctx: RunContext[AgentDeps],
        *,
        requests: DeferredToolRequests,
    ) -> Never:
        del ctx
        raise PendingToolCallsError(requests)


tool_boundary = WaymarkToolBoundary()
