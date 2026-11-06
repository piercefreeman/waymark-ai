"""The capability that stops a graph node before any tool body runs.

Every tool call is deferred out of the model-step action, so each one can run as its own
Waymark action instead.
"""

from typing import Never

from pydantic_ai import RunContext
from pydantic_ai.capabilities import (
    AbstractCapability,
    ValidatedToolArgs,
    WrapToolExecuteHandler,
)
from pydantic_ai.capabilities.abstract import CapabilityOrdering
from pydantic_ai.exceptions import CallDeferred
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import DeferredToolRequests, ToolDefinition

from ..types import AgentDeps


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
        del ctx, call, tool_def, handler
        raise CallDeferred()

    async def handle_deferred_tool_calls(
        self,
        ctx: RunContext[AgentDeps],
        *,
        requests: DeferredToolRequests,
    ) -> Never:
        del ctx
        raise PendingToolCallsError(requests)


tool_boundary = WaymarkToolBoundary()
