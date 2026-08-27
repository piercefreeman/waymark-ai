from typing import Literal, Never

from pydantic import TypeAdapter, ValidationError
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

from .types import AgentDeps, WaymarkToolPolicy, WorkflowToolArgs

workflow_args_adapter = TypeAdapter(WorkflowToolArgs)


def _tool_policy(tool_def: ToolDefinition) -> WaymarkToolPolicy | Literal[False]:
    metadata = tool_def.metadata or {}
    policy = metadata.get("waymark", {})
    if policy is False:
        return False
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
        metadata: dict[str, object] = {"waymark": policy}
        if policy is False:
            metadata["waymark_args"] = workflow_args_adapter.dump_python(args, mode="json")
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
