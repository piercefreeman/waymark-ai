from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.capabilities.abstract import CapabilityOrdering
from pydantic_ai.exceptions import CallDeferred, UserError
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolDefinition


class _WaymarkToolPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    attempts: int = Field(default=3, ge=1)
    backoff_seconds: float = Field(default=2.0, ge=0)
    timeout: float = Field(default=120.0, gt=0)


workflow_args_adapter = TypeAdapter(Any)


def _tool_policy(tool_def: ToolDefinition) -> dict[str, Any] | Literal[False]:
    metadata = tool_def.metadata or {}
    policy = metadata.get("waymark", {})
    if policy is False:
        return False
    try:
        return _WaymarkToolPolicy.model_validate(policy).model_dump()
    except ValidationError as error:
        message = f"Tool {tool_def.name!r} has invalid 'waymark' metadata: {error}"
        raise UserError(message) from error


class PendingToolCallsError(Exception):
    def __init__(self, requests: DeferredToolRequests) -> None:
        self.requests = requests


class WaymarkToolBoundary(AbstractCapability[Any]):
    """Stop a graph node after validation, before user tool code runs."""

    @classmethod
    def get_serialization_name(cls) -> None:
        return None

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position="outermost")

    async def wrap_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: Any,
        handler: Any,
    ) -> Any:
        del ctx, call, handler
        policy = _tool_policy(tool_def)
        metadata = {"waymark": policy}
        if policy is False:
            metadata["waymark_args"] = workflow_args_adapter.dump_python(args, mode="json")
        raise CallDeferred(metadata=metadata)

    async def handle_deferred_tool_calls(
        self,
        ctx: RunContext[Any],
        *,
        requests: DeferredToolRequests,
    ) -> DeferredToolResults | None:
        del ctx
        raise PendingToolCallsError(requests)


tool_boundary = WaymarkToolBoundary()
