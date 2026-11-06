import math
from typing import TYPE_CHECKING, Any, Never

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

from .types import AgentDeps, ToolValue

if TYPE_CHECKING:
    from .request import AIRequestBase


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


class DelegateAgent(CallDeferred):
    """Ask the enclosing Waymark workflow to run a subagent for this tool call.

    With ``wait=True`` (the default) the workflow runs ``request`` through ``run_agent``
    as durable nested work, every subagent model and tool step checkpointed on its own,
    and the subagent's output becomes this tool call's result. The request's type must
    then be one of the request types the workflow declares in ``PydanticAIWorkflow[...]``.

    With ``wait=False`` the subagent starts as its own Waymark workflow instance and this
    tool call returns a handle right away: ``{"subagent_id", "mailbox", "status"}``. When
    the subagent finishes, it sends its output (or failure) to ``mailbox`` through the
    delegating agent's ``message_writer``; check on it later from another tool with
    ``WaitForMessages(mailbox, timeout_seconds=0)``, or wait for it.
    """

    def __init__(self, request: "AIRequestBase[Any]", *, wait: bool = True) -> None:
        self.request = request
        self.wait = wait
        super().__init__()

    def __reduce__(self) -> tuple[Any, ...]:
        return _rebuild_delegate_agent, (self.request, self.wait)


def _rebuild_delegate_agent(request: "AIRequestBase[Any]", wait: bool) -> DelegateAgent:
    return DelegateAgent(request, wait=wait)


def subagent_mailbox(subagent_id: str) -> str:
    """Return the mailbox a detached subagent reports its result to."""
    return f"waymark-subagent:{subagent_id}"


class WaitForMessages(CallDeferred):
    """Ask the enclosing Waymark workflow to wait for messages on a mailbox.

    The workflow polls the agent's registered ``message_reader`` every ``poll_seconds``,
    sleeping durably in between, and returns the first non-empty batch as this tool
    call's result. With ``timeout_seconds`` set, ``timeout_result`` is returned instead
    once that much polling time has passed; ``0`` checks the mailbox exactly once.

    NOTE: Polling is a stopgap. The long-term answer is likely native message queues in
    Waymark itself, so a workflow can block on a message instead of waking up to poll.
    """

    def __init__(
        self,
        mailbox: str,
        *,
        poll_seconds: float = 1.0,
        timeout_seconds: float | None = None,
        timeout_result: ToolValue = "No messages received.",
    ) -> None:
        poll_seconds = float(poll_seconds)
        if not math.isfinite(poll_seconds) or poll_seconds <= 0:
            raise ValueError("poll interval must be finite and positive")
        if timeout_seconds is not None:
            timeout_seconds = float(timeout_seconds)
            if not math.isfinite(timeout_seconds) or timeout_seconds < 0:
                raise ValueError("message timeout must be finite and non-negative")
        if not mailbox:
            raise ValueError("mailbox must be a non-empty string")
        self.mailbox = mailbox
        self.poll_seconds = poll_seconds
        self.timeout_seconds = timeout_seconds
        self.timeout_result = timeout_result
        super().__init__()

    def __reduce__(self) -> tuple[Any, ...]:
        return (
            _rebuild_wait_for_messages,
            (self.mailbox, self.poll_seconds, self.timeout_seconds, self.timeout_result),
        )


def _rebuild_wait_for_messages(
    mailbox: str,
    poll_seconds: float,
    timeout_seconds: float | None,
    timeout_result: ToolValue,
) -> WaitForMessages:
    return WaitForMessages(
        mailbox,
        poll_seconds=poll_seconds,
        timeout_seconds=timeout_seconds,
        timeout_result=timeout_result,
    )


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
