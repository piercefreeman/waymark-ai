"""Subagents a tool hands work to by raising ``DelegateAgent``."""

from typing import TYPE_CHECKING, Any

from pydantic_ai.exceptions import CallDeferred

if TYPE_CHECKING:
    from ..request import AIRequestBase


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
