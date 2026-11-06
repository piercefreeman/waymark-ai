"""Exceptions a tool raises to hand its result to the enclosing Waymark workflow.

Each one is a Pydantic AI ``CallDeferred``: the tool action returns it to the workflow,
which sleeps (``sleep``), runs or starts a subagent (``subagent``), or polls a mailbox
(``messages``) before the tool result goes back to the model. ``boundary`` defers every
tool call out of the model-step action in the first place.
"""

from .boundary import PendingToolCallsError, WaymarkToolBoundary, tool_boundary
from .messages import WaitForMessages
from .sleep import DurableSleep
from .subagent import DelegateAgent, subagent_mailbox

__all__ = [
    "DelegateAgent",
    "DurableSleep",
    "PendingToolCallsError",
    "WaitForMessages",
    "WaymarkToolBoundary",
    "subagent_mailbox",
    "tool_boundary",
]
