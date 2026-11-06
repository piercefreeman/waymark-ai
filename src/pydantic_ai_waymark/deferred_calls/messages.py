"""Mailbox waits a tool asks for by raising ``WaitForMessages``."""

import math
from typing import Any

from pydantic_ai.exceptions import CallDeferred

from ..types import ToolValue


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
