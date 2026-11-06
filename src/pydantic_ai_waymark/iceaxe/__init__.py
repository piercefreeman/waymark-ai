"""Iceaxe-backed agent mailboxes for ``WaitForMessages``. Install with the ``iceaxe`` extra.

``models`` holds the ``AgentMessageRecord`` table; ``inbox`` holds the functions that send
and claim messages and the ``waymark_agent`` hooks built from them.

NOTE: This table is a polling workaround until Waymark has native message queues; it
should be replaceable by them without changing how tools raise ``WaitForMessages``.
"""

from .inbox import claim_messages, message_reader, message_writer, send_message
from .models import AgentMessageRecord

__all__ = [
    "AgentMessageRecord",
    "claim_messages",
    "message_reader",
    "message_writer",
    "send_message",
]
