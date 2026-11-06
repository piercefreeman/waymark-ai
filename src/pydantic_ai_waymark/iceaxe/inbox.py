"""Mailbox operations over ``AgentMessageRecord``, and the hooks built from them.

Register ``message_reader(...)`` on agents that wait for messages and
``message_writer(...)`` on agents that start detached subagents; write your own messages
with ``send_message``.
"""

from collections.abc import Callable, Sequence
from typing import Any

from iceaxe import DBConnection, column, select, update
from mountaineer_di import Depends

from ..types import AgentMessage, MessageQuery, OutgoingMessage
from .models import AgentMessageRecord, utcnow


async def send_message(
    db: DBConnection,
    *,
    mailbox: str,
    content: str,
    sender: str | None = None,
    dedupe_key: str | None = None,
) -> None:
    """Insert one message into a mailbox.

    With ``dedupe_key`` set, a second send with the same key leaves the first message as
    it was, so a retried sender delivers once.
    """
    record = AgentMessageRecord(
        mailbox=mailbox,
        content=content,
        sender=sender,
        dedupe_key=dedupe_key,
    )
    if dedupe_key is None:
        await db.insert([record])
        return
    # Updating the conflicting key to itself keeps the stored row unchanged.
    await db.upsert(
        [record],
        conflict_fields=(AgentMessageRecord.dedupe_key,),
        update_fields=(AgentMessageRecord.dedupe_key,),
    )


async def claim_messages(
    db: DBConnection,
    query: MessageQuery,
    *,
    batch_size: int = 100,
) -> list[AgentMessage]:
    """Claim the oldest unread messages in a mailbox for one waiting tool call.

    Rows already claimed under ``query.claim_key`` are returned again instead of claiming
    more, which keeps a rerun Waymark poll from dropping or double-delivering messages.
    ``SKIP LOCKED`` lets concurrent waiters on one mailbox claim disjoint rows.
    """
    async with db.transaction(ensure=True):
        claimed = await db.exec(
            select(AgentMessageRecord)
            .where(
                AgentMessageRecord.mailbox == query.mailbox,
                AgentMessageRecord.claimed_by == query.claim_key,
            )
            .order_by(AgentMessageRecord.id)
        )
        if claimed:
            return [record.to_agent_message() for record in claimed]

        pending = await db.exec(
            select(AgentMessageRecord)
            .where(
                AgentMessageRecord.mailbox == query.mailbox,
                AgentMessageRecord.claimed_by == None,  # noqa: E711
            )
            .order_by(AgentMessageRecord.id)
            .limit(batch_size)
            .for_update(skip_locked=True)
        )
        if not pending:
            return []
        await db.exec(
            update(AgentMessageRecord)
            .set(AgentMessageRecord.claimed_by, query.claim_key)
            .set(AgentMessageRecord.claimed_at, utcnow())
            .where(column(AgentMessageRecord.id).in_([record.id for record in pending]))
        )
        return [record.to_agent_message() for record in pending]


def message_reader(
    get_db: Callable[..., Any],
    *,
    batch_size: int = 100,
) -> Callable[..., Any]:
    """Build a ``waymark_agent(message_reader=...)`` hook backed by ``AgentMessageRecord``.

    ``get_db`` is a ``mountaineer_di`` dependency provider that yields a
    ``DBConnection``, the same kind of provider the payload codecs accept.
    """

    async def read_messages(
        query: MessageQuery,
        db: DBConnection = Depends(get_db),
    ) -> Sequence[AgentMessage]:
        return await claim_messages(db, query, batch_size=batch_size)

    return read_messages


def message_writer(get_db: Callable[..., Any]) -> Callable[..., Any]:
    """Build a ``waymark_agent(message_writer=...)`` hook backed by ``AgentMessageRecord``.

    ``get_db`` is the same kind of ``mountaineer_di`` provider ``message_reader`` takes.
    """

    async def write_message(
        message: OutgoingMessage,
        db: DBConnection = Depends(get_db),
    ) -> None:
        await send_message(
            db,
            mailbox=message.mailbox,
            content=message.content,
            sender=message.sender,
            dedupe_key=message.dedupe_key,
        )

    return write_message
