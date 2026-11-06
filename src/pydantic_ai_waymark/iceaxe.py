"""Iceaxe-backed agent mailboxes for ``WaitForMessages``.

Install with the ``iceaxe`` extra. Import ``AgentMessageRecord`` into the module your
Iceaxe migrations scan so its table is created, register ``message_reader(...)`` on the
agents that wait for messages and ``message_writer(...)`` on agents that start detached
subagents, and write your own messages with ``send_message``.

NOTE: This table is a polling workaround until Waymark has native message queues; it
should be replaceable by them without changing how tools raise ``WaitForMessages``.
"""

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

from iceaxe import (
    DBConnection,
    Field,
    IndexConstraint,
    PostgresDateTime,
    TableBase,
    UniqueConstraint,
    column,
    select,
    update,
)
from mountaineer_di import Depends

from .types import AgentMessage, MessageQuery, OutgoingMessage


def _utcnow() -> datetime:
    return datetime.now(UTC)


class AgentMessageRecord(TableBase):
    """One message sent to an agent mailbox.

    A row is unclaimed until a waiting tool call reads it; ``claimed_by`` then records the
    claim key so a rerun of the same poll returns the same rows. ``dedupe_key`` is unique
    when set, so a rerun send stores its message once.
    """

    table_name = "waymark_agent_messages"
    table_args = [
        IndexConstraint(columns=["mailbox", "claimed_by"]),
        UniqueConstraint(columns=["dedupe_key"]),
    ]

    id: int | None = Field(primary_key=True, default=None)
    mailbox: str
    content: str
    sender: str | None = None
    dedupe_key: str | None = None
    created_at: datetime = Field(
        default_factory=_utcnow,
        postgres_config=PostgresDateTime(timezone=True),
    )
    claimed_by: str | None = None
    claimed_at: datetime | None = Field(
        default=None,
        postgres_config=PostgresDateTime(timezone=True),
    )

    def to_agent_message(self) -> AgentMessage:
        return AgentMessage(
            id=str(self.id),
            content=self.content,
            sender=self.sender,
            sent_at=self.created_at,
        )


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
            .set(AgentMessageRecord.claimed_at, _utcnow())
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
