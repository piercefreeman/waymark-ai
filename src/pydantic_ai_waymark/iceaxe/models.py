"""The Iceaxe table that backs agent mailboxes.

Import ``AgentMessageRecord`` into the module your Iceaxe migrations scan so its table is
created.
"""

from datetime import UTC, datetime

from iceaxe import Field, IndexConstraint, PostgresDateTime, TableBase, UniqueConstraint

from ..types import AgentMessage


def utcnow() -> datetime:
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
        default_factory=utcnow,
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
