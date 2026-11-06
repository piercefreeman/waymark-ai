import asyncio
import json
import os
from collections.abc import AsyncIterator

import asyncpg
import pytest
from iceaxe import DBConnection
from iceaxe.schemas.cli import create_all
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from waymark import workflow

from pydantic_ai_waymark import (
    AIRequestBase,
    DelegateAgent,
    MessageQuery,
    PydanticAIWorkflow,
    WaitForMessages,
    waymark_agent,
)
from pydantic_ai_waymark.iceaxe import (
    AgentMessageRecord,
    claim_messages,
    message_reader,
    message_writer,
    send_message,
)


@pytest.fixture(autouse=True)
def reset_message_table() -> None:
    if os.environ.get("WAYMARK_AI_TEST_DATABASE_URL") is None:
        return

    async def reset() -> None:
        db = await connect()
        try:
            await db.conn.execute(f"DROP TABLE IF EXISTS {AgentMessageRecord.table_name}")
            await create_all(db, [AgentMessageRecord])
        finally:
            await db.conn.close()

    asyncio.run(reset())


DATABASE_URL = os.environ.get("WAYMARK_AI_TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    DATABASE_URL is None,
    reason="set WAYMARK_AI_TEST_DATABASE_URL to run the Iceaxe mailbox tests",
)


async def connect() -> DBConnection:
    assert DATABASE_URL is not None
    return DBConnection(await asyncpg.connect(DATABASE_URL))


async def get_db() -> AsyncIterator[DBConnection]:
    db = await connect()
    try:
        yield db
    finally:
        await db.conn.close()


mailbox_agent = waymark_agent(
    Agent(TestModel(call_tools=["wait_for_operator"]), name="mailbox_agent"),
    message_reader=message_reader(get_db),
)


report_agent = waymark_agent(Agent(TestModel(custom_output_text="report"), name="report_agent"))
dispatch_agent = waymark_agent(
    Agent(TestModel(call_tools=["dispatch"]), name="dispatch_agent"),
    message_reader=message_reader(get_db),
    message_writer=message_writer(get_db),
)


class ReportRequest(AIRequestBase[None]):
    agent = report_agent


class DispatchRequest(AIRequestBase[None]):
    agent = dispatch_agent


@dispatch_agent.tool_plain
def dispatch(question: str) -> str:
    raise DelegateAgent(ReportRequest(prompt=question), wait=False)


@workflow
class DispatchWorkflow(PydanticAIWorkflow[DispatchRequest]):
    async def run(self, request: DispatchRequest) -> str:
        result = await self.run_agent(request)
        return result.output


class MailboxRequest(AIRequestBase[None]):
    agent = mailbox_agent


@mailbox_agent.tool_plain
def wait_for_operator() -> str:
    raise WaitForMessages("operator", poll_seconds=0.001, timeout_seconds=1)


@workflow
class MailboxWorkflow(PydanticAIWorkflow[MailboxRequest]):
    async def run(self, request: MailboxRequest) -> str:
        result = await self.run_agent(request)
        return result.output


def query(mailbox: str, claim_key: str) -> MessageQuery:
    return MessageQuery(
        mailbox=mailbox,
        claim_key=claim_key,
        agent_reference="tests:agent",
        run_id="run",
        tool_call_id="call",
        tool_name="wait",
    )


def test_claims_are_idempotent_per_claim_key() -> None:
    async def scenario() -> None:
        db = await connect()
        try:
            await send_message(db, mailbox="inbox", content="first", sender="human")
            await send_message(db, mailbox="inbox", content="second")
            await send_message(db, mailbox="elsewhere", content="not mine")

            claimed = await claim_messages(db, query("inbox", "run:a"))
            assert [message.content for message in claimed] == ["first", "second"]
            assert claimed[0].sender == "human"
            assert claimed[0].sent_at is not None

            # A rerun of the same poll gets the same rows back.
            assert await claim_messages(db, query("inbox", "run:a")) == claimed
            # A different waiter only sees messages nobody has claimed.
            assert await claim_messages(db, query("inbox", "run:b")) == []

            await send_message(db, mailbox="inbox", content="third")
            later = await claim_messages(db, query("inbox", "run:b"))
            assert [message.content for message in later] == ["third"]
        finally:
            await db.conn.close()

    asyncio.run(scenario())


def test_concurrent_waiters_claim_disjoint_messages() -> None:
    async def scenario() -> None:
        db = await connect()
        try:
            for index in range(20):
                await send_message(db, mailbox="shared", content=str(index))
        finally:
            await db.conn.close()

        async def claim(claim_key: str) -> list[str]:
            waiter_db = await connect()
            try:
                messages = await claim_messages(waiter_db, query("shared", claim_key), batch_size=5)
                return [message.id for message in messages]
            finally:
                await waiter_db.conn.close()

        batches = await asyncio.gather(*[claim(f"run:{index}") for index in range(4)])
        claimed_ids = [message_id for batch in batches for message_id in batch]
        assert len(claimed_ids) == 20
        assert len(set(claimed_ids)) == 20

    asyncio.run(scenario())


def test_workflow_receives_messages_from_the_iceaxe_mailbox() -> None:
    async def seed() -> None:
        db = await connect()
        try:
            await send_message(db, mailbox="operator", content="approved", sender="ops")
        finally:
            await db.conn.close()

    asyncio.run(seed())

    result = asyncio.run(MailboxWorkflow().run(MailboxRequest(prompt="wait for ops")))

    assert '"content":"approved"' in result
    assert '"sender":"ops"' in result


def test_sends_with_a_dedupe_key_are_stored_once() -> None:
    async def scenario() -> None:
        db = await connect()
        try:
            await send_message(db, mailbox="inbox", content="once", dedupe_key="k")
            await send_message(db, mailbox="inbox", content="again", dedupe_key="k")
            claimed = await claim_messages(db, query("inbox", "run:a"))
            assert [message.content for message in claimed] == ["once"]
        finally:
            await db.conn.close()

    asyncio.run(scenario())


def test_detached_subagent_reports_through_the_iceaxe_mailbox() -> None:
    result = asyncio.run(DispatchWorkflow().run(DispatchRequest(prompt="dispatch")))
    handle = json.loads(result)["dispatch"]
    assert handle["status"] == "started"

    async def read_report() -> list[str]:
        db = await connect()
        try:
            messages = await claim_messages(db, query(handle["mailbox"], "run:check"))
            return [message.content for message in messages]
        finally:
            await db.conn.close()

    [content] = asyncio.run(read_report())
    report = json.loads(content)
    assert report == {
        "subagent_id": handle["subagent_id"],
        "status": "completed",
        "output": "report",
    }
