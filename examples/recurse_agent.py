"""An agent that splits a question into sub-questions and delegates each back to itself.

``delegate_subquestion`` waits for a copy of this agent one level deeper, so the run forms
a tree of durable subagents; calls the model makes in the same turn fan out in parallel.
At the top level only, ``start_fact_check`` starts a separate fact-checking agent without
waiting, so it runs while the sub-questions are answered, and ``check_fact_check`` reads
its report from the Iceaxe mailbox just before the final answer.
"""

import os
import re
from collections.abc import AsyncIterator
from typing import Literal

import asyncpg
from iceaxe import DBConnection
from iceaxe.schemas.cli import create_all
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.tools import ToolDefinition
from waymark import workflow

from pydantic_ai_waymark import (
    AIRequestBase,
    DelegateAgent,
    PydanticAIWorkflow,
    WaitForMessages,
    waymark_agent,
)
from pydantic_ai_waymark.iceaxe import AgentMessageRecord, message_reader, message_writer

# Global


async def get_db() -> AsyncIterator[DBConnection]:
    """Open a connection to the database Waymark already uses."""
    db = DBConnection(await asyncpg.connect(os.environ["WAYMARK_DATABASE_URL"]))
    try:
        yield db
    finally:
        await db.conn.close()


async def create_mailbox_table() -> None:
    """Create the Iceaxe mailbox table if it doesn't exist yet."""
    async for db in get_db():
        exists = await db.conn.fetchval(
            "SELECT to_regclass($1) IS NOT NULL",
            AgentMessageRecord.table_name,
        )
        if not exists:
            await create_all(db, [AgentMessageRecord])


# Main Agent


class RecurseDependencies(BaseModel):
    depth: int = Field(default=0, ge=0)
    max_depth: int = Field(default=2, ge=0, le=3)


class RecurseAnswer(BaseModel):
    answer: str
    subquestions: list[str]


recurse_agent = waymark_agent(
    Agent(
        "openai:gpt-5.2",
        name="recurse_agent",
        deps_type=RecurseDependencies,
        output_type=RecurseAnswer,
        defer_model_check=True,
    ),
    message_reader=message_reader(get_db),
    message_writer=message_writer(get_db),
)


class RecurseRequest(AIRequestBase[RecurseDependencies]):
    agent = recurse_agent

    deps: RecurseDependencies | None = Field(default_factory=RecurseDependencies)


# A sub-question that repeats nearly all of the question's key words restates the whole question.
RESTATEMENT_OVERLAP = 0.9


def significant_words(text: str) -> set[str]:
    """Crude stems of the words longer than three letters, so "walk" matches "walking"."""
    return {word[:4] for word in re.findall(r"[a-z0-9']+", text.lower()) if len(word) > 3}


def restates(question: str, prompt: str) -> bool:
    prompt_words = significant_words(prompt)
    if not prompt_words:
        return False
    shared = prompt_words & significant_words(question)
    return len(shared) / len(prompt_words) >= RESTATEMENT_OVERLAP


@recurse_agent.instructions
def recurse_instructions(ctx: RunContext[RecurseDependencies]) -> str:
    if ctx.deps.depth >= ctx.deps.max_depth:
        return "Answer the question directly and concisely. Do not delegate."
    instructions = (
        "Delegate only when the question has two or more distinct parts that can be answered "
        "independently. Then call delegate_subquestion once per part, at most three, all in "
        "the same turn so they run in parallel, and combine the answers. Each sub-question "
        "must be narrower than the question; never delegate the whole question or a "
        "rephrasing of it. Otherwise answer directly. List the sub-questions you delegated "
        "in subquestions."
    )
    if ctx.deps.depth == 0:
        instructions += (
            " If your answer will rest on one checkable factual claim, call start_fact_check "
            "for it at most once, in your first turn alongside any delegations, and call "
            "check_fact_check with the returned mailbox only right before your final answer. "
            "Mention in the answer what the fact check said."
        )
    return instructions


async def top_level_only(
    ctx: RunContext[RecurseDependencies],
    tool_def: ToolDefinition,
) -> ToolDefinition | None:
    """Offer the fact-check tools only to the top-level agent, so each claim is checked once."""
    return tool_def if ctx.deps.depth == 0 else None


@recurse_agent.tool
def delegate_subquestion(ctx: RunContext[RecurseDependencies], question: str) -> RecurseAnswer:
    """Answer one distinct part of the question with another copy of this agent."""
    if ctx.deps.depth >= ctx.deps.max_depth:
        raise ModelRetry("Maximum depth reached; answer this part directly.")
    if isinstance(ctx.prompt, str) and restates(question, ctx.prompt):
        raise ModelRetry(
            "That restates the whole question. Delegate only narrower, distinct parts, or "
            "answer directly."
        )
    deps = ctx.deps.model_copy(update={"depth": ctx.deps.depth + 1})
    raise DelegateAgent(RecurseRequest(prompt=question, deps=deps))


@recurse_agent.tool_plain(prepare=top_level_only)
def start_fact_check(claim: str) -> dict[str, str]:
    """Start fact-checking a claim in the background; returns a handle with its mailbox."""
    raise DelegateAgent(FactCheckRequest(prompt=claim), wait=False)


@recurse_agent.tool_plain(prepare=top_level_only)
def check_fact_check(mailbox: str) -> list[dict[str, str]]:
    """Wait up to 30 seconds for a fact check started with start_fact_check."""
    raise WaitForMessages(
        mailbox,
        poll_seconds=1,
        timeout_seconds=30,
        timeout_result="The fact check is still running; answer without it.",
    )


@workflow
class RecurseWorkflow(PydanticAIWorkflow[RecurseRequest]):
    async def run(self, request: RecurseRequest) -> RecurseAnswer:
        result = await self.run_agent(request)
        return result.output


# Sub Agent


class FactCheck(BaseModel):
    claim: str
    verdict: Literal["plausible", "doubtful", "unverifiable"]
    reasoning: str
    verify_with: str


fact_check_agent = waymark_agent(
    Agent(
        "openai:gpt-5.2",
        name="fact_check_agent",
        instructions=(
            "You have no tools or sources, so you cannot confirm anything. Judge only whether "
            "the claim is plausible from general knowledge, never add specifics such as route "
            "numbers, schedules, or addresses that the claim does not contain, and name the "
            "source that would confirm it in verify_with."
        ),
        output_type=FactCheck,
        defer_model_check=True,
    )
)


class FactCheckRequest(AIRequestBase[None]):
    agent = fact_check_agent
