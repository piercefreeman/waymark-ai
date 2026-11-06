"""A worker that dies mid-action is invisible to the agent.

Plain Pydantic AI has no notion of a lost worker, so a waymark-ai run must end the
way the same agent's plain run would. A lost model step reruns. A lost tool call
reruns the same call with no retry prompt, bounded by the tool's own ``retries=``
budget and counted apart from Pydantic AI's ``ModelRetry`` attempts; past it the
run fails with the error Pydantic AI raises for a tool that exceeds its retries.

Waymark matches a workflow's ``except`` clauses by the raised class's name and
bases, so an action raising an ``Exception`` named ``ActionExecutionLost`` reaches
the workflow exactly as the runtime's own lost-action error does. (The real class
derives from ``BaseException``, which an action can't raise: the worker runtime
reports only ``Exception``s, as a real lost action never comes from the action.)
"""

import asyncio

import pytest
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from waymark import ActionExecutionNotStarted, workflow
from waymark.exceptions import WorkflowFailedError

from pydantic_ai_waymark import AgentResult, AIRequestBase, PydanticAIWorkflow, waymark_agent


class ActionExecutionLost(Exception):  # noqa: N818 - named as Waymark's own
    """Stands in for Waymark's lost-action error; matched by name in the workflow."""


# How the fake runtime should treat the next calls: "lost", "not_started",
# "retry" (the tool raises ModelRetry), or anything else to run normally.
plan: dict[str, list[str]] = {}
runs: dict[str, int] = {}


@pytest.fixture(autouse=True)
def reset() -> None:
    plan.clear()
    runs.clear()


def step(name: str) -> None:
    runs[name] = runs.get(name, 0) + 1
    queued = plan.get(name) or []
    outcome = queued.pop(0) if queued else "run"
    if outcome == "lost":
        raise ActionExecutionLost()
    if outcome == "not_started":
        raise ActionExecutionNotStarted()
    if outcome == "retry":
        raise ModelRetry("try again")


def call_then_answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    step("model")
    returned = [
        part
        for message in messages
        for part in message.parts
        if isinstance(part, ToolReturnPart) or part.part_kind == "retry-prompt"
    ]
    # Answer from a real result; call the tool again after a retry request.
    if returned and isinstance(returned[-1], ToolReturnPart) and returned[-1].outcome != "retried":
        return ModelResponse(parts=[TextPart(f"answer: {returned[-1].content}")])
    return ModelResponse(parts=[ToolCallPart("lookup", {"topic": "refunds"})])


def make_agent(name: str, retries: int) -> Agent:
    agent = Agent(FunctionModel(call_then_answer), name=name)

    @agent.tool_plain(retries=retries)
    def lookup(topic: str) -> str:
        """Look a topic up."""
        step("lookup")
        return f"policy for {topic}"

    return agent


# The same agent as a plain Pydantic AI run (the reference) and as waymark-ai runs.
plain_agent = make_agent("plain_lookup", retries=2)
lookup_agent = waymark_agent(make_agent("lost_lookup", retries=2))
strict_agent = waymark_agent(make_agent("lost_strict", retries=0))
single_agent = waymark_agent(make_agent("lost_single", retries=1))


class LookupRequest(AIRequestBase[None]):
    agent = lookup_agent


class StrictRequest(AIRequestBase[None]):
    agent = strict_agent


class SingleRequest(AIRequestBase[None]):
    agent = single_agent


@workflow
class LookupWorkflow(PydanticAIWorkflow[LookupRequest]):
    async def run(self, request: LookupRequest) -> AgentResult:
        return await self.run_agent(request)


@workflow
class StrictWorkflow(PydanticAIWorkflow[StrictRequest]):
    async def run(self, request: StrictRequest) -> AgentResult:
        return await self.run_agent(request)


@workflow
class SingleWorkflow(PydanticAIWorkflow[SingleRequest]):
    async def run(self, request: SingleRequest) -> AgentResult:
        return await self.run_agent(request)


def shape(messages: list[ModelMessage]) -> list[tuple[str, ...]]:
    """What the agent saw and said, without timestamps or ids."""
    parts: list[tuple[str, ...]] = []
    for message in messages:
        for part in message.parts:
            if isinstance(part, ToolCallPart):
                parts.append(("call", part.tool_name))
            elif isinstance(part, ToolReturnPart) and part.outcome == "retried":
                parts.append(("retry", part.tool_name))
            elif isinstance(part, ToolReturnPart):
                parts.append(("return", part.tool_name, str(part.content)))
            elif part.part_kind == "retry-prompt":
                # How older Pydantic AI releases carry a tool's ModelRetry.
                parts.append(("retry", str(getattr(part, "tool_name", ""))))
            elif isinstance(part, TextPart):
                parts.append(("text", part.content))
            else:
                parts.append((part.part_kind,))
    return parts


def waymark_history(result: AgentResult) -> list[ModelMessage]:
    from pydantic_ai.messages import ModelMessagesTypeAdapter

    return ModelMessagesTypeAdapter.validate_json(result.message_history)


def run_lookup() -> AgentResult:
    result = asyncio.run(LookupWorkflow().run(LookupRequest(prompt="refunds?")))
    return AgentResult.model_validate(result) if isinstance(result, dict) else result


def test_a_lost_tool_call_reruns_and_the_agent_never_sees_it() -> None:
    reference = plain_agent.run_sync("refunds?")
    runs.clear()
    plan["lookup"] = ["lost", "lost"]

    result = run_lookup()

    # Two lost attempts fit the tool's retries=2; the third runs.
    assert runs["lookup"] == 3
    assert result.output == reference.output == "answer: policy for refunds"
    # Same conversation as the plain run: one call, one return, no retry prompt.
    assert shape(waymark_history(result)) == shape(reference.all_messages())


def test_a_tool_that_was_never_started_reruns_too() -> None:
    plan["lookup"] = ["not_started"]
    result = run_lookup()
    assert runs["lookup"] == 2 and result.output == "answer: policy for refunds"


def test_a_lost_model_step_reruns_invisibly() -> None:
    reference = plain_agent.run_sync("refunds?")
    runs.clear()
    plan["model"] = ["lost"]
    result = run_lookup()
    assert result.output == reference.output
    assert shape(waymark_history(result)) == shape(reference.all_messages())


def test_losing_a_tool_more_often_than_its_budget_fails_as_pydantic_ai_does() -> None:
    # Plain Pydantic AI: a tool that keeps failing past retries=1.
    plan["lookup"] = ["retry", "retry", "retry"]
    with pytest.raises(UnexpectedModelBehavior) as plain_error:
        make_agent("plain_single", retries=1).run_sync("refunds?")
    runs.clear()

    plan["lookup"] = ["lost", "lost", "lost"]
    with pytest.raises(WorkflowFailedError) as waymark_error:
        asyncio.run(SingleWorkflow().run(SingleRequest(prompt="refunds?")))

    # One rerun (retries=1), then the same failure Pydantic AI gives.
    assert runs["lookup"] == 2
    assert str(plain_error.value) in str(waymark_error.value)


def test_retries_zero_never_reruns_a_lost_tool() -> None:
    plan["lookup"] = ["lost"]
    with pytest.raises(WorkflowFailedError, match="exceeded max retries count of 0"):
        asyncio.run(StrictWorkflow().run(StrictRequest(prompt="refunds?")))
    assert runs["lookup"] == 1


def test_lost_attempts_are_counted_apart_from_model_retries() -> None:
    # The tool asks the model to retry once (a Pydantic AI retry the agent sees),
    # then the retried call is lost twice (which it doesn't). retries=2 covers both
    # counts separately, so the run ends as the plain run with one ModelRetry would.
    plan["lookup"] = ["retry"]
    reference = plain_agent.run_sync("refunds?")
    runs.clear()

    plan["lookup"] = ["retry", "lost", "lost"]
    result = run_lookup()

    assert result.output == reference.output
    assert shape(waymark_history(result)) == shape(reference.all_messages())
    assert ("retry", "lookup") in shape(waymark_history(result))
