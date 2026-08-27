import asyncio
import json
from typing import get_type_hints

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent, ModelMessagesTypeAdapter
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.models.test import TestModel
from waymark import workflow

from pydantic_ai_waymark import (
    AgentResult,
    AIRequestBase,
    BackoffConfig,
    DurableSleep,
    PydanticAIWorkflow,
    run_agent_node,
    run_agent_tool,
    waymark_agent,
)


class Answer(BaseModel):
    value: str


test_agent = waymark_agent(
    Agent(
        TestModel(custom_output_args={"value": "durable"}),
        name="test_agent",
        output_type=Answer,
    )
)
planning_failures = [0]
planning_failure_kind = ["transient"]
planning_calls: list[str] = []


@test_agent.instructions
def transient_planning() -> str:
    planning_calls.append("called")
    if planning_failures[0]:
        planning_failures[0] -= 1
        if planning_failure_kind[0] == "transient":
            raise ModelAPIError("test", "transient planning failure")
        raise RuntimeError("transient planning failure")
    return "Answer the request."

tool_calls: list[str] = []
tool_failures = [0]
tool_agent = waymark_agent(Agent(TestModel(call_tools=["lookup"]), name="tool_agent"))
sleep_agent = waymark_agent(Agent(TestModel(call_tools=["pause"]), name="sleep_agent"))
parallel_events: list[str] = []
parallel_agent = waymark_agent(
    Agent(
        TestModel(call_tools=["before_a", "before_b", "barrier", "after"]),
        name="parallel_agent",
    )
)


@tool_agent.tool_plain(
    metadata={"waymark": {"attempts": 2, "backoff_seconds": 0, "timeout": 10}}
)
def lookup(query: str) -> str:
    tool_calls.append(query)
    if tool_failures[0]:
        tool_failures[0] -= 1
        raise RuntimeError("transient lookup failure")
    return "found"


@sleep_agent.tool_plain
def pause() -> str:
    raise DurableSleep(0.001, result="waited")


async def record_tool(name: str) -> str:
    parallel_events.append(f"{name}:start")
    await asyncio.sleep(0.01)
    parallel_events.append(f"{name}:end")
    return name


@parallel_agent.tool_plain
async def before_a() -> str:
    return await record_tool("before_a")


@parallel_agent.tool_plain
async def before_b() -> str:
    return await record_tool("before_b")


@parallel_agent.tool_plain(sequential=True)
async def barrier() -> str:
    return await record_tool("barrier")


@parallel_agent.tool_plain
async def after() -> str:
    return await record_tool("after")


class AgentRequest(AIRequestBase[None]):
    agent = test_agent


class ToolRequest(AIRequestBase[None]):
    agent = tool_agent


class SleepRequest(AIRequestBase[None]):
    agent = sleep_agent


class ParallelRequest(AIRequestBase[None]):
    agent = parallel_agent


@workflow
class AgentWorkflow(PydanticAIWorkflow[AgentRequest]):
    async def run(self, request: AgentRequest) -> Answer:
        return (await self.run_agent(request)).output


@workflow
class ToolWorkflow(PydanticAIWorkflow[ToolRequest]):
    async def run(self, request: ToolRequest) -> str:
        return (await self.run_agent(request)).output


@workflow
class SleepWorkflow(PydanticAIWorkflow[SleepRequest]):
    async def run(self, request: SleepRequest) -> str:
        return (await self.run_agent(request)).output


@workflow
class ParallelWorkflow(PydanticAIWorkflow[ParallelRequest]):
    async def run(self, request: ParallelRequest) -> str:
        return (await self.run_agent(request)).output


@workflow
class UnionWorkflow(PydanticAIWorkflow[AgentRequest | ToolRequest]):
    async def run(self, request: AgentRequest | ToolRequest) -> Answer | str:
        return (await self.run_agent(request)).output


async def drive(agent_name: str) -> tuple[AgentResult, int]:
    transition = None
    tool_results = []
    for step_count in range(1, 10):
        transition = await run_agent_node(
            agent_name,
            "answer this",
            transition,
            tool_results,
        )
        assert isinstance(transition, BaseModel)
        tool_results = []
        if transition.kind == "done":
            return transition.result, step_count
        tool_results = [
            await run_agent_tool(agent_name, transition, call)
            for call in transition.tool_calls
        ]
    raise AssertionError("agent did not finish")


def test_model_run_is_checkpointed_by_graph_node() -> None:
    result, step_count = asyncio.run(drive("test_agent"))

    assert result.output == Answer(value="durable")
    assert ModelMessagesTypeAdapter.validate_json(result.message_history)
    assert ModelMessagesTypeAdapter.validate_json(result.new_messages)
    assert result.usage.requests == 1
    assert step_count == 3


def test_tool_round_trip_resumes_across_graph_node_actions() -> None:
    tool_calls.clear()
    result, step_count = asyncio.run(drive("tool_agent"))

    assert result.output == '{"lookup":"found"}'
    assert result.usage.requests == 2
    assert tool_calls == ["a"]
    assert step_count == 6


def test_waymark_executes_compiled_agent_state_machine() -> None:
    result = asyncio.run(AgentWorkflow().run(AgentRequest(prompt="answer this")))

    assert result == Answer(value="durable")
    ir_text = str(AgentWorkflow.workflow_ir())
    assert "pydantic_ai_agent_node" in ir_text
    assert "pydantic_ai_agent_tool" in ir_text
    assert "pydantic_ai_agent_step_limit" not in ir_text
    assert "sleep_stmt" in ir_text


def test_transient_model_failure_retries_with_backoff() -> None:
    planning_calls.clear()
    planning_failures[0] = 1
    planning_failure_kind[0] = "transient"

    result = asyncio.run(
        AgentWorkflow().run(
            AgentRequest(
                prompt="answer this",
                model_retry=BackoffConfig(initial_seconds=0, max_seconds=0),
            )
        )
    )

    assert result == Answer(value="durable")
    assert planning_calls == ["called", "called"]


def test_permanent_planning_failure_is_not_retried() -> None:
    planning_calls.clear()
    planning_failures[0] = 1
    planning_failure_kind[0] = "permanent"

    with pytest.raises(RuntimeError, match="transient planning failure"):
        asyncio.run(AgentWorkflow().run(AgentRequest(prompt="answer this")))

    assert planning_calls == ["called"]


def test_model_retry_attempts_and_backoff_are_bounded() -> None:
    planning_calls.clear()
    planning_failures[0] = 3
    planning_failure_kind[0] = "transient"
    config = BackoffConfig(
        attempts=2,
        initial_seconds=2,
        multiplier=3,
        max_seconds=5,
    )

    with pytest.raises(RuntimeError, match="failed after 2 transient attempts"):
        asyncio.run(
            AgentWorkflow().run(
                AgentRequest(
                    prompt="answer this",
                    model_retry=config.model_copy(update={"initial_seconds": 0}),
                )
            )
        )

    planning_failures[0] = 0
    assert planning_calls == ["called", "called"]
    assert asyncio.run(AgentWorkflow()._model_retry_backoff_seconds(config, 1)) == 2
    assert asyncio.run(AgentWorkflow()._model_retry_backoff_seconds(config, 2)) == 5


def test_waymark_executes_each_tool_as_its_own_action() -> None:
    tool_calls.clear()
    tool_failures[0] = 0

    result = asyncio.run(ToolWorkflow().run(ToolRequest(prompt="answer this")))

    assert result == '{"lookup":"found"}'
    assert tool_calls == ["a"]


def test_tool_action_policy_comes_from_tool_metadata() -> None:
    tool_calls.clear()
    tool_failures[0] = 1

    result = asyncio.run(ToolWorkflow().run(ToolRequest(prompt="answer this")))

    assert result == '{"lookup":"found"}'
    assert tool_calls == ["a", "a"]


def test_tool_can_request_a_durable_workflow_sleep() -> None:
    result = asyncio.run(SleepWorkflow().run(SleepRequest(prompt="answer this")))

    assert result == '{"pause":"waited"}'
    tool_ir = next(
        fn for fn in SleepWorkflow.workflow_ir().functions if fn.name == "_run_agent_tool_call"
    )
    tool_ir_text = str(tool_ir)
    assert "sleep_stmt" in tool_ir_text
    assert 'name: "sleep_seconds"' in tool_ir_text


def test_tools_parallelize_around_sequential_barriers() -> None:
    parallel_events.clear()

    asyncio.run(ParallelWorkflow().run(ParallelRequest(prompt="run tools")))

    parallel_ir = next(
        fn
        for fn in ParallelWorkflow.workflow_ir().functions
        if fn.name == "_run_agent_tool_segment"
    )
    assert "spread_expr" in str(parallel_ir)
    assert parallel_events.index("barrier:start") > max(
        parallel_events.index("before_a:end"),
        parallel_events.index("before_b:end"),
    )
    assert parallel_events.index("after:start") > parallel_events.index("barrier:end")


def test_request_serializes_agent_by_module_variable() -> None:
    payload = json.loads(AgentRequest(prompt="answer this").to_json())
    run_types = get_type_hints(AgentWorkflow.run)

    assert payload["agent_reference"] == f"{__name__}:test_agent"
    assert run_types == {"request": AgentRequest, "return": Answer}
    assert "run" not in PydanticAIWorkflow.__dict__
    assert "run_workflow_tool" not in PydanticAIWorkflow.__dict__


def test_union_request_routes_to_its_declared_agent() -> None:
    tool_calls.clear()

    result = asyncio.run(UnionWorkflow().run(ToolRequest(prompt="answer this")))

    assert result == '{"lookup":"found"}'
    assert tool_calls == ["a"]


def test_factory_requires_a_name() -> None:
    with pytest.raises(ValueError, match="needs a name"):
        exec(
            compile(
                "unnamed = waymark_agent(Agent(TestModel()))",
                __file__,
                "exec",
            ),
            {
                "__name__": __name__,
                "Agent": Agent,
                "TestModel": TestModel,
                "waymark_agent": waymark_agent,
            },
        )


def test_factory_requires_module_scope() -> None:
    def build_agent() -> None:
        waymark_agent(Agent(TestModel(), name="nested"))

    with pytest.raises(RuntimeError, match="module level"):
        build_agent()
