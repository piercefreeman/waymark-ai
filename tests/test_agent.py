import asyncio
import json
from typing import get_type_hints

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent, ModelMessagesTypeAdapter
from pydantic_ai.models.test import TestModel
from waymark import workflow

from pydantic_ai_waymark import (
    AgentResult,
    AIRequestBase,
    PydanticAIWorkflow,
    WorkflowToolArgs,
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
planning_calls: list[str] = []


@test_agent.instructions
def transient_planning() -> str:
    planning_calls.append("called")
    if planning_failures[0]:
        planning_failures[0] -= 1
        raise RuntimeError("transient planning failure")
    return "Answer the request."

tool_calls: list[str] = []
tool_failures = [0]
tool_agent = waymark_agent(Agent(TestModel(call_tools=["lookup"]), name="tool_agent"))
workflow_tool_agent = waymark_agent(
    Agent(TestModel(call_tools=["pause"]), name="workflow_tool_agent")
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


class AgentRequest(AIRequestBase[None]):
    agent = test_agent


class ToolRequest(AIRequestBase[None]):
    agent = tool_agent


class WorkflowToolRequest(AIRequestBase[None]):
    agent = workflow_tool_agent


@workflow
class AgentWorkflow(PydanticAIWorkflow[AgentRequest]):
    pass


@workflow
class ToolWorkflow(PydanticAIWorkflow[ToolRequest]):
    pass


@workflow
class WorkflowToolWorkflow(PydanticAIWorkflow[WorkflowToolRequest]):
    @staticmethod
    @workflow_tool_agent.tool_plain(metadata={"waymark": False})
    async def pause() -> str:
        seconds = 0.001
        await asyncio.sleep(seconds)
        return "waited"

    async def run_workflow_tool(
        self,
        agent_name: str,
        tool_name: str,
        args: WorkflowToolArgs,
    ) -> str:
        if tool_name == "pause":
            return await self.pause()
        return await self._unsupported_workflow_tool(agent_name, tool_name)

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


def test_planning_retries_without_a_hard_limit() -> None:
    planning_calls.clear()
    planning_failures[0] = 1

    result = asyncio.run(AgentWorkflow().run(AgentRequest(prompt="answer this")))

    assert result == Answer(value="durable")
    assert planning_calls == ["called", "called"]


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


def test_async_tool_can_run_as_compiled_workflow_logic() -> None:
    result = asyncio.run(
        WorkflowToolWorkflow().run(WorkflowToolRequest(prompt="answer this"))
    )

    assert result == '{"pause":"waited"}'
    pause_ir = next(
        fn for fn in WorkflowToolWorkflow.workflow_ir().functions if fn.name == "pause"
    )
    sleep = next(
        stmt.sleep_stmt for stmt in pause_ir.body.statements if stmt.HasField("sleep_stmt")
    )
    assert sleep.duration.variable.name == "seconds"


def test_request_serializes_agent_by_module_variable() -> None:
    payload = json.loads(AgentRequest(prompt="answer this").to_json())
    run_types = get_type_hints(AgentWorkflow.run)

    assert payload["agent_reference"] == f"{__name__}:test_agent"
    assert run_types == {"request": AgentRequest, "return": Answer}


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
