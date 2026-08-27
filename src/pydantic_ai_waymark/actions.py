import asyncio
from typing import Any, cast

from pydantic_ai import ModelMessagesTypeAdapter, _agent_graph
from pydantic_ai.exceptions import (
    ApprovalRequired,
    CallDeferred,
    ModelRetry,
    ToolFailed,
    ToolFailedError,
    ToolRetryError,
)
from pydantic_graph import End
from waymark import action

from .durability import PendingToolCallsError, tool_boundary
from .registry import registered_agent
from .serialization import (
    deferred_results,
    dump_deps_state,
    dump_node,
    dump_tool_call,
    load_node,
    load_tool_call,
    restore_deps_state,
    restore_graph_state,
    state_adapter,
    usage_adapter,
)
from .types import AgentResult, AgentTransition, ToolActionResult


@action(name="pydantic_ai_agent_node")
async def run_agent_node(
    agent_name: str,
    prompt: str | None,
    transition: AgentTransition | None = None,
    tool_results: list[ToolActionResult] | None = None,
    *,
    message_history: str | None = None,
    deps: Any = None,
    model: str | None = None,
    conversation_id: str | None = None,
    run_id: str | None = None,
) -> AgentTransition:
    """Run exactly one Pydantic AI graph node as a Waymark action."""
    agent = registered_agent(agent_name)
    restored_state = (
        state_adapter.validate_json(transition["state"]) if transition is not None else None
    )
    history = (
        restored_state.message_history
        if restored_state is not None
        else ModelMessagesTypeAdapter.validate_json(message_history)
        if message_history is not None
        else None
    )

    async with agent.iter(
        prompt,
        message_history=history,
        deps=deps,
        model=model,
        conversation_id=conversation_id if transition is None else None,
        run_id=run_id if transition is None else None,
        infer_name=False,
        capabilities=[tool_boundary],
    ) as agent_run:
        node: _agent_graph.AgentNode[Any, Any]
        if restored_state is None:
            node = cast(_agent_graph.AgentNode[Any, Any], agent_run.next_node)
        else:
            assert transition is not None
            restore_graph_state(agent_run, transition)
            await restore_deps_state(agent_run, transition["deps_state"])
            node = load_node(transition["node"])

        if tool_results:
            if not isinstance(node, _agent_graph.CallToolsNode):
                raise RuntimeError("tool results can only resume a call-tools node")
            node.tool_call_results = deferred_results(tool_results)
            node.tool_call_metadata = transition["tool_metadata"] if transition is not None else {}

        try:
            next_node = await agent_run.next(node)
        except PendingToolCallsError as pending:
            tool_metadata = {}
            for tool_call_id, metadata in pending.requests.metadata.items():
                remaining = {
                    key: value
                    for key, value in metadata.items()
                    if key not in {"waymark", "waymark_args"}
                }
                if remaining:
                    tool_metadata[tool_call_id] = remaining
            return {
                "kind": "tools",
                "result": None,
                "state": state_adapter.dump_json(agent_run.ctx.state).decode(),
                "node": dump_node(node),
                "deps_state": dump_deps_state(agent_run),
                "tool_calls": [
                    {
                        "call": dump_tool_call(call),
                        "waymark": pending.requests.metadata[call.tool_call_id]["waymark"],
                        "workflow_args": pending.requests.metadata[call.tool_call_id].get(
                            "waymark_args"
                        ),
                    }
                    for call in pending.requests.calls
                ],
                "approvals": [dump_tool_call(call) for call in pending.requests.approvals],
                "tool_metadata": tool_metadata,
            }

        if isinstance(next_node, End):
            result = agent_run.result
            assert result is not None
            completed: AgentResult = {
                "output": result.output,
                "message_history": result.all_messages_json().decode(),
                "new_messages": result.new_messages_json().decode(),
                "usage": usage_adapter.dump_python(result.usage, mode="json"),
                "run_id": result.run_id,
                "conversation_id": result.conversation_id,
            }
            return {
                "kind": "done",
                "result": completed,
                "state": None,
                "node": None,
                "deps_state": None,
                "tool_calls": [],
                "approvals": [],
                "tool_metadata": {},
            }

        return {
            "kind": "node",
            "result": None,
            "state": state_adapter.dump_json(agent_run.ctx.state).decode(),
            "node": dump_node(next_node),
            "deps_state": dump_deps_state(agent_run),
            "tool_calls": [],
            "approvals": [],
            "tool_metadata": {},
        }


@action(name="pydantic_ai_agent_tool")
async def run_agent_tool(
    agent_name: str,
    transition: AgentTransition,
    tool_call: dict[str, Any],
    *,
    deps: Any = None,
    model: str | None = None,
) -> ToolActionResult:
    """Execute one validated Pydantic AI tool call as a Waymark action."""
    agent = registered_agent(agent_name)
    state = state_adapter.validate_json(transition["state"])
    node = load_node(transition["node"])
    assert isinstance(node, _agent_graph.CallToolsNode)
    call = load_tool_call(tool_call["call"])

    async with agent.iter(
        node.user_prompt,
        message_history=state.message_history,
        deps=deps,
        model=model,
        infer_name=False,
    ) as agent_run:
        restore_graph_state(agent_run, transition)
        await restore_deps_state(agent_run, transition["deps_state"])
        metadata = transition["tool_metadata"].get(call.tool_call_id)
        try:
            async with asyncio.timeout(tool_call["waymark"]["timeout"]):
                value = await agent_run.ctx.deps.tool_manager.handle_call(call, metadata=metadata)
        except ToolRetryError as error:
            return {
                "kind": "retry_prompt",
                "tool_call_id": call.tool_call_id,
                "tool_name": call.tool_name,
                "content": error.tool_retry.content,
            }
        except ToolFailedError as error:
            return {
                "kind": "tool_failed",
                "tool_call_id": call.tool_call_id,
                "tool_name": call.tool_name,
                "message": str(error.tool_failed.content),
            }
        except ModelRetry as error:
            return {
                "kind": "model_retry",
                "tool_call_id": call.tool_call_id,
                "tool_name": call.tool_name,
                "message": error.message,
            }
        except ToolFailed as error:
            return {
                "kind": "tool_failed",
                "tool_call_id": call.tool_call_id,
                "tool_name": call.tool_name,
                "message": error.message,
            }
        except (CallDeferred, ApprovalRequired):
            return {
                "kind": "deferred",
                "tool_call_id": call.tool_call_id,
                "tool_name": call.tool_name,
            }
        return {
            "kind": "return",
            "tool_call_id": call.tool_call_id,
            "tool_name": call.tool_name,
            "value": value,
        }


@action(name="pydantic_ai_agent_tool_attempts_exhausted")
async def raise_tool_attempts_exhausted(tool_name: str, attempts: int) -> ToolActionResult:
    raise RuntimeError(f"tool {tool_name!r} failed after {attempts} Waymark attempts")


@action(name="pydantic_ai_workflow_tool_not_configured")
async def raise_workflow_tool_not_configured(agent_name: str, tool_name: str) -> Any:
    raise RuntimeError(
        f"workflow tool {tool_name!r} for agent {agent_name!r} has no compiled handler"
    )


@action(name="pydantic_ai_agent_approval_required")
async def raise_approval_required(agent_name: str) -> AgentResult:
    raise RuntimeError(f"agent {agent_name!r} requested human approval")
