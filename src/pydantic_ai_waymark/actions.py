from typing import Never, cast

from pydantic import TypeAdapter
from pydantic_ai import ModelMessagesTypeAdapter, _agent_graph
from pydantic_ai.exceptions import (
    ApprovalRequired,
    CallDeferred,
    ModelAPIError,
    ModelHTTPError,
    ModelRetry,
    ToolFailed,
    ToolFailedError,
    ToolRetryError,
)
from pydantic_ai.messages import ToolCallPart
from pydantic_graph import End
from waymark import action

from .durability import DurableSleep, PendingToolCallsError, tool_boundary
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
from .types import (
    AgentDeps,
    AgentOutput,
    AgentResult,
    AgentTransition,
    DoneTransition,
    JsonValue,
    NodeTransition,
    PendingTransition,
    PersistedPydanticNode,
    PydanticRunNode,
    ToolActionResult,
    ToolCall,
    ToolMetadata,
    ToolsTransition,
    UsagePayload,
)

tool_metadata_adapter = TypeAdapter(dict[str, JsonValue])


class RetryableAgentError(RuntimeError):
    """A transient provider failure that may safely rerun its graph-node action."""


def _pending_tool_call(call: ToolCallPart, *, sequential: bool) -> ToolCall:
    return ToolCall(
        call=dump_tool_call(call),
        sequential=sequential,
    )


@action(name="pydantic_ai_agent_node")
async def run_agent_node(
    agent_name: str,
    prompt: str | None,
    transition: AgentTransition | None = None,
    tool_results: list[ToolActionResult] | None = None,
    *,
    message_history: str | None = None,
    deps: AgentDeps = None,
    model: str | None = None,
    conversation_id: str | None = None,
    run_id: str | None = None,
) -> AgentTransition:
    """Run exactly one Pydantic AI graph node as a Waymark action."""
    if transition is not None and transition.kind == "done":
        raise RuntimeError("a completed agent transition cannot be resumed")
    agent = registered_agent(agent_name)
    restored_state = (
        state_adapter.validate_json(transition.state) if transition is not None else None
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
        node: PydanticRunNode
        if restored_state is None:
            node = cast(PydanticRunNode, agent_run.next_node)
        else:
            assert transition is not None
            restore_graph_state(agent_run, transition)
            await restore_deps_state(agent_run, transition.deps_state)
            node = load_node(transition.node)

        if tool_results:
            if not isinstance(node, _agent_graph.CallToolsNode):
                raise RuntimeError("tool results can only resume a call-tools node")
            call_tools_node = cast(
                _agent_graph.CallToolsNode[AgentDeps, AgentOutput],
                node,
            )
            call_tools_node.tool_call_results = deferred_results(tool_results)
            call_tools_node.tool_call_metadata = (
                transition.tool_metadata if transition is not None else {}
            )
            # Supplied deferred results bypass Pydantic AI's local tool executor, so
            # mirror its retry bookkeeping before advancing to the next run step.
            for result in tool_results:
                if result["kind"] in {"retry_prompt", "model_retry"}:
                    agent_run.ctx.deps.tool_manager.failed_tools.add(result["tool_name"])
                elif result["kind"] == "return":
                    agent_run.ctx.deps.tool_manager.succeeded_tools.add(result["tool_name"])

        try:
            next_node = await agent_run.next(node)
        except PendingToolCallsError as pending:
            tool_metadata: ToolMetadata = {}
            for tool_call_id, metadata in pending.requests.metadata.items():
                if metadata:
                    tool_metadata[tool_call_id] = tool_metadata_adapter.validate_python(metadata)
            return ToolsTransition(
                kind="tools",
                state=state_adapter.dump_json(agent_run.ctx.state).decode(),
                node=dump_node(cast(PersistedPydanticNode, node)),
                deps_state=dump_deps_state(agent_run),
                tool_calls=[
                    _pending_tool_call(
                        call,
                        sequential=agent_run.ctx.deps.tool_manager.is_sequential(call),
                    )
                    for call in pending.requests.calls
                ],
                approvals=[dump_tool_call(call) for call in pending.requests.approvals],
                tool_metadata=tool_metadata,
            )
        except ModelHTTPError as error:
            if error.status_code not in (408, 409, 429) and error.status_code < 500:
                raise
            raise RetryableAgentError(str(error)) from error
        except ModelAPIError as error:
            raise RetryableAgentError(str(error)) from error

        if isinstance(next_node, End):
            result = agent_run.result
            assert result is not None
            completed = AgentResult(
                output=result.output,
                message_history=result.all_messages_json().decode(),
                new_messages=result.new_messages_json().decode(),
                usage=UsagePayload.model_validate(
                    usage_adapter.dump_python(result.usage, mode="json")
                ),
                run_id=result.run_id,
                conversation_id=result.conversation_id,
            )
            return DoneTransition(
                kind="done",
                result=completed,
            )

        if not isinstance(next_node, (_agent_graph.ModelRequestNode, _agent_graph.CallToolsNode)):
            raise RuntimeError(f"unsupported next agent node: {type(next_node).__name__}")
        return NodeTransition(
            kind="node",
            state=state_adapter.dump_json(agent_run.ctx.state).decode(),
            node=dump_node(next_node),
            deps_state=dump_deps_state(agent_run),
        )


@action(name="pydantic_ai_agent_tool")
async def run_agent_tool(
    agent_name: str,
    transition: PendingTransition,
    tool_call: ToolCall,
    *,
    deps: AgentDeps = None,
    model: str | None = None,
) -> ToolActionResult:
    """Execute one validated Pydantic AI tool call as a Waymark action."""
    agent = registered_agent(agent_name)
    state = state_adapter.validate_json(transition.state)
    node = load_node(transition.node)
    assert isinstance(node, _agent_graph.CallToolsNode)
    call = load_tool_call(tool_call.call)

    async with agent.iter(
        node.user_prompt,
        message_history=state.message_history,
        deps=deps,
        model=model,
        infer_name=False,
    ) as agent_run:
        restore_graph_state(agent_run, transition)
        await restore_deps_state(agent_run, transition.deps_state)
        metadata = transition.tool_metadata.get(call.tool_call_id)
        try:
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
        except DurableSleep as sleep:
            return {
                "kind": "sleep",
                "tool_call_id": call.tool_call_id,
                "tool_name": call.tool_name,
                "seconds": sleep.seconds,
                "value": sleep.result,
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


@action(name="pydantic_ai_agent_model_attempts_exhausted")
async def raise_model_attempts_exhausted(attempts: int) -> Never:
    raise RuntimeError(f"model request failed after {attempts} transient attempts")


@action(name="pydantic_ai_parallel_tool_results")
async def resolve_parallel_tool_results(
    results: list[object],
) -> list[ToolActionResult]:
    for result in results:
        if isinstance(result, BaseException):
            raise result
    return cast(list[ToolActionResult], results)


@action(name="pydantic_ai_agent_approval_required")
async def raise_approval_required(agent_name: str) -> Never:
    raise RuntimeError(f"agent {agent_name!r} requested human approval")
