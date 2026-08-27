import dataclasses
from dataclasses import replace
from typing import Literal, cast

from pydantic import TypeAdapter
from pydantic_ai import ModelMessagesTypeAdapter, RunUsage, _agent_graph
from pydantic_ai.exceptions import ModelRetry, ToolFailed
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    ToolCallPart,
)
from pydantic_ai.tools import DeferredToolResult, DeferredToolResults

from .types import (
    AgentNodePayload,
    CallToolsNodePayload,
    DepsState,
    ModelRequestNodePayload,
    PendingTransition,
    PersistedPydanticNode,
    RegisteredAgent,
    RegisteredAgentRun,
    SerializedToolCall,
    ToolActionResult,
    ToolMetadata,
)

state_adapter = TypeAdapter(_agent_graph.GraphAgentState)
usage_adapter = TypeAdapter(RunUsage)
tool_call_adapter = TypeAdapter(ToolCallPart)


def restore_agent_deps(agent: RegisteredAgent, deps: object) -> object:
    if deps is None or isinstance(deps, agent.deps_type):
        return deps
    return TypeAdapter(agent.deps_type).validate_python(deps)


def dump_message(message: ModelMessage) -> str:
    return ModelMessagesTypeAdapter.dump_json([message]).decode()


def load_message(value: str) -> ModelMessage:
    messages = ModelMessagesTypeAdapter.validate_json(value)
    if len(messages) != 1:
        raise ValueError("expected exactly one serialized model message")
    return messages[0]


def dump_tool_call(call: ToolCallPart) -> SerializedToolCall:
    return SerializedToolCall.model_validate(tool_call_adapter.dump_python(call, mode="json"))


def load_tool_call(value: SerializedToolCall) -> ToolCallPart:
    return tool_call_adapter.validate_python(value.model_dump(mode="json"))


def dump_node(node: PersistedPydanticNode) -> AgentNodePayload:
    if isinstance(node, _agent_graph.ModelRequestNode):
        return ModelRequestNodePayload(
            kind="model_request",
            request=dump_message(node.request),
            is_resuming_without_prompt=node.is_resuming_without_prompt,
            resume_suspended=(
                dump_message(node._resume_suspended) if node._resume_suspended is not None else None
            ),
        )
    if isinstance(node, _agent_graph.CallToolsNode):
        return CallToolsNodePayload(
            kind="call_tools",
            model_response=dump_message(node.model_response),
            tool_call_results=node.tool_call_results,
            tool_call_metadata=cast(ToolMetadata | None, node.tool_call_metadata),
            user_prompt=node.user_prompt,
        )
    raise TypeError(f"unsupported Pydantic AI graph node: {type(node).__name__}")


def load_node(value: AgentNodePayload) -> PersistedPydanticNode:
    match value.kind:
        case "model_request":
            assert isinstance(value, ModelRequestNodePayload)
            suspended = value.resume_suspended
            return _agent_graph.ModelRequestNode(
                request=cast(ModelRequest, load_message(value.request)),
                is_resuming_without_prompt=value.is_resuming_without_prompt,
                _resume_suspended=(
                    cast(ModelResponse, load_message(suspended)) if suspended is not None else None
                ),
            )
        case "call_tools":
            assert isinstance(value, CallToolsNodePayload)
            return _agent_graph.CallToolsNode(
                model_response=cast(ModelResponse, load_message(value.model_response)),
                tool_call_results=value.tool_call_results,
                tool_call_metadata=value.tool_call_metadata,
                user_prompt=value.user_prompt,
            )
        case kind:
            raise ValueError(f"unknown Pydantic AI graph node kind: {kind!r}")


def dump_deps_state(agent_run: RegisteredAgentRun) -> DepsState:
    deps = agent_run.ctx.deps
    tool_manager = deps.tool_manager
    tool_context = tool_manager.ctx
    return DepsState(
        new_message_index=deps.new_message_index,
        resumed_request=(
            dump_message(deps.resumed_request) if deps.resumed_request is not None else None
        ),
        resumed_request_index=deps.resumed_request_index,
        model_id=deps.model_id,
        model_selected_for_step=deps.model_selected_for_step,
        loaded_capability_ids=sorted(deps.loaded_capability_ids),
        discovered_tool_names=sorted(deps.discovered_tool_names),
        tool_run_step=tool_context.run_step if tool_context is not None else None,
        tool_retries=dict(tool_context.retries) if tool_context is not None else {},
        failed_tools=sorted(tool_manager.failed_tools),
        succeeded_tools=sorted(tool_manager.succeeded_tools),
    )


async def restore_deps_state(agent_run: RegisteredAgentRun, value: DepsState) -> None:
    deps = agent_run.ctx.deps
    deps.new_message_index = value.new_message_index
    serialized_request = value.resumed_request
    deps.resumed_request = (
        cast(ModelRequest, load_message(serialized_request))
        if serialized_request is not None
        else None
    )
    deps.resumed_request_index = value.resumed_request_index
    deps.model_id = value.model_id
    deps.model_selected_for_step = value.model_selected_for_step
    deps.loaded_capability_ids.clear()
    deps.loaded_capability_ids.update(value.loaded_capability_ids)
    deps.discovered_tool_names.clear()
    deps.discovered_tool_names.update(value.discovered_tool_names)

    tool_run_step = value.tool_run_step
    if tool_run_step is not None:
        run_context = replace(
            _agent_graph.build_run_context(agent_run.ctx),
            run_step=tool_run_step,
            retries=value.tool_retries,
        )
        tool_manager = await deps.tool_manager.for_run_step(run_context)
        tool_manager.failed_tools.update(value.failed_tools)
        tool_manager.succeeded_tools.update(value.succeeded_tools)
        deps.tool_manager = tool_manager


def restore_graph_state(
    agent_run: RegisteredAgentRun,
    transition: PendingTransition,
) -> _agent_graph.GraphAgentState:
    restored_state = state_adapter.validate_json(transition.state)
    for state_field in dataclasses.fields(restored_state):
        setattr(agent_run.ctx.state, state_field.name, getattr(restored_state, state_field.name))
    return restored_state


def deferred_results(
    results: list[ToolActionResult],
) -> dict[str, DeferredToolResult | Literal["skip"]]:
    calls: dict[str, object] = {}
    for result in results:
        tool_call_id = result["tool_call_id"]
        if result["kind"] == "return":
            calls[tool_call_id] = result["value"]
        elif result["kind"] == "model_retry":
            calls[tool_call_id] = ModelRetry(result["message"])
        elif result["kind"] == "tool_failed":
            calls[tool_call_id] = ToolFailed(result["message"])
        elif result["kind"] == "retry_prompt":
            calls[tool_call_id] = RetryPromptPart(
                content=result["content"],
                tool_name=result["tool_name"],
                tool_call_id=tool_call_id,
            )
        else:
            raise RuntimeError(
                f"tool {result['tool_name']!r} remained deferred inside its Waymark action"
            )
    return cast(
        dict[str, DeferredToolResult | Literal["skip"]],
        DeferredToolResults(calls=calls).to_tool_call_results(),
    )
