import dataclasses
import inspect
from collections.abc import Callable
from dataclasses import replace
from typing import Any, Literal, cast

from mountaineer_di import provide_dependencies
from pydantic import TypeAdapter
from pydantic_ai import ModelMessagesTypeAdapter, RunUsage, _agent_graph
from pydantic_ai.exceptions import ModelRetry, ToolFailed
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    ToolCallPart,
    tool_return_content_ta,
)
from pydantic_ai.tools import DeferredToolResult, DeferredToolResults

from .registry import registered_codecs
from .types import (
    AgentNodePayload,
    CallToolsNodePayload,
    DeferredToolResultsPayload,
    DepsState,
    GraphStatePayload,
    MessagesPayload,
    ModelRequestNodePayload,
    Payload,
    PayloadKind,
    PersistedPydanticNode,
    RegisteredAgent,
    RegisteredAgentRun,
    SerializedPayload,
    SerializedToolCall,
    ToolActionResult,
    ToolMetadata,
    UserPrompt,
    UserPromptPayload,
)

state_adapter = TypeAdapter(_agent_graph.GraphAgentState)
usage_adapter = TypeAdapter(RunUsage)
tool_call_adapter = TypeAdapter(ToolCallPart)
wire_adapter = TypeAdapter(Any)
payload_adapter = TypeAdapter(Payload)
user_prompt_adapter = TypeAdapter(UserPrompt)
tool_results_adapter = TypeAdapter(dict[str, DeferredToolResult | Literal["skip"]])


def restore_agent_deps(agent: RegisteredAgent, deps: object) -> object:
    if deps is None or isinstance(deps, agent.deps_type):
        return deps
    return TypeAdapter(agent.deps_type).validate_python(deps)


async def _transform_payload(agent_name: str, value: object, *, serialize: bool) -> object:
    codecs = registered_codecs(agent_name)
    if codecs is None:
        return value
    serializer, serializer_parameter, deserializer, deserializer_parameter = codecs
    codec, parameter = (
        (serializer, serializer_parameter) if serialize else (deserializer, deserializer_parameter)
    )
    async with provide_dependencies(codec, {parameter: value}) as kwargs:
        kwargs.pop(parameter)
        result = cast(Callable[..., object], codec)(value, **kwargs)
        return await result if inspect.isawaitable(result) else result


async def serialize_payload(agent_name: str, payload: Payload) -> Any:
    """Transform one value before it crosses a durable action boundary."""
    if registered_codecs(agent_name) is None:
        return payload.value
    result = SerializedPayload.model_validate(
        await _transform_payload(agent_name, payload, serialize=True)
    )
    if result.kind != payload.kind:
        raise ValueError(
            f"serializer changed payload kind from {payload.kind!r} to {result.kind!r}"
        )
    return result.value


async def deserialize_payload(agent_name: str, kind: PayloadKind, value: Any) -> Any:
    """Restore one value after it crosses a durable action boundary."""
    if registered_codecs(agent_name) is None:
        return value
    payload = SerializedPayload(kind=kind, value=value)
    result = payload_adapter.validate_python(
        await _transform_payload(agent_name, payload, serialize=False)
    )
    if result.kind != kind:
        raise ValueError(f"deserializer changed payload kind from {kind!r} to {result.kind!r}")
    return result.value


async def _dump_json(agent_name: str, payload: Payload, adapter: Any) -> str:
    if registered_codecs(agent_name) is None:
        return adapter.dump_json(payload.value).decode()
    transformed = await serialize_payload(agent_name, payload)
    return wire_adapter.dump_json(transformed).decode()


async def _load_json(
    agent_name: str,
    kind: PayloadKind,
    value: str,
    adapter: Any,
) -> Any:
    if registered_codecs(agent_name) is None:
        return adapter.validate_json(value)
    transformed = await deserialize_payload(
        agent_name,
        kind,
        wire_adapter.validate_json(value),
    )
    return adapter.validate_python(transformed)


async def dump_messages(agent_name: str, messages: list[ModelMessage]) -> str:
    return await _dump_json(agent_name, MessagesPayload(value=messages), ModelMessagesTypeAdapter)


async def load_messages(agent_name: str, value: str) -> list[ModelMessage]:
    return await _load_json(agent_name, "messages", value, ModelMessagesTypeAdapter)


async def load_message_history(agent_name: str, chunks: list[str] | None) -> list[ModelMessage]:
    history: list[ModelMessage] = []
    for chunk in chunks or []:
        history.extend(await load_messages(agent_name, chunk))
    return history


async def dump_message(agent_name: str, message: ModelMessage) -> str:
    return await dump_messages(agent_name, [message])


async def load_message(agent_name: str, value: str) -> ModelMessage:
    messages = await load_messages(agent_name, value)
    if len(messages) != 1:
        raise ValueError("expected exactly one serialized model message")
    return messages[0]


async def dump_graph_state(
    agent_name: str,
    state: _agent_graph.GraphAgentState,
) -> str:
    return await _dump_json(
        agent_name,
        GraphStatePayload(value=replace(state, message_history=[])),
        state_adapter,
    )


async def load_graph_state(agent_name: str, value: str) -> _agent_graph.GraphAgentState:
    return await _load_json(agent_name, "graph_state", value, state_adapter)


def dump_tool_call(call: ToolCallPart) -> SerializedToolCall:
    return SerializedToolCall.model_validate(tool_call_adapter.dump_python(call, mode="json"))


def load_tool_call(value: SerializedToolCall) -> ToolCallPart:
    return tool_call_adapter.validate_python(value.model_dump(mode="json"))


async def dump_node(agent_name: str, node: PersistedPydanticNode) -> AgentNodePayload:
    if isinstance(node, _agent_graph.ModelRequestNode):
        return ModelRequestNodePayload(
            kind="model_request",
            request=await dump_message(agent_name, node.request),
            is_resuming_without_prompt=node.is_resuming_without_prompt,
            resume_suspended=(
                await dump_message(agent_name, node._resume_suspended)
                if node._resume_suspended is not None
                else None
            ),
        )
    if isinstance(node, _agent_graph.CallToolsNode):
        return CallToolsNodePayload(
            kind="call_tools",
            model_response=await dump_message(agent_name, node.model_response),
            tool_call_results=await serialize_payload(
                agent_name,
                DeferredToolResultsPayload(value=node.tool_call_results),
            ),
            tool_call_metadata=cast(ToolMetadata | None, node.tool_call_metadata),
            user_prompt=await serialize_payload(
                agent_name,
                UserPromptPayload(value=node.user_prompt),
            ),
        )
    raise TypeError(f"unsupported Pydantic AI graph node: {type(node).__name__}")


async def load_node(agent_name: str, value: AgentNodePayload) -> PersistedPydanticNode:
    match value.kind:
        case "model_request":
            assert isinstance(value, ModelRequestNodePayload)
            suspended = value.resume_suspended
            return _agent_graph.ModelRequestNode(
                request=cast(ModelRequest, await load_message(agent_name, value.request)),
                is_resuming_without_prompt=value.is_resuming_without_prompt,
                _resume_suspended=(
                    cast(ModelResponse, await load_message(agent_name, suspended))
                    if suspended is not None
                    else None
                ),
            )
        case "call_tools":
            assert isinstance(value, CallToolsNodePayload)
            tool_call_results = await deserialize_payload(
                agent_name,
                "deferred_tool_results",
                value.tool_call_results,
            )
            return _agent_graph.CallToolsNode(
                model_response=cast(
                    ModelResponse,
                    await load_message(agent_name, value.model_response),
                ),
                tool_call_results=(
                    tool_results_adapter.validate_python(tool_call_results)
                    if tool_call_results is not None
                    else None
                ),
                tool_call_metadata=value.tool_call_metadata,
                user_prompt=user_prompt_adapter.validate_python(
                    await deserialize_payload(
                        agent_name,
                        "user_prompt",
                        value.user_prompt,
                    )
                ),
            )
        case kind:
            raise ValueError(f"unknown Pydantic AI graph node kind: {kind!r}")


async def dump_deps_state(agent_name: str, agent_run: RegisteredAgentRun) -> DepsState:
    deps = agent_run.ctx.deps
    tool_manager = deps.tool_manager
    tool_context = tool_manager.ctx
    return DepsState(
        new_message_index=deps.new_message_index,
        resumed_request=(
            await dump_message(agent_name, deps.resumed_request)
            if deps.resumed_request is not None
            else None
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


async def restore_deps_state(
    agent_name: str,
    agent_run: RegisteredAgentRun,
    value: DepsState,
) -> None:
    deps = agent_run.ctx.deps
    deps.new_message_index = value.new_message_index
    serialized_request = value.resumed_request
    deps.resumed_request = (
        cast(ModelRequest, await load_message(agent_name, serialized_request))
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
    restored_state: _agent_graph.GraphAgentState,
) -> _agent_graph.GraphAgentState:
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
            # NOTE: Work around Waymark returning persisted action values as JSON-compatible data.
            calls[tool_call_id] = tool_return_content_ta.validate_python(result["value"])
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
