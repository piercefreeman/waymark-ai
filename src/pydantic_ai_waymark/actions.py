from collections.abc import Sequence
from typing import Any, Never, cast

from pydantic import TypeAdapter
from pydantic_ai import _agent_graph
from pydantic_ai.exceptions import (
    ApprovalRequired,
    CallDeferred,
    ModelAPIError,
    ModelHTTPError,
    ModelRetry,
    ToolFailed,
    ToolFailedError,
    ToolRetryError,
    UnexpectedModelBehavior,
)
from pydantic_ai.messages import TextPart, ToolCallPart
from pydantic_core import ErrorDetails, to_json
from pydantic_graph import End
from waymark import action

from .deferred_calls import (
    DelegateAgent,
    DurableSleep,
    PendingToolCallsError,
    WaitForMessages,
    subagent_mailbox,
    tool_boundary,
)
from .registry import registered_agent, registered_message_reader, registered_message_writer
from .request import AIRequestBase, load_request, request_type_reference
from .serialization import (
    bind_node_to_history,
    call_with_dependencies,
    deferred_results,
    deserialize_payload,
    dump_deps_state,
    dump_graph_state,
    dump_messages,
    dump_node,
    dump_tool_call,
    load_graph_state,
    load_message_history,
    load_node,
    load_tool_call,
    restore_agent_deps,
    restore_deps_state,
    restore_graph_state,
    serialize_payload,
    usage_adapter,
)
from .types import (
    AgentCheckpoint,
    AgentDeps,
    AgentMessage,
    AgentOutput,
    AgentOutputPayload,
    AgentResult,
    AgentTransition,
    DelegateAgentResult,
    DoneTransition,
    JsonValue,
    MessageQuery,
    NodeTransition,
    OutgoingMessage,
    PersistedPydanticNode,
    PydanticRunNode,
    ToolActionResult,
    ToolCall,
    ToolMetadata,
    ToolOutputPayload,
    ToolReturnResult,
    ToolsTransition,
    UsagePayload,
    WaitForMessagesResult,
)

tool_metadata_adapter = TypeAdapter(dict[str, JsonValue])
tool_action_results_adapter = TypeAdapter(list[ToolActionResult])
agent_messages_adapter = TypeAdapter(list[AgentMessage])


class RetryableAgentError(RuntimeError):
    """A transient provider failure that may safely rerun its graph-node action."""


def _pending_tool_call(call: ToolCallPart, *, tool_manager: Any) -> ToolCall:
    return ToolCall(
        call=dump_tool_call(call),
        sequential=tool_manager.is_sequential(call),
        max_retries=_tool_max_retries(tool_manager, call.tool_name),
    )


def _tool_max_retries(tool_manager: Any, tool_name: str) -> int:
    """The retry budget Pydantic AI holds this tool to: its own `retries=`, else the
    agent's default (also what an unknown tool name is charged against)."""
    tools = tool_manager.tools or {}
    tool = tools.get(tool_name)
    if tool is not None:
        return tool.max_retries
    return tool_manager.default_max_retries


@action(name="pydantic_ai_agent_node")
async def run_agent_node(
    agent_name: str,
    prompt: str | None,
    checkpoint: AgentCheckpoint,
    tool_results: list[ToolActionResult] | None = None,
    *,
    deps: AgentDeps = None,
    model: str | None = None,
    conversation_id: str | None = None,
    run_id: str | None = None,
) -> AgentTransition:
    """Run exactly one Pydantic AI graph node as a Waymark action."""
    agent = registered_agent(agent_name)
    deps = restore_agent_deps(agent, deps)
    restored_state = (
        await load_graph_state(agent_name, checkpoint.state)
        if checkpoint.state is not None
        else None
    )
    history = await load_message_history(agent_name, checkpoint.message_history)
    if restored_state is not None:
        restored_state.message_history = history
    history_size = len(history)

    async with agent.iter(
        prompt,
        message_history=history or None,
        deps=deps,
        model=model,
        conversation_id=conversation_id if checkpoint.state is None else None,
        run_id=run_id if checkpoint.state is None else None,
        infer_name=False,
        capabilities=[tool_boundary],
    ) as agent_run:
        node: PydanticRunNode
        if restored_state is None:
            node = cast(PydanticRunNode, agent_run.next_node)
        else:
            if checkpoint.node is None or checkpoint.deps_state is None:
                raise RuntimeError("resumed agent checkpoint is incomplete")
            restore_graph_state(agent_run, restored_state)
            await restore_deps_state(agent_name, agent_run, checkpoint.deps_state)
            node = bind_node_to_history(
                await load_node(agent_name, checkpoint.node),
                agent_run.ctx.state.message_history,
            )

        if tool_results:
            if not isinstance(node, _agent_graph.CallToolsNode):
                raise RuntimeError("tool results can only resume a call-tools node")
            call_tools_node = cast(
                _agent_graph.CallToolsNode[AgentDeps, AgentOutput],
                node,
            )
            restored_tool_results = tool_action_results_adapter.validate_python(
                await deserialize_payload(
                    agent_name,
                    "tool_action_results",
                    tool_results,
                )
            )
            call_tools_node.tool_call_results = deferred_results(restored_tool_results)
            call_tools_node.tool_call_metadata = checkpoint.tool_metadata
            # Supplied deferred results bypass Pydantic AI's local tool executor, so
            # mirror its retry bookkeeping before advancing to the next run step.
            for result in restored_tool_results:
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
                state=await dump_graph_state(agent_name, agent_run.ctx.state),
                node=await dump_node(agent_name, cast(PersistedPydanticNode, node)),
                deps_state=await dump_deps_state(agent_name, agent_run),
                history_delta=(
                    await dump_messages(
                        agent_name,
                        agent_run.ctx.state.message_history[history_size:],
                    )
                    if len(agent_run.ctx.state.message_history) > history_size
                    else None
                ),
                tool_calls=[
                    _pending_tool_call(
                        call,
                        tool_manager=agent_run.ctx.deps.tool_manager,
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
                output=await serialize_payload(
                    agent_name,
                    AgentOutputPayload(value=result.output),
                ),
                message_history=await dump_messages(agent_name, result.all_messages()),
                new_messages=await dump_messages(agent_name, result.new_messages()),
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
            state=await dump_graph_state(agent_name, agent_run.ctx.state),
            node=await dump_node(agent_name, next_node),
            deps_state=await dump_deps_state(agent_name, agent_run),
            history_delta=(
                await dump_messages(
                    agent_name,
                    agent_run.ctx.state.message_history[history_size:],
                )
                if len(agent_run.ctx.state.message_history) > history_size
                else None
            ),
            messages=(
                [
                    part.content
                    for part in next_node.model_response.parts
                    if isinstance(part, TextPart) and part.content
                ]
                if isinstance(next_node, _agent_graph.CallToolsNode)
                else []
            ),
        )


@action(name="pydantic_ai_agent_tool")
async def run_agent_tool(
    agent_name: str,
    checkpoint: AgentCheckpoint,
    tool_call: ToolCall,
    *,
    deps: AgentDeps = None,
    model: str | None = None,
    prompt: str | None = None,
) -> ToolActionResult:
    """Execute one validated Pydantic AI tool call as a Waymark action.

    ``prompt`` is the run's user prompt, so tools see it as ``ctx.prompt``; a restored
    call-tools node does not carry it. It defaults to ``None`` so in-flight runs whose
    workflows were compiled without it still resume.
    """
    agent = registered_agent(agent_name)
    deps = restore_agent_deps(agent, deps)
    if checkpoint.state is None or checkpoint.node is None or checkpoint.deps_state is None:
        raise RuntimeError("tool agent checkpoint is incomplete")
    state = await load_graph_state(agent_name, checkpoint.state)
    state.message_history = await load_message_history(agent_name, checkpoint.message_history)
    node = bind_node_to_history(
        await load_node(agent_name, checkpoint.node),
        state.message_history,
    )
    assert isinstance(node, _agent_graph.CallToolsNode)
    call = load_tool_call(tool_call.call)

    async with agent.iter(
        prompt if prompt is not None else node.user_prompt,
        message_history=state.message_history,
        deps=deps,
        model=model,
        infer_name=False,
    ) as agent_run:
        restore_graph_state(agent_run, state)
        await restore_deps_state(agent_name, agent_run, checkpoint.deps_state)
        metadata = checkpoint.tool_metadata.get(call.tool_call_id)
        try:
            value = await agent_run.ctx.deps.tool_manager.handle_call(call, metadata=metadata)
        except ToolRetryError as error:
            # NOTE: Newer Pydantic AI releases carry retries as `ToolReturnPart` or
            # `RetryFeedbackPart`, whose content is not narrowed to retry feedback.
            retry_content: object = error.tool_retry.content
            return {
                "kind": "retry_prompt",
                "tool_call_id": call.tool_call_id,
                "tool_name": call.tool_name,
                "content": (
                    cast(str | list[ErrorDetails], retry_content)
                    if isinstance(retry_content, (str, list))
                    else str(retry_content)
                ),
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
                "value": await serialize_payload(
                    agent_name,
                    ToolOutputPayload(value=sleep.result),
                ),
            }
        except DelegateAgent as delegation:
            if not delegation.wait and registered_message_writer(agent_name) is None:
                raise RuntimeError(
                    f"agent {agent_name!r} starts a detached subagent but registers no "
                    "message_writer; pass message_writer= to waymark_agent()"
                ) from None
            subagent_id = f"{agent_run.ctx.state.run_id}:{call.tool_call_id}"
            return {
                "kind": "delegate",
                "tool_call_id": call.tool_call_id,
                "tool_name": call.tool_name,
                "request": delegation.request,
                "request_type": request_type_reference(type(delegation.request)),
                "wait": delegation.wait,
                "subagent_id": subagent_id,
                "mailbox": subagent_mailbox(subagent_id),
            }
        except WaitForMessages as wait:
            if registered_message_reader(agent_name) is None:
                raise RuntimeError(
                    f"agent {agent_name!r} waits for messages but registers no "
                    "message_reader; pass message_reader= to waymark_agent()"
                ) from None
            return {
                "kind": "wait_messages",
                "tool_call_id": call.tool_call_id,
                "tool_name": call.tool_name,
                "mailbox": wait.mailbox,
                "run_id": agent_run.ctx.state.run_id,
                # Tool-call IDs are only unique within a run, so scope claims to it.
                "claim_key": f"{agent_run.ctx.state.run_id}:{call.tool_call_id}",
                "poll_seconds": wait.poll_seconds,
                "timeout_seconds": wait.timeout_seconds,
                "timeout_value": await serialize_payload(
                    agent_name,
                    ToolOutputPayload(value=wait.timeout_result),
                ),
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
            "value": await serialize_payload(
                agent_name,
                ToolOutputPayload(value=value),
            ),
        }


@action(name="pydantic_ai_agent_delegate_result")
async def delegate_tool_result(
    agent_name: str,
    subagent_name: str,
    tool_call_id: str,
    tool_name: str,
    output: AgentOutput,
) -> ToolReturnResult:
    """Turn a finished subagent's output into the delegating tool call's return value.

    The two agents may register different payload codecs, so the output is restored with
    the subagent's codec and stored again with the delegating agent's.
    """
    value = await deserialize_payload(subagent_name, "agent_output", output)
    return {
        "kind": "return",
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
        "value": await serialize_payload(agent_name, ToolOutputPayload(value=value)),
    }


@action(name="pydantic_ai_agent_start_detached_subagent")
async def start_detached_subagent(
    agent_name: str,
    delegation: DelegateAgentResult,
) -> ToolReturnResult:
    """Queue a detached subagent as its own workflow and return its handle to the model.

    NOTE: Waymark does not deduplicate workflow instances, so a rerun of this action after
    the instance was queued starts the subagent twice; its result is still delivered once
    because the report's ``dedupe_key`` is the subagent id.
    """
    from .subagents import DetachedSubagentRequest, SubagentWorkflow

    # NOTE: `@workflow` adds `_blocking` to the public `run`, which type checkers can't see.
    subagent_workflow: Any = SubagentWorkflow()
    await subagent_workflow.run(
        DetachedSubagentRequest(
            # NOTE: Waymark passes this nested request as a plain dict; the subagent
            # workflow rebuilds its concrete class from `request_type`.
            request=_request_data(delegation["request"]),
            request_type=delegation["request_type"],
            parent_agent=agent_name,
            subagent_id=delegation["subagent_id"],
            mailbox=delegation["mailbox"],
        ),
        _blocking=False,
    )
    return {
        "kind": "return",
        "tool_call_id": delegation["tool_call_id"],
        "tool_name": delegation["tool_name"],
        "value": await serialize_payload(
            agent_name,
            ToolOutputPayload(
                value={
                    "subagent_id": delegation["subagent_id"],
                    "mailbox": delegation["mailbox"],
                    "status": "started",
                }
            ),
        ),
    }


def _request_data(request: object) -> dict[str, JsonValue]:
    if isinstance(request, AIRequestBase):
        return request.model_dump(mode="json")
    return tool_metadata_adapter.validate_python(request)


@action(name="pydantic_ai_agent_load_subagent_request")
async def load_subagent_request(
    request_type: str,
    request: dict[str, JsonValue],
) -> AIRequestBase[Any]:
    """Rebuild a detached subagent's request as its concrete request class."""
    return load_request(request_type, request)


async def _send_message(agent_name: str, message: OutgoingMessage) -> None:
    writer = registered_message_writer(agent_name)
    if writer is None:
        raise RuntimeError(f"agent {agent_name!r} registers no message_writer")
    hook, parameter = writer
    await call_with_dependencies(hook, parameter, message)


@action(name="pydantic_ai_agent_report_subagent_result")
async def report_subagent_result(
    parent_agent: str,
    subagent_name: str,
    subagent_id: str,
    mailbox: str,
    output: AgentOutput,
) -> None:
    """Send a finished detached subagent's output to its mailbox."""
    value = await deserialize_payload(subagent_name, "agent_output", output)
    report = {"subagent_id": subagent_id, "status": "completed", "output": value}
    await _send_message(
        parent_agent,
        OutgoingMessage(
            mailbox=mailbox,
            content=to_json(report).decode(),
            sender=subagent_name,
            dedupe_key=subagent_id,
        ),
    )


@action(name="pydantic_ai_agent_report_subagent_failure")
async def report_subagent_failure(
    parent_agent: str,
    subagent_name: str,
    subagent_id: str,
    mailbox: str,
    error: object,
) -> Never:
    """Send a failed detached subagent's error to its mailbox, then fail its workflow."""
    message = error.get("message") if isinstance(error, dict) else None
    report = {
        "subagent_id": subagent_id,
        "status": "failed",
        "error": str(message if message is not None else error),
    }
    await _send_message(
        parent_agent,
        OutgoingMessage(
            mailbox=mailbox,
            content=to_json(report).decode(),
            sender=subagent_name,
            dedupe_key=subagent_id,
        ),
    )
    raise RuntimeError(f"detached subagent {subagent_id!r} failed: {report['error']}")


@action(name="pydantic_ai_agent_read_messages")
async def read_agent_messages(
    agent_name: str,
    wait: WaitForMessagesResult,
) -> ToolActionResult:
    """Poll a waiting tool call's mailbox once through the agent's message reader.

    Returns a ``return`` result holding the messages when any arrived, and the unchanged
    ``wait_messages`` result when the mailbox is still empty.
    """
    reader = registered_message_reader(agent_name)
    if reader is None:
        raise RuntimeError(f"agent {agent_name!r} registers no message_reader")
    hook, parameter = reader
    query = MessageQuery(
        mailbox=wait["mailbox"],
        claim_key=wait["claim_key"],
        agent_reference=agent_name,
        run_id=wait["run_id"],
        tool_call_id=wait["tool_call_id"],
        tool_name=wait["tool_name"],
    )
    messages: Sequence[AgentMessage] = agent_messages_adapter.validate_python(
        await call_with_dependencies(hook, parameter, query)
    )
    if not messages:
        return wait
    return {
        "kind": "return",
        "tool_call_id": wait["tool_call_id"],
        "tool_name": wait["tool_name"],
        "value": await serialize_payload(
            agent_name,
            ToolOutputPayload(
                value=[message.model_dump(mode="json", exclude_none=True) for message in messages]
            ),
        ),
    }


@action(name="pydantic_ai_agent_tool_retries_exhausted")
async def raise_tool_retries_exhausted(tool_name: str, max_retries: int) -> Never:
    """Fail the run the way Pydantic AI does when a tool exceeds its retry budget.

    Reached when Waymark lost the tool's action more often than the tool's own
    ``retries=`` allows, so the run ends exactly as a plain Pydantic AI run whose
    tool kept failing.
    """
    raise _tool_retries_exceeded(tool_name, max_retries)


def _tool_retries_exceeded(tool_name: str, max_retries: int) -> UnexpectedModelBehavior:
    """Pydantic AI's own error for a tool over its budget, from its own check, so the
    message matches whichever Pydantic AI version is installed."""
    from types import SimpleNamespace

    from pydantic_ai.tool_manager import ToolManager

    spent = SimpleNamespace(ctx=SimpleNamespace(retries={tool_name: max_retries}))
    try:
        ToolManager._check_max_retries(  # noqa: SLF001
            spent,  # ty: ignore[invalid-argument-type]
            tool_name,
            max_retries,
            RuntimeError("Waymark lost the tool's action"),
        )
    except UnexpectedModelBehavior as error:
        return error
    except Exception:
        pass
    return UnexpectedModelBehavior(
        f"Tool {tool_name!r} exceeded max retries count of {max_retries}"
    )


@action(name="pydantic_ai_agent_model_attempts_exhausted")
async def raise_model_attempts_exhausted(attempts: int) -> Never:
    raise RuntimeError(f"model request failed after {attempts} transient attempts")


# NOTE: No longer emitted; parallel tool segments now rely on `asyncio.gather` raising the
# first failure. Kept registered so in-flight runs compiled against it can still finish.
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
