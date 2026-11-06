import asyncio
import types
from typing import Any, Generic, TypeVar, get_args, get_origin

from waymark import ActionExecutionLost, ActionExecutionNotStarted, Workflow

from .actions import (
    RetryableAgentError,
    delegate_tool_result,
    raise_approval_required,
    raise_model_attempts_exhausted,
    raise_tool_retries_exhausted,
    read_agent_messages,
    run_agent_node,
    run_agent_tool,
    start_detached_subagent,
)
from .request import AIRequestBase
from .types import (
    AgentCheckpoint,
    AgentResult,
    AgentTransition,
    BackoffConfig,
    DelegateAgentResult,
    JsonValue,
    SerializedToolCall,
    ToolActionResult,
    ToolCall,
    WaitForMessagesResult,
)

AIRequestT = TypeVar("AIRequestT", bound=AIRequestBase[Any])


class PydanticAIWorkflow(Workflow, Generic[AIRequestT]):
    """Workflow base that compiles Pydantic AI graph and tool transitions to actions."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _specialize_request_helpers(cls)

    # Public

    async def run_agent(
        self,
        request: AIRequestT,
    ) -> AgentResult:
        """Drive one agent request through graph transitions until it returns a result.

        Concrete workflow entrypoints call this once per request; each iteration runs
        the next graph node, stops on a final result, or dispatches its pending tools.
        """
        await self.on_agent_start(request)
        tool_results: list[ToolActionResult] = []
        message_history: list[str] = []
        if request.message_history is not None:
            message_history = [request.message_history]
        checkpoint = AgentCheckpoint(message_history=message_history)
        transition: AgentTransition | None = None
        while True:
            transition = await self._next_agent_transition(
                request,
                checkpoint,
                tool_results,
            )
            if transition.kind == "done":
                result = transition.result
                transition = None
                checkpoint = AgentCheckpoint()
                tool_results = []
                await self.on_agent_end(request, result)
                return result
            message_history = checkpoint.message_history
            if transition.history_delta is not None:
                # NOTE: Waymark only appends to fresh local lists, and this one aliases the
                # previous checkpoint's history, so build the extended list instead.
                message_history = message_history + [transition.history_delta]
            messages = transition.messages
            approvals = transition.approvals
            tool_calls = transition.tool_calls
            checkpoint = AgentCheckpoint(
                state=transition.state,
                node=transition.node,
                deps_state=transition.deps_state,
                tool_metadata=transition.tool_metadata,
                message_history=message_history,
            )
            transition = None
            message_history = []
            for message in messages:
                await self.on_message(request, message)
            await self._handle_agent_approvals(request.agent_reference, approvals)
            messages = []
            approvals = []
            tool_results = await self._handle_agent_tools(
                request,
                checkpoint,
                tool_calls,
            )
            tool_calls = []

    # Overrides

    async def on_agent_start(self, agent_request: AIRequestT) -> None:
        """Run when an agent request starts."""

    async def on_agent_end(self, agent_request: AIRequestT, payload: AgentResult) -> None:
        """Run when an agent request completes."""

    async def on_message(self, agent_request: AIRequestT, message: str) -> None:
        """Run when the model returns a plain text message."""

    async def on_tool_start(
        self,
        agent_request: AIRequestT,
        tool_id: str,
        tool_args: str | dict[str, JsonValue] | None,
    ) -> None:
        """Run before a tool action starts."""

    async def on_tool_end(
        self,
        agent_request: AIRequestT,
        tool_id: str,
        payload: ToolActionResult,
    ) -> None:
        """Run after a tool action completes."""

    # Helper functions

    async def _next_agent_transition(
        self,
        request: AIRequestT,
        checkpoint: AgentCheckpoint,
        tool_results: list[ToolActionResult],
    ) -> AgentTransition:
        """Run the next replay-safe Pydantic AI graph node as a Waymark action.

        ``run_agent`` calls this initially and after every tool batch, passing the
        prior transition and results so the graph can return its next node or result.
        ``RetryableAgentError`` and a lost or never-started action (a worker restart)
        use the request's bounded retry configuration: the node is replay-safe, so
        rerunning it is invisible to the agent.
        """
        model_attempt = 0
        while True:
            model_attempt += 1
            try:
                return await self.run_action(
                    run_agent_node(
                        request.agent_reference,
                        request.prompt,
                        checkpoint,
                        tool_results,
                        deps=request.deps,
                        model=request.model,
                        conversation_id=request.conversation_id,
                        run_id=request.run_id,
                    )
                )
            except (RetryableAgentError, ActionExecutionLost, ActionExecutionNotStarted):
                if model_attempt >= request.model_retry.attempts:
                    await raise_model_attempts_exhausted(model_attempt)
                backoff_seconds = await self._model_retry_backoff_seconds(
                    request.model_retry,
                    model_attempt,
                )
                if backoff_seconds > 0.0:
                    await asyncio.sleep(backoff_seconds)

    async def _model_retry_backoff_seconds(
        self,
        config: BackoffConfig,
        failed_attempt: int,
    ) -> float:
        """Calculate the next bounded exponential delay after a transient failure.

        ``_next_agent_transition`` calls this only between retryable model attempts;
        the first failure uses ``initial_seconds`` and later failures multiply it.
        """
        delay = config.initial_seconds
        backoff_step = 1
        while backoff_step < failed_attempt:
            delay = delay * config.multiplier
            backoff_step += 1
        if delay > config.max_seconds:
            return config.max_seconds
        return delay

    async def _handle_agent_approvals(
        self,
        agent_name: str,
        approvals: list[SerializedToolCall],
    ) -> None:
        """Reject a pending transition that contains approval-required tool calls.

        ``run_agent`` calls this before dispatching every non-final transition;
        transitions without approval requests pass through unchanged.
        """
        if approvals:
            await raise_approval_required(agent_name)

    async def _handle_agent_tools(
        self,
        request: AIRequestT,
        checkpoint: AgentCheckpoint,
        tool_calls: list[ToolCall],
    ) -> list[ToolActionResult]:
        """Execute one transition's tool calls in their required barrier order.

        ``run_agent`` calls this after approval handling; ordinary calls run in
        parallel segments while calls marked sequential run alone between segments.
        """
        # NOTE: Waymark only appends to lists never passed to a call, and each batch is
        # passed to a segment, so the lists here grow by concatenation instead.
        # NOTE: Waymark only appends to lists never passed to a call, and each batch is
        # passed to a segment, so the lists here grow by concatenation instead.
        results: list[ToolActionResult] = []
        parallel_calls: list[ToolCall] = []
        for tool_call in tool_calls:
            if tool_call.sequential:
                if parallel_calls:
                    parallel_results = await self._run_agent_tool_segment(
                        request,
                        checkpoint,
                        parallel_calls,
                    )
                    results = results + parallel_results
                    parallel_calls = []
                sequential_result = await self._run_agent_tool_call(
                    request,
                    checkpoint,
                    tool_call,
                )
                results = results + [sequential_result]
            else:
                parallel_calls = parallel_calls + [tool_call]
        if parallel_calls:
            parallel_results = await self._run_agent_tool_segment(
                request,
                checkpoint,
                parallel_calls,
            )
            results = results + parallel_results
        return results

    async def _run_agent_tool_segment(
        self,
        request: AIRequestT,
        checkpoint: AgentCheckpoint,
        tool_calls: list[ToolCall],
    ) -> list[ToolActionResult]:
        """Execute one barrier-delimited group of parallelizable tool calls.

        ``_handle_agent_tools`` calls this before each sequential call and after
        the final call; singleton groups run directly and larger groups fan out.
        """
        if len(tool_calls) == 1:
            result = await self._run_agent_tool_call(request, checkpoint, tool_calls[0])
            return [result]
        results: list[ToolActionResult] = await asyncio.gather(
            *[self._run_agent_tool_call(request, checkpoint, tool_call) for tool_call in tool_calls]
        )
        return results

    async def _run_agent_tool_call(
        self,
        request: AIRequestT,
        checkpoint: AgentCheckpoint,
        tool_call: ToolCall,
    ) -> ToolActionResult:
        """Dispatch one tool call through its configured durability boundary.

        Tool handling calls this directly for sequential calls and through a segment
        otherwise; each call runs as a separately persisted Waymark action while
        Pydantic AI applies the tool's native retry and timeout settings.
        """
        await self.on_tool_start(
            request,
            tool_call.call.tool_call_id,
            tool_call.call.args,
        )
        action_result = await self._run_agent_tool_action(request, checkpoint, tool_call)

        if action_result["kind"] == "sleep":
            sleep_seconds = action_result["seconds"]
            await asyncio.sleep(sleep_seconds)
            result: ToolActionResult = {
                "kind": "return",
                "tool_call_id": action_result["tool_call_id"],
                "tool_name": action_result["tool_name"],
                "value": action_result["value"],
            }
        elif action_result["kind"] == "delegate":
            result = await self._run_subagent(request, action_result)
        elif action_result["kind"] == "wait_messages":
            result = await self._wait_for_agent_messages(request, action_result)
        else:
            result = action_result
        await self.on_tool_end(request, tool_call.call.tool_call_id, result)
        return result

    async def _run_agent_tool_action(
        self,
        request: AIRequestT,
        checkpoint: AgentCheckpoint,
        tool_call: ToolCall,
    ) -> ToolActionResult:
        """Run one tool call's action, rerunning it when Waymark loses it.

        A plain Pydantic AI run has no notion of a worker dying, so the agent never
        sees one: a lost or never-started attempt reruns the same call, with no retry
        prompt. The tool's own retry budget (``retries=``, carried on ``tool_call``)
        bounds it, counted apart from Pydantic AI's own ``ModelRetry`` attempts; past
        it the run fails as Pydantic AI fails a tool that exceeds its retries. A lost
        attempt may have finished, so a tool with side effects that must not repeat
        sets ``retries=0``.
        """
        lost_attempts = 0
        while True:
            try:
                return await self.run_action(
                    run_agent_tool(
                        request.agent_reference,
                        checkpoint,
                        tool_call,
                        deps=request.deps,
                        model=request.model,
                        prompt=request.prompt,
                    )
                )
            except (ActionExecutionLost, ActionExecutionNotStarted):
                lost_attempts += 1
                if lost_attempts > tool_call.max_retries:
                    await raise_tool_retries_exhausted(
                        tool_call.call.tool_name, tool_call.max_retries
                    )

    async def _run_subagent(
        self,
        request: AIRequestT,
        delegation: DelegateAgentResult,
    ) -> ToolActionResult:
        """Run or start a tool's delegated subagent and return the tool call's result.

        ``_run_agent_tool_call`` calls this when a tool raised ``DelegateAgent``. A waiting
        delegation runs the subagent through ``run_agent`` as nested durable work with its
        own hooks; a detached one queues it as a separate workflow and returns its handle.
        """
        if not delegation["wait"]:
            return await self._start_detached_subagent(request, delegation)
        # NOTE: `DelegateAgentResult` types the request as `BaseModel` to avoid an import
        # cycle; it is always the `AIRequestBase` subclass the tool raised with.
        subagent_request = delegation["request"]
        subagent_result = await self.run_agent(subagent_request)  # ty: ignore[invalid-argument-type]
        return await self._delegate_tool_result(
            request, delegation, subagent_request, subagent_result
        )

    async def _start_detached_subagent(
        self,
        request: AIRequestT,
        delegation: DelegateAgentResult,
    ) -> ToolActionResult:
        """Queue a detached subagent. Only a start that provably never ran is retried:
        a lost one may already have queued the subagent."""
        attempt = 0
        while True:
            attempt += 1
            try:
                return await self.run_action(
                    start_detached_subagent(request.agent_reference, delegation)
                )
            except ActionExecutionNotStarted:
                if attempt >= request.model_retry.attempts:
                    await raise_model_attempts_exhausted(attempt)

    async def _delegate_tool_result(
        self,
        request: AIRequestT,
        delegation: DelegateAgentResult,
        subagent_request: Any,
        subagent_result: AgentResult,
    ) -> ToolActionResult:
        """Turn a subagent's output into the delegating tool's result; replay-safe,
        so a lost attempt reruns."""
        attempt = 0
        while True:
            attempt += 1
            try:
                return await self.run_action(
                    delegate_tool_result(
                        request.agent_reference,
                        subagent_request.agent_reference,
                        delegation["tool_call_id"],
                        delegation["tool_name"],
                        subagent_result.output,
                    )
                )
            except (ActionExecutionLost, ActionExecutionNotStarted):
                if attempt >= request.model_retry.attempts:
                    await raise_model_attempts_exhausted(attempt)

    async def _wait_for_agent_messages(
        self,
        request: AIRequestT,
        wait: WaitForMessagesResult,
    ) -> ToolActionResult:
        """Poll a mailbox through the agent's message reader until messages or timeout.

        ``_run_agent_tool_call`` calls this when a tool raised ``WaitForMessages``; each
        poll is its own action and the interval between polls is a durable sleep.
        """
        # NOTE: A workaround until Waymark has native message queues; with them this loop
        # becomes a single durable wait on the mailbox.
        waited_seconds = 0.0
        while True:
            poll = await self._read_agent_messages(request, wait)
            if poll["kind"] == "return":
                return poll
            timeout_seconds = wait["timeout_seconds"]
            if timeout_seconds is not None:
                if waited_seconds >= timeout_seconds:
                    timed_out: ToolActionResult = {
                        "kind": "return",
                        "tool_call_id": wait["tool_call_id"],
                        "tool_name": wait["tool_name"],
                        "value": wait["timeout_value"],
                    }
                    return timed_out
            poll_seconds = wait["poll_seconds"]
            await asyncio.sleep(poll_seconds)
            waited_seconds = waited_seconds + poll_seconds

    async def _read_agent_messages(
        self,
        request: AIRequestT,
        wait: WaitForMessagesResult,
    ) -> ToolActionResult:
        """One mailbox poll. Claims are keyed by ``claim_key``, so a lost poll reruns."""
        attempt = 0
        while True:
            attempt += 1
            try:
                return await self.run_action(read_agent_messages(request.agent_reference, wait))
            except (ActionExecutionLost, ActionExecutionNotStarted):
                if attempt >= request.model_retry.attempts:
                    await raise_model_attempts_exhausted(attempt)


def _specialized_request_type(workflow_cls: type) -> object | None:
    """Return the request type a subclass fills ``AIRequestT`` with, if it is concrete."""
    for base in types.get_original_bases(workflow_cls):
        origin = get_origin(base)
        if isinstance(origin, type) and issubclass(origin, PydanticAIWorkflow):
            arguments = get_args(base)
            if len(arguments) != 1:
                return None
            request_type = arguments[0]
            if isinstance(request_type, TypeVar) or any(
                isinstance(argument, TypeVar) for argument in get_args(request_type)
            ):
                return None
            return request_type
    return None


def _specialize_request_helpers(workflow_cls: type) -> None:
    """Give a specialized subclass its own copies of the request-typed base helpers.

    Waymark compiles each ``self.<helper>()`` it reaches from the source of the method as
    written, resolving annotations in that function's globals. On this base class those
    still name the open ``AIRequestT``, which Waymark rejects; it does not apply the
    subclass's ``PydanticAIWorkflow[SomeRequest]`` specialization. Each copy shares the
    original code but resolves ``AIRequestT`` to the subclass's request type, so the
    compiled workflow is closed while type checkers keep the generic signatures.
    """
    request_type = _specialized_request_type(workflow_cls)
    if request_type is None:
        return
    for name, helper in vars(PydanticAIWorkflow).items():
        if not isinstance(helper, types.FunctionType) or getattr(workflow_cls, name) is not helper:
            continue
        annotations = helper.__annotations__
        if not any(annotation is AIRequestT for annotation in annotations.values()):
            continue
        specialized = types.FunctionType(
            helper.__code__,
            {**helper.__globals__, "AIRequestT": request_type},
            helper.__name__,
            helper.__defaults__,
            helper.__closure__,
        )
        specialized.__kwdefaults__ = helper.__kwdefaults__
        specialized.__qualname__ = f"{workflow_cls.__qualname__}.{name}"
        specialized.__module__ = helper.__module__
        specialized.__doc__ = helper.__doc__
        specialized.__annotations__ = {
            key: request_type if annotation is AIRequestT else annotation
            for key, annotation in annotations.items()
        }
        setattr(workflow_cls, name, specialized)
