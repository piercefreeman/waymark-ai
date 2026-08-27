import asyncio
from typing import Any, Generic, TypeVar

from waymark import Workflow

from .actions import (
    RetryableAgentError,
    raise_approval_required,
    raise_model_attempts_exhausted,
    resolve_parallel_tool_results,
    run_agent_node,
    run_agent_tool,
)
from .request import AIRequestBase
from .types import (
    AgentResult,
    AgentTransition,
    BackoffConfig,
    JsonValue,
    PendingTransition,
    ToolActionResult,
    ToolCall,
)

AIRequestT = TypeVar("AIRequestT", bound=AIRequestBase[Any])


class PydanticAIWorkflow(Workflow, Generic[AIRequestT]):
    """Workflow base that compiles Pydantic AI graph and tool transitions to actions."""

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
        transition: AgentTransition | None = None
        tool_results: list[ToolActionResult] = []
        while True:
            transition = await self._next_agent_transition(
                request,
                transition,
                tool_results,
            )
            if transition.kind == "done":
                await self.on_agent_end(request, transition.result)
                return transition.result
            for message in transition.messages:
                await self.on_message(request, message)
            await self._handle_agent_approvals(request.agent_reference, transition)
            tool_results = await self._handle_agent_tools(
                request,
                transition,
            )

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
        transition: AgentTransition | None,
        tool_results: list[ToolActionResult],
    ) -> AgentTransition:
        """Run the next replay-safe Pydantic AI graph node as a Waymark action.

        ``run_agent`` calls this initially and after every tool batch, passing the
        prior transition and results so the graph can return its next node or result.
        Only ``RetryableAgentError`` uses the request's bounded retry configuration.
        """
        model_attempt = 0
        while True:
            model_attempt += 1
            try:
                return await self.run_action(
                    run_agent_node(
                        request.agent_reference,
                        request.prompt,
                        transition,
                        tool_results,
                        message_history=request.message_history,
                        deps=request.deps,
                        model=request.model,
                        conversation_id=request.conversation_id,
                        run_id=request.run_id,
                    )
                )
            except RetryableAgentError:
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
        transition: PendingTransition,
    ) -> None:
        """Reject a pending transition that contains approval-required tool calls.

        ``run_agent`` calls this before dispatching every non-final transition;
        transitions without approval requests pass through unchanged.
        """
        if transition.approvals:
            await raise_approval_required(agent_name)

    async def _handle_agent_tools(
        self,
        request: AIRequestT,
        transition: PendingTransition,
    ) -> list[ToolActionResult]:
        """Execute one transition's tool calls in their required barrier order.

        ``run_agent`` calls this after approval handling; ordinary calls run in
        parallel segments while calls marked sequential run alone between segments.
        """
        results: list[ToolActionResult] = []
        parallel_calls: list[ToolCall] = []
        for tool_call in transition.tool_calls:
            if tool_call.sequential:
                if parallel_calls:
                    parallel_results = await self._run_agent_tool_segment(
                        request, transition, parallel_calls
                    )
                    for parallel_result in parallel_results:
                        results.append(parallel_result)
                    parallel_calls = []
                results.append(await self._run_agent_tool_call(request, transition, tool_call))
            else:
                parallel_calls.append(tool_call)
        if parallel_calls:
            parallel_results = await self._run_agent_tool_segment(
                request, transition, parallel_calls
            )
            for parallel_result in parallel_results:
                results.append(parallel_result)
        return results

    async def _run_agent_tool_segment(
        self,
        request: AIRequestT,
        transition: PendingTransition,
        tool_calls: list[ToolCall],
    ) -> list[ToolActionResult]:
        """Execute one barrier-delimited group of parallelizable tool calls.

        ``_handle_agent_tools`` calls this before each sequential call and after
        the final call; singleton groups run directly and larger groups fan out.
        """
        if len(tool_calls) == 1:
            result = await self._run_agent_tool_call(request, transition, tool_calls[0])
            return [result]
        gathered: list[object] = await asyncio.gather(
            *[
                self._run_agent_tool_call(request, transition, tool_call)
                for tool_call in tool_calls
            ],
            return_exceptions=True,
        )
        results = await resolve_parallel_tool_results(gathered)
        return results

    async def _run_agent_tool_call(
        self,
        request: AIRequestT,
        transition: PendingTransition,
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
        action_result = await self.run_action(
            run_agent_tool(
                request.agent_reference,
                transition,
                tool_call,
                deps=request.deps,
                model=request.model,
            )
        )

        if action_result["kind"] == "sleep":
            sleep_seconds = action_result["seconds"]
            await asyncio.sleep(sleep_seconds)
            result: ToolActionResult = {
                "kind": "return",
                "tool_call_id": action_result["tool_call_id"],
                "tool_name": action_result["tool_name"],
                "value": action_result["value"],
            }
        else:
            result = action_result
        await self.on_tool_end(request, tool_call.call.tool_call_id, result)
        return result
