import asyncio
from typing import Any, Generic, Never, TypeVar

from waymark import Workflow

from .actions import (
    raise_approval_required,
    raise_tool_attempts_exhausted,
    raise_workflow_tool_not_configured,
    resolve_parallel_tool_results,
    run_agent_node,
    run_agent_tool,
)
from .request import AIRequestBase
from .types import (
    AgentResult,
    AgentTransition,
    PendingTransition,
    ToolActionResult,
    ToolCall,
    ToolValue,
    WorkflowToolArgs,
)

AIRequestT = TypeVar("AIRequestT", bound=AIRequestBase[Any])


class PydanticAIWorkflow(Workflow, Generic[AIRequestT]):
    """Workflow base that compiles Pydantic AI graph and tool transitions to actions."""

    async def run_agent(
        self,
        request: AIRequestT,
    ) -> AgentResult:
        transition: AgentTransition | None = None
        tool_results: list[ToolActionResult] = []
        while True:
            transition = await self._next_agent_transition(
                request,
                transition,
                tool_results,
            )
            if transition.kind == "done":
                return transition.result
            await self._handle_agent_approvals(request.agent_reference, transition)
            tool_results = await self._handle_agent_tools(
                request,
                transition,
            )

    async def _next_agent_transition(
        self,
        request: AIRequestT,
        transition: AgentTransition | None,
        tool_results: list[ToolActionResult],
    ) -> AgentTransition:
        while True:
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
            except Exception:
                await asyncio.sleep(2)

    async def _handle_agent_approvals(
        self,
        agent_name: str,
        transition: PendingTransition,
    ) -> None:
        if transition.approvals:
            await raise_approval_required(agent_name)

    async def _handle_agent_tools(
        self,
        request: AIRequestT,
        transition: PendingTransition,
    ) -> list[ToolActionResult]:
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
                results.append(
                    await self._run_agent_tool_call(request, transition, tool_call)
                )
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
        if len(tool_calls) == 1:
            result = await self._run_agent_tool_call(
                request, transition, tool_calls[0]
            )
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
        if tool_call.waymark is False:
            # metadata={"waymark": False}: execute deterministic, compileable
            # workflow code, such as asyncio.sleep(), outside an action.
            if tool_call.workflow_args is None:
                return await self._unsupported_workflow_tool(
                    request.agent_reference, tool_call.call.tool_name
                )
            workflow_value = await self.run_workflow_tool(
                request.agent_reference,
                tool_call.call.tool_name,
                tool_call.workflow_args,
            )
            return {
                "kind": "return",
                "tool_call_id": tool_call.call.tool_call_id,
                "tool_name": tool_call.call.tool_name,
                "value": workflow_value,
            }

        # Default or metadata={"waymark": {...}}: execute side-effecting tool
        # code as an action, using the retry policy validated into this model.
        tool_attempt = 0
        action_result: ToolActionResult | None = None
        while True:
            tool_attempt += 1
            try:
                action_result = await self.run_action(
                    run_agent_tool(
                        request.agent_reference,
                        transition,
                        tool_call,
                        deps=request.deps,
                        model=request.model,
                    )
                )
                break
            except Exception:
                if tool_attempt >= tool_call.waymark.attempts:
                    await raise_tool_attempts_exhausted(
                        tool_call.call.tool_name, tool_attempt
                    )
                if tool_call.waymark.backoff_seconds > 0.0:
                    await asyncio.sleep(tool_call.waymark.backoff_seconds)

        if action_result["kind"] == "sleep":
            sleep_seconds = action_result["seconds"]
            await asyncio.sleep(sleep_seconds)
            return {
                "kind": "return",
                "tool_call_id": action_result["tool_call_id"],
                "tool_name": action_result["tool_name"],
                "value": action_result["value"],
            }
        return action_result

    async def run_workflow_tool(
        self,
        agent_name: str,
        tool_name: str,
        args: WorkflowToolArgs,
    ) -> ToolValue:
        return await self._unsupported_workflow_tool(agent_name, tool_name)

    async def _unsupported_workflow_tool(self, agent_name: str, tool_name: str) -> Never:
        return await raise_workflow_tool_not_configured(agent_name, tool_name)
