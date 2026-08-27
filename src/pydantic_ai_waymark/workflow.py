import asyncio
from typing import Never

from waymark import Workflow

from .actions import (
    raise_approval_required,
    raise_tool_attempts_exhausted,
    raise_workflow_tool_not_configured,
    run_agent_node,
    run_agent_tool,
)
from .types import (
    AgentDeps,
    AgentResult,
    AgentTransition,
    PendingTransition,
    ToolActionResult,
    ToolCall,
    ToolValue,
    WorkflowToolArgs,
)


class PydanticAIWorkflow(Workflow):
    """Workflow base that compiles Pydantic AI graph and tool transitions to actions."""

    async def run_agent(
        self,
        agent_name: str,
        prompt: str | None,
        message_history: str | None = None,
        deps: AgentDeps = None,
        model: str | None = None,
        conversation_id: str | None = None,
        run_id: str | None = None,
    ) -> AgentResult:
        transition: AgentTransition | None = None
        tool_results: list[ToolActionResult] = []
        while True:
            transition = await self._next_agent_transition(
                agent_name,
                prompt,
                transition,
                tool_results,
                message_history,
                deps,
                model,
                conversation_id,
                run_id,
            )
            if transition.kind == "done":
                return transition.result
            await self._handle_agent_approvals(agent_name, transition)
            tool_results = await self._handle_agent_tools(
                agent_name,
                transition,
                deps,
                model,
            )

    async def _next_agent_transition(
        self,
        agent_name: str,
        prompt: str | None,
        transition: AgentTransition | None,
        tool_results: list[ToolActionResult],
        message_history: str | None,
        deps: AgentDeps,
        model: str | None,
        conversation_id: str | None,
        run_id: str | None,
    ) -> AgentTransition:
        while True:
            try:
                return await self.run_action(
                    run_agent_node(
                        agent_name,
                        prompt,
                        transition,
                        tool_results,
                        message_history=message_history,
                        deps=deps,
                        model=model,
                        conversation_id=conversation_id,
                        run_id=run_id,
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
        agent_name: str,
        transition: PendingTransition,
        deps: AgentDeps,
        model: str | None,
    ) -> list[ToolActionResult]:
        results: list[ToolActionResult] = []
        for tool_call in transition.tool_calls:
            results.append(
                await self._run_agent_tool_call(
                    agent_name,
                    transition,
                    tool_call,
                    deps,
                    model,
                )
            )
        return results

    async def _run_agent_tool_call(
        self,
        agent_name: str,
        transition: PendingTransition,
        tool_call: ToolCall,
        deps: AgentDeps,
        model: str | None,
    ) -> ToolActionResult:
        if tool_call.waymark is False:
            # metadata={"waymark": False}: execute deterministic, compileable
            # workflow code, such as asyncio.sleep(), outside an action.
            if tool_call.workflow_args is None:
                return await self._unsupported_workflow_tool(
                    agent_name, tool_call.call.tool_name
                )
            workflow_value = await self.run_workflow_tool(
                agent_name,
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
        while True:
            tool_attempt += 1
            try:
                return await self.run_action(
                    run_agent_tool(
                        agent_name,
                        transition,
                        tool_call,
                        deps=deps,
                        model=model,
                    )
                )
            except Exception:
                if tool_attempt >= tool_call.waymark.attempts:
                    await raise_tool_attempts_exhausted(
                        tool_call.call.tool_name, tool_attempt
                    )
                if tool_call.waymark.backoff_seconds > 0.0:
                    await asyncio.sleep(tool_call.waymark.backoff_seconds)

    async def run_workflow_tool(
        self,
        agent_name: str,
        tool_name: str,
        args: WorkflowToolArgs,
    ) -> ToolValue:
        return await self._unsupported_workflow_tool(agent_name, tool_name)

    async def _unsupported_workflow_tool(self, agent_name: str, tool_name: str) -> Never:
        return await raise_workflow_tool_not_configured(agent_name, tool_name)
