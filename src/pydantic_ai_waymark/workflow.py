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
    ToolActionResult,
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
        previous_transition: AgentTransition | None = None
        tool_results: list[ToolActionResult] = []
        while True:
            current_transition: AgentTransition | None = None
            while True:
                try:
                    current_transition = await self.run_action(
                        run_agent_node(
                            agent_name,
                            prompt,
                            previous_transition,
                            tool_results,
                            message_history=message_history,
                            deps=deps,
                            model=model,
                            conversation_id=conversation_id,
                            run_id=run_id,
                        )
                    )
                    previous_transition = current_transition
                    break
                except Exception:
                    await asyncio.sleep(2)
            tool_results = []
            if current_transition is None:
                continue
            if current_transition["kind"] == "done":
                return current_transition["result"]
            if current_transition["approvals"]:
                return await raise_approval_required(agent_name)
            for tool_call in current_transition["tool_calls"]:
                # This branch is selected only by metadata={"waymark": False} on the
                # Pydantic AI tool. Pydantic has already validated the arguments and
                # stored them in the durable transition. The workflow implementation
                # must dispatch the tool in run_workflow_tool using only deterministic,
                # Waymark-compileable code. asyncio.sleep() is a durable workflow timer;
                # external I/O and other side effects do not belong in this branch.
                if tool_call["waymark"] is False:
                    workflow_args = tool_call["workflow_args"]
                    if workflow_args is None:
                        return await self._unsupported_workflow_tool(
                            agent_name, tool_call["call"]["tool_name"]
                        )
                    workflow_value = await self.run_workflow_tool(
                        agent_name,
                        tool_call["call"]["tool_name"],
                        workflow_args,
                    )
                    tool_results.append(
                        {
                            "kind": "return",
                            "tool_call_id": tool_call["call"]["tool_call_id"],
                            "tool_name": tool_call["call"]["tool_name"],
                            "value": workflow_value,
                        }
                    )
                else:
                    # Every ordinary tool takes this branch. tool_call["waymark"] is
                    # the default policy or metadata={"waymark": {...}} supplied by
                    # the tool. Its Python body runs as an action, where external I/O
                    # and non-deterministic work are safe. Attempts and durable retry
                    # backoff come from that per-tool policy.
                    tool_attempt = 0
                    while True:
                        tool_attempt += 1
                        try:
                            action_result = await self.run_action(
                                run_agent_tool(
                                    agent_name,
                                    current_transition,
                                    tool_call,
                                    deps=deps,
                                    model=model,
                                )
                            )
                            tool_results.append(action_result)
                            break
                        except Exception:
                            if tool_attempt >= tool_call["waymark"]["attempts"]:
                                await raise_tool_attempts_exhausted(
                                    tool_call["call"]["tool_name"], tool_attempt
                                )
                            if tool_call["waymark"]["backoff_seconds"] > 0.0:
                                await asyncio.sleep(tool_call["waymark"]["backoff_seconds"])

    async def run_workflow_tool(
        self,
        agent_name: str,
        tool_name: str,
        args: WorkflowToolArgs,
    ) -> ToolValue:
        return await self._unsupported_workflow_tool(agent_name, tool_name)

    async def _unsupported_workflow_tool(self, agent_name: str, tool_name: str) -> Never:
        return await raise_workflow_tool_not_configured(agent_name, tool_name)
