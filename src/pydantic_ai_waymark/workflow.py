import asyncio
from types import FunctionType
from typing import Any, Generic, Never, TypeVar, cast, get_args, get_origin

from waymark import Workflow

from .actions import (
    raise_approval_required,
    raise_tool_attempts_exhausted,
    raise_workflow_tool_not_configured,
    run_agent_node,
    run_agent_tool,
)
from .request import AIRequestBase
from .types import (
    AgentOutput,
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

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "run" in cls.__dict__:
            return
        request_type = next(
            (
                get_args(base)[0]
                for base in getattr(cls, "__orig_bases__", ())
                if get_origin(base) is PydanticAIWorkflow
            ),
            None,
        )
        output_type = getattr(getattr(request_type, "agent", None), "output_type", None)
        if not isinstance(request_type, type) or not isinstance(output_type, type):
            return

        # Waymark uses run()'s concrete return annotation to rebuild Pydantic outputs.
        run = FunctionType(
            PydanticAIWorkflow.run.__code__,
            PydanticAIWorkflow.run.__globals__,
            name="run",
        )
        run.__annotations__ = {"request": request_type, "return": output_type}
        run.__qualname__ = f"{cls.__qualname__}.run"
        cls.run = cast(Any, run)

    async def run(self, request: AIRequestT) -> AgentOutput:
        result = await self.run_agent(request)
        return result.output

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
        for tool_call in transition.tool_calls:
            results.append(
                await self._run_agent_tool_call(
                    request,
                    transition,
                    tool_call,
                )
            )
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
        while True:
            tool_attempt += 1
            try:
                return await self.run_action(
                    run_agent_tool(
                        request.agent_reference,
                        transition,
                        tool_call,
                        deps=request.deps,
                        model=request.model,
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
