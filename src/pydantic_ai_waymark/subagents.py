"""The workflow that runs subagents started with ``DelegateAgent(..., wait=False)``.

NOTE: Detached subagents report back through mailboxes that the parent polls. That's a
workaround for Waymark having no native message queue or workflow-to-workflow signal;
once Waymark has one, results should be delivered through it instead of a polled table.
"""

from typing import Any

from pydantic import BaseModel
from waymark import workflow

from .actions import load_subagent_request, report_subagent_failure, report_subagent_result
from .request import AIRequestBase
from .types import JsonValue
from .workflow import PydanticAIWorkflow


class DetachedSubagentRequest(BaseModel):
    """Input for one detached subagent run and where it reports back."""

    #: The subagent request as JSON data, and the ``module:QualName`` of its class.
    request: dict[str, JsonValue]
    request_type: str
    parent_agent: str
    subagent_id: str
    mailbox: str


@workflow
class SubagentWorkflow(PydanticAIWorkflow[AIRequestBase[Any]]):
    """Run one detached subagent request and report its result to its mailbox."""

    async def run(self, detached: DetachedSubagentRequest) -> None:
        subagent_request = await self.run_action(
            load_subagent_request(detached.request_type, detached.request)
        )
        try:
            result = await self.run_agent(subagent_request)
        except Exception as error:
            await self.run_action(
                report_subagent_failure(
                    detached.parent_agent,
                    subagent_request.agent_reference,
                    detached.subagent_id,
                    detached.mailbox,
                    error,
                )
            )
            return
        await self.run_action(
            report_subagent_result(
                detached.parent_agent,
                subagent_request.agent_reference,
                detached.subagent_id,
                detached.mailbox,
                result.output,
            )
        )
