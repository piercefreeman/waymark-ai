import asyncio
from typing import Any

from pydantic import BaseModel
from pydantic_ai import Agent
from waymark import workflow

from pydantic_ai_waymark import PydanticAIWorkflow, waymark_agent


class SupportReply(BaseModel):
    answer: str
    needs_human: bool


support_agent = waymark_agent(
    Agent(
        "openai:gpt-5.2",
        name="support_agent",
        instructions=(
            "Call lookup_support_policy, inspect_account_context, and estimate_resolution_time "
            "before answering. Then call wait_for_follow_up with seconds=45. "
            "Answer concisely and escalate when account access is required."
        ),
        output_type=SupportReply,
        defer_model_check=True,
    )
)


@support_agent.tool_plain
def lookup_support_policy(topic: str) -> str:
    """Look up the support policy relevant to the customer's question."""
    return (
        f"Policy for {topic}: explain the next step clearly; never claim an account change "
        "was made; set needs_human=true when private account access is required."
    )


@support_agent.tool_plain
def inspect_account_context(question: str) -> str:
    """Determine whether answering a question requires private account access."""
    return (
        f"The question {question!r} requires a human when it asks about a specific charge, "
        "order, refund, or account change."
    )


@support_agent.tool_plain
def estimate_resolution_time(issue_type: str) -> str:
    """Return the normal support resolution window for an issue type."""
    return f"Typical resolution window for {issue_type}: 1-3 business days after review."


@workflow
class SupportWorkflow(PydanticAIWorkflow):
    @staticmethod
    async def wait_for_follow_up(seconds: float = 45) -> str:
        """Wait durably before returning the support answer."""
        await asyncio.sleep(seconds)
        return "Follow-up wait completed."

    async def run_workflow_tool(
        self,
        agent_name: str,
        tool_name: str,
        args: dict[str, Any],
    ) -> Any:
        if tool_name == "wait_for_follow_up":
            return await self.wait_for_follow_up(args["seconds"])
        return await self._unsupported_workflow_tool(agent_name, tool_name)

    async def run(self, prompt: str, message_history: str | None = None) -> SupportReply:
        result = await self.run_agent(
            "support_agent",
            prompt,
            message_history=message_history,
        )
        return result["output"]


support_agent.tool_plain(metadata={"waymark": False})(SupportWorkflow.wait_for_follow_up)
