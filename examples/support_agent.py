from pydantic import BaseModel
from pydantic_ai import Agent
from waymark import workflow

from pydantic_ai_waymark import (
    AIRequestBase,
    DurableSleep,
    PydanticAIWorkflow,
    waymark_agent,
)


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


class SupportRequest(AIRequestBase[None]):
    agent = support_agent


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


@support_agent.tool_plain
def wait_for_follow_up(seconds: float = 45) -> str:
    """Wait durably before returning the support answer."""
    raise DurableSleep(seconds, result="Follow-up wait completed.")


@workflow
class SupportWorkflow(PydanticAIWorkflow[SupportRequest]):
    async def run(self, request: SupportRequest) -> SupportReply:
        return (await self.run_agent(request)).output
