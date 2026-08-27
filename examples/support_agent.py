import asyncio

from pydantic import BaseModel
from pydantic_ai import Agent
from waymark import action, workflow

from pydantic_ai_waymark import (
    AgentResult,
    AIRequestBase,
    DurableSleep,
    PydanticAIWorkflow,
    waymark_agent,
)


class SupportReply(BaseModel):
    answer: str
    needs_human: bool


class ReviewReply(BaseModel):
    summary: str
    concerns: list[str]


class ParallelSupportReply(BaseModel):
    support: SupportReply
    review: ReviewReply


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

review_agent = waymark_agent(
    Agent(
        "openai:gpt-5.2",
        name="review_agent",
        instructions=(
            "Independently review the support request for safety, policy, and escalation risks."
        ),
        output_type=ReviewReply,
        defer_model_check=True,
    )
)


class SupportRequest(AIRequestBase[None]):
    agent = support_agent


class ReviewRequest(AIRequestBase[None]):
    agent = review_agent


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


@action
async def combine_parallel_support_results(
    results: tuple[object, object],
) -> ParallelSupportReply:
    for result in results:
        if isinstance(result, BaseException):
            raise result
    support_result = AgentResult.model_validate(results[0])
    review_result = AgentResult.model_validate(results[1])
    return ParallelSupportReply(
        support=SupportReply.model_validate(support_result.output),
        review=ReviewReply.model_validate(review_result.output),
    )


@workflow
class ParallelSupportWorkflow(PydanticAIWorkflow[SupportRequest | ReviewRequest]):
    async def run(
        self,
        support_request: SupportRequest,
        review_request: ReviewRequest,
    ) -> ParallelSupportReply:
        results: tuple[object, object] = await asyncio.gather(
            self.run_agent(support_request),
            self.run_agent(review_request),
            return_exceptions=True,
        )
        return await combine_parallel_support_results(results)
