import asyncio

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
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


class SleepDependencies(BaseModel):
    seconds: float = Field(default=5, ge=0, le=60)


support_agent = waymark_agent(
    Agent(
        "openai:gpt-5.2",
        name="support_agent",
        instructions=(
            "Call lookup_support_policy, inspect_account_context, and estimate_resolution_time "
            "before answering. Answer concisely and escalate when account access is required."
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

sleep_agent = waymark_agent(
    Agent(
        "openai:gpt-5.2",
        name="sleep_agent",
        deps_type=SleepDependencies,
        instructions="Call simulate_processing exactly once, then briefly confirm completion.",
        defer_model_check=True,
    )
)


class SupportRequest(AIRequestBase[None]):
    agent = support_agent


class ReviewRequest(AIRequestBase[None]):
    agent = review_agent


class SleepRequest(AIRequestBase[SleepDependencies]):
    agent = sleep_agent


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


@sleep_agent.tool
def simulate_processing(ctx: RunContext[SleepDependencies]) -> str:
    """Wait for the configured processing duration."""
    seconds = ctx.deps.seconds
    raise DurableSleep(seconds, result=f"Processing completed after {seconds:g} seconds.")


@workflow
class SupportWorkflow(PydanticAIWorkflow[SupportRequest]):
    async def run(self, request: SupportRequest) -> AgentResult:
        return await self.run_agent(request)


@workflow
class SleepWorkflow(PydanticAIWorkflow[SleepRequest]):
    async def run(self, request: SleepRequest) -> str:
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
