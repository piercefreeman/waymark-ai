# pydantic-ai-waymark

Run a Pydantic AI agent as a compiled, durable
[Waymark](https://github.com/piercefreeman/waymark) state machine.

Build agents that can run reliably for days, weeks, or years. The compilation
layer turns Pydantic AI control flow into an efficient Waymark state machine, so
an agent can persist its progress, sleep without occupying a worker, and wake at
the right time to continue from its last completed step.

## Install

The project and lockfile are controlled by [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

The project uses the latest Waymark `0.30` development release.

Install the provider used by the example:

```bash
uv sync --extra openai
```

## Define an agent

This library wraps your existing Pydantic AI agents, so define agents and tools
as you normally would. The only change is to wrap the completed `Agent(...)`
initialization in `waymark_agent(...)`:

```python
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai_waymark import AIRequestBase, waymark_agent


class Reply(BaseModel):
    answer: str
    needs_human: bool


support_agent = waymark_agent(
    Agent(
        "openai:gpt-5.2",
        name="support_agent",
        instructions="Answer concisely.",
        output_type=Reply,
        defer_model_check=True,
    )
)


@support_agent.tool_plain
def lookup_policy(topic: str) -> str:
    """Look up the support policy for a topic."""
    return f"Policy for {topic}: escalate account changes."
```

## Compile the agent into a workflow

Parameterize `PydanticAIWorkflow` with the request type, implement the Waymark
entrypoint, and call `run_agent` from it:

```python
from waymark import workflow
from pydantic_ai_waymark import AIRequestBase, PydanticAIWorkflow


class SupportRequest(AIRequestBase[None]):
    agent = support_agent


@workflow
class SupportWorkflow(PydanticAIWorkflow[SupportRequest]):
    async def run(self, request: SupportRequest) -> Reply:
        return (await self.run_agent(request)).output


reply = await SupportWorkflow().run(
    SupportRequest(prompt="How do I update my account?")
)
```

The request parameter may be a union such as
`PydanticAIWorkflow[SupportRequest | SalesRequest]`. This simply acts as a typehint
for the run_agent function. You can similarly nest these values within a large request blob:

```python
class Agent1Request(APIRequestBase[None]):
    agent = agent_1

class Agent2Request(APIRequestBase[None]):
    agent = agent_2

class MainRequest(BaseModel):
    request_1: Agent1Request
    request_2: Agent2Request

@workflow
class MultiAgentWorkflow(PydanticAIWorkflow[Agent1Request | Agent2Request]):
    async def run(self, request: MainRequest) -> None:
        response_1 = await self.run_agent(request.request_1)
        response_2 = await self.run_agent(request.request_2)

reply = await MultiAgentWorkflow().run(
    MainRequest(
        request_1=SupportRequest(prompt="What's your name?")
        request_2=SupportRequest(prompt="What's your name?")
    )
)
```

`AIRequestBase` also accepts `message_history`, `deps`, `model`,
`conversation_id`, and `run_id`. Its serialized representation includes the stable
agent reference needed by the worker.

## Extras

We make our best effort to wrap `pydantic-ai`'s features 1:1 - just with the addition of the magic of durable execution. For instance, you can use Pydantic AI's existing retry and timeout settings as usual:

```python
@support_agent.tool_plain(
    retries=3,
    timeout=120,
)
def lookup_policy(topic: str) -> str:
    return f"Policy for {topic}: escalate account changes."
```

Timeouts and `ModelRetry` responses follow Pydantic AI's retry flow across
durable Waymark actions. Other exceptions fail the workflow immediately.

Tools run in parallel by default. Mark a tool as sequential when it must run
alone:

```python
@support_agent.tool_plain
async def read_profile() -> str:
    return "Profile loaded."


@support_agent.tool_plain(sequential=True)
async def update_account() -> str:
    return "Account updated."


@support_agent.tool_plain
async def send_confirmation() -> str:
    return "Confirmation sent."
```

`sequential=True` acts as a barrier. If the model calls `read_profile`,
`lookup_policy`, `update_account`, and `send_confirmation` in that order,
Waymark resolves them as follows:

1. `read_profile` and `lookup_policy` run in parallel.
2. `update_account` runs alone after both finish.
3. `send_confirmation` starts after `update_account` finishes.

Calls after the barrier can run in parallel again until the next sequential
tool call.

To let a tool request a durable wait, raise `DurableSleep`. The tool action
records the request, the workflow performs the timer, and the supplied result
is returned to the model under the original tool-call ID:

```python
from pydantic_ai_waymark import DurableSleep


@support_agent.tool_plain
def wait_for_follow_up(seconds: float = 45) -> str:
    raise DurableSleep(seconds, result="Follow-up wait completed.")
```

## Docker Compose example

The example includes Postgres, Waymark workers, the Waymark dashboard, and a
small FastAPI form. Put the OpenAI key in the repository's `.env` file and run:

```bash
cp .env.example .env
docker compose -f examples/docker-compose.yml up --build
```

Open [http://localhost:8000](http://localhost:8000). The Waymark dashboard is at
[http://localhost:24119](http://localhost:24119).

The example agent calls three action tools and one `DurableSleep` tool. A
request visibly exercises model actions, separate tool actions, a 45-second
durable timer, and the resumed model action.

Stop the stack and remove its example database with:

```bash
docker compose -f examples/docker-compose.yml down -v
```

## Workers

The module containing registered agents must be importable by each worker:

```bash
export WAYMARK_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/waymark
export WAYMARK_USER_MODULE=examples.support_agent
export OPENAI_API_KEY=...
uv run waymark-start-workers
```

## Checks

```bash
uv run pytest -q
uv run ruff check .
uv run ty check
```
