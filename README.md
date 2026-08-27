# pydantic-ai-waymark

Run a Pydantic AI agent as a compiled, durable
[Waymark](https://github.com/piercefreeman/waymark) state machine.

This is not an `agent.run()` wrapper. Each Pydantic AI graph transition is a
Waymark action, and every user tool call is another Waymark action. A completed
model request or tool call is therefore replayed from Waymark's persisted result
instead of being executed again after a worker restart.

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

Define and register a normal module-level Pydantic AI agent. Its tools remain
normal Pydantic AI tools:

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


class SupportRequest(AIRequestBase[None]):
    agent = support_agent


@support_agent.tool_plain
def lookup_policy(topic: str) -> str:
    """Look up the support policy for a topic."""
    return f"Policy for {topic}: escalate account changes."
```

Set Waymark's action policy on the Pydantic tool definition:

```python
@support_agent.tool_plain(
    metadata={"waymark": {"attempts": 3, "backoff_seconds": 2, "timeout": 120}}
)
def lookup_policy(topic: str) -> str:
    return f"Policy for {topic}: escalate account changes."
```

These values control Waymark action attempts. Pydantic AI's `retries=` and
`timeout=` arguments remain model-facing tool retry settings.

Tools run in parallel by default. Pydantic AI's `sequential=True` flag is a
barrier: earlier tools finish first, the sequential tool runs alone, and later
tools start only after it finishes.

To let a tool request a durable wait, raise `DurableSleep`. The tool action
records the request, the workflow performs the timer, and the supplied result
is returned to the model under the original tool-call ID:

```python
from pydantic_ai_waymark import DurableSleep


@support_agent.tool_plain
def wait_for_follow_up(seconds: float = 45) -> str:
    raise DurableSleep(seconds, result="Follow-up wait completed.")
```

The request serializes the agent as its module and variable name, not as a live
Python object. Define the agent, request, and workflow at module scope so every
Waymark worker can import the same agent reference.

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
`PydanticAIWorkflow[SupportRequest | SalesRequest]`. Each request's serialized
agent reference selects the matching class-level `agent` when Waymark rebuilds
the workflow input.

`AIRequestBase` also accepts `message_history`, `deps`, `model`,
`conversation_id`, and `run_id`. Its `to_json()` representation includes the
stable agent reference needed by the worker.

## Durable boundaries

Pydantic AI's Temporal and DBOS integrations use capability hooks and model /
toolset wrappers to put provider requests and tool executions into their
runtime's durable units. Waymark cannot use that implementation literally:
Waymark executes compiled workflow IR rather than replaying the Python body.

This adapter exposes the same work to Waymark's compiler:

| Pydantic AI progression | Waymark boundary |
| --- | --- |
| `UserPromptNode` | `pydantic_ai_agent_node` action |
| `ModelRequestNode` | `pydantic_ai_agent_node` action |
| `CallToolsNode` validation/control | `pydantic_ai_agent_node` action |
| Each validated user tool | `pydantic_ai_agent_tool` action |
| Applying recorded tool results | resumed `pydantic_ai_agent_node` action |

The adapter uses a short-lived Pydantic AI capability to defer validated tools
before their Python bodies execute. The compiled workflow schedules each tool,
then resumes the same `CallToolsNode` with `DeferredToolResults`. Graph state,
retry state, usage, messages, run ID, and conversation ID cross every action
boundary.

Graph-node checkpoints retry indefinitely with durable backoff. They have no
adapter-level timeout; model request timeouts come from Pydantic AI's
`ModelSettings` or the provider client. Tool actions remain bounded by their
`metadata["waymark"]` policy because they may have side effects.

`DurableSleep` subclasses Pydantic AI's `CallDeferred`. The tool body raises it
inside `pydantic_ai_agent_tool`; the action returns a typed sleep request rather
than failing. Waymark then executes `asyncio.sleep(...)` at workflow level, so
no worker is occupied and a restart resumes at the timer deadline. Afterward,
the requested value is applied through `DeferredToolResults` like a normal tool
return.

Other deterministic workflow-native tools can still use
`metadata={"waymark": False}` and a compiled `run_workflow_tool` dispatcher.

The graph-state wire format currently relies on Pydantic AI's private graph
types, so the dependency is deliberately pinned to the `2.21.x` line. Upgrade
Pydantic AI only with the adapter tests passing. Human-approval and already
external/deferred tools currently stop with a clear error instead of being
auto-approved.

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
