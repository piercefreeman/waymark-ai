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
from pydantic_ai_waymark import waymark_agent


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

Declare a deterministic workflow tool once on the workflow and decorate it
directly with the agent. `run_workflow_tool` is the compiler-visible dispatch
table; calls to `asyncio.sleep()` become durable Waymark timers:

```python
@workflow
class SupportWorkflow(PydanticAIWorkflow):
    @staticmethod
    @support_agent.tool_plain(metadata={"waymark": False})
    async def wait_for_follow_up(seconds: float = 45) -> str:
        await asyncio.sleep(seconds)
        return "Follow-up wait completed."

    async def run_workflow_tool(self, agent_name, tool_name, args):
        if tool_name == "wait_for_follow_up":
            return await self.wait_for_follow_up(args["seconds"])
        return await self._unsupported_workflow_tool(agent_name, tool_name)
```

The explicit dispatch is required because Waymark statically compiles workflow
calls; it cannot dynamically invoke a Python callable selected by the model.

Define the agent and workflow at module scope so every Waymark worker can
import the same registration by its stable name.

## Compile the agent into a workflow

Inherit from `PydanticAIWorkflow` and call `run_agent` from compiled workflow
code:

```python
from waymark import workflow
from pydantic_ai_waymark import PydanticAIWorkflow


@workflow
class SupportWorkflow(PydanticAIWorkflow):
    async def run(
        self,
        prompt: str,
        message_history: str | None = None,
    ) -> Reply:
        result = await self.run_agent(
            "support_agent",
            prompt,
            message_history=message_history,
        )
        return result["output"]
```

The result also contains `message_history`, `new_messages`, `usage`, `run_id`,
and `conversation_id`. Pass `message_history` to a later workflow invocation to
continue the conversation.

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

Tools marked with `metadata={"waymark": False}` dispatch through
`run_workflow_tool` instead of `pydantic_ai_agent_tool`. Waymark compiles that
method, including `await asyncio.sleep(...)`, so no worker is occupied and a
restart during the wait resumes at the timer deadline.

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

The example agent calls three action tools and one workflow-level sleep tool. A
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
