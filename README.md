# pydantic-ai-waymark

`pydantic-ai-waymark` runs Pydantic AI agents as durable
[Waymark](https://github.com/piercefreeman/waymark) workflows. Waymark saves progress
at model and tool action boundaries so a run can resume after a worker restarts.
Durable timers let an agent wait without keeping a worker occupied.

You define agents and tools with Pydantic AI; Waymark handles persistence and
scheduling through PostgreSQL and a pool of workers.

## Install

Requires Python 3.12 or later.

```bash
uv add pydantic-ai-waymark
```

For the OpenAI examples below, include the provider extra:

```bash
uv add "pydantic-ai-waymark[openai]"
```

## Define an agent

Register the agent with `waymark_agent` at module scope, then add its tools.
Save these definitions in an importable module, such as `support_workflow.py`:

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

## Run the agent in a workflow

Add a request type and a workflow to the same module. `run_agent` returns an
`AgentResult`; its `output` contains the agent's answer.

```python
from waymark import workflow
from pydantic_ai_waymark import AIRequestBase, PydanticAIWorkflow


class SupportRequest(AIRequestBase[None]):
    agent = support_agent


@workflow
class SupportWorkflow(PydanticAIWorkflow[SupportRequest]):
    async def run(self, request: SupportRequest) -> Reply:
        result = await self.run_agent(request)
        return result.output
```

`AIRequestBase` accepts a JSON-encoded `message_history`, per-run `deps`, a `model`
override, `conversation_id`, and `run_id`. It records the agent's module reference
so workers can load it.

### Start workers

Start PostgreSQL and create a database first, or use the
[Docker Compose example](#docker-compose-example) below.

Set `WAYMARK_DATABASE_URL` to your PostgreSQL database and `WAYMARK_USER_MODULE`
to the module containing your agents and workflows. Each worker must be able to
import that module. Run this from the project directory:

```bash
export WAYMARK_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/waymark
export WAYMARK_USER_MODULE=support_workflow
export OPENAI_API_KEY=...
uv run waymark-executor
```

With workers running, call the workflow from an async function in your application:

```python
from support_workflow import SupportRequest, SupportWorkflow


async def answer_question() -> str:
    reply = await SupportWorkflow().run(
        SupportRequest(prompt="How do I update my account?")
    )
    return reply.answer
```

The application needs the same `WAYMARK_DATABASE_URL`. The call waits for the
workflow to finish.

## Docker Compose example

The repository includes a Docker Compose setup with PostgreSQL, Waymark workers,
the Waymark dashboard, and a FastAPI form.

From the repository root, copy the environment file:

```bash
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`, then start the stack:

```bash
docker compose -f examples/docker-compose.yml up --build
```

Open [http://localhost:8000](http://localhost:8000) for the forms and
[http://localhost:24119](http://localhost:24119) for the dashboard.

The support form calls three tools. The sleep form lets you choose how long the
agent waits before it resumes. The recursive form runs
[`examples/recurse_agent.py`](examples/recurse_agent.py): an agent that splits a
question into parts, delegates each part to a copy of itself, and fact-checks claims
through a detached subagent and the Iceaxe mailbox.

Stop the stack and delete its example database with:

```bash
docker compose -f examples/docker-compose.yml down -v
```

## Multiple agents

The type parameter on `PydanticAIWorkflow` declares which request types
`run_agent` accepts. The workflow's input can contain several of those requests.
Add a second agent and a combined request to the module above:

```python
review_agent = waymark_agent(
    Agent(
        "openai:gpt-5.2",
        name="review_agent",
        instructions="Review the support question for risks that need human attention.",
        defer_model_check=True,
    )
)


class ReviewRequest(AIRequestBase[None]):
    agent = review_agent


class SupportReviewRequest(BaseModel):
    support: SupportRequest
    review: ReviewRequest


@workflow
class SupportReviewWorkflow(PydanticAIWorkflow[SupportRequest | ReviewRequest]):
    async def run(self, request: SupportReviewRequest) -> tuple[Reply, str]:
        support = await self.run_agent(request.support)
        review = await self.run_agent(request.review)
        return support.output, review.output
```

From an async function, pass each agent its own request:

```python
from support_workflow import (
    Reply,
    ReviewRequest,
    SupportRequest,
    SupportReviewRequest,
    SupportReviewWorkflow,
)


async def answer_and_review(question: str) -> tuple[Reply, str]:
    return await SupportReviewWorkflow().run(
        SupportReviewRequest(
            support=SupportRequest(prompt=question),
            review=ReviewRequest(prompt=question),
        )
    )
```

These calls run in order. See
[`ParallelSupportWorkflow`](examples/support_agent.py) for a version that runs
support and review concurrently.

## Tool execution

### Retries and timeouts

To set retries and a timeout, replace the earlier `lookup_policy` definition:

```python
@support_agent.tool_plain(
    retries=3,
    timeout=120,
)
def lookup_policy(topic: str) -> str:
    return f"Policy for {topic}: escalate account changes."
```

Timeouts and `ModelRetry` responses follow Pydantic AI's retry flow across
Waymark actions. Unhandled tool exceptions fail the workflow. Transient
model-provider failures use the request's `model_retry` backoff settings.

Tools that require approval currently stop the workflow with an error. To pause for a
human, use [message passing](#message-passing) instead.

### Sequential tools

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

### Durable waits

Raise `DurableSleep` to wait before returning a tool result. Waymark stores the
timer; when it expires, the supplied result goes back to the model under the
original tool-call ID:

```python
from pydantic_ai_waymark import DurableSleep


@support_agent.tool_plain
def wait_for_follow_up(seconds: float = 5) -> str:
    raise DurableSleep(seconds, result="Follow-up wait completed.")
```

## Subagents

A tool hands work to another agent by raising `DelegateAgent` with that agent's
request. Tools can either wait for the subagent's answer or start it and check on it
later.

### Wait for the subagent

By default the workflow runs the subagent through `run_agent` as nested durable work:
each subagent model and tool step is its own action, so a worker restart resumes partway
through the subagent instead of rerunning it. The subagent's output becomes the tool
result. Add the subagent's request type to the workflow's type parameter:

```python
from pydantic_ai_waymark import DelegateAgent

research_agent = waymark_agent(
    Agent("openai:gpt-5.2", name="research_agent", defer_model_check=True)
)


class ResearchRequest(AIRequestBase[None]):
    agent = research_agent


@support_agent.tool_plain
def ask_researcher(question: str) -> str:
    """Ask the research agent a question."""
    raise DelegateAgent(ResearchRequest(prompt=question))


@workflow
class ResearchedSupportWorkflow(PydanticAIWorkflow[SupportRequest | ResearchRequest]):
    async def run(self, request: SupportRequest) -> Reply:
        result = await self.run_agent(request)
        return result.output
```

Subagents can use `DurableSleep`, wait for messages, and delegate further.
[`examples/recurse_agent.py`](examples/recurse_agent.py) shows an agent that delegates to
itself, with a depth limit carried in its dependencies. When the model
calls several delegating tools in one turn, their subagents run in parallel. Lifecycle
hooks fire for subagents too, with the subagent's request. Subagent token usage is
reported in the subagent's own `AgentResult`, not added to the delegating agent's usage.

### Start the subagent and check later

With `wait=False`, the subagent starts as its own Waymark workflow instance and the tool
returns a handle right away:

```json
{"subagent_id": "...", "mailbox": "waymark-subagent:...", "status": "started"}
```

When the subagent finishes, it sends a report to that mailbox through the delegating
agent's `message_writer` (see [Message passing](#message-passing)). The report is a JSON
message: `{"subagent_id", "status": "completed", "output"}`, or
`{"subagent_id", "status": "failed", "error"}`. A later tool checks the mailbox once
with `timeout_seconds=0`, or waits for the report:

```python
@support_agent.tool_plain
def start_research(question: str) -> str:
    """Start the research agent without waiting for it."""
    raise DelegateAgent(ResearchRequest(prompt=question), wait=False)


@support_agent.tool_plain
def check_research(mailbox: str) -> str:
    """Check whether the research agent has reported back."""
    raise WaitForMessages(mailbox, timeout_seconds=0, timeout_result="still running")
```

Detached subagents need the delegating agent to register both `message_reader` and
`message_writer`, and their request types don't need to appear in the workflow's
type parameter. Waymark does not deduplicate workflow instances, so if the action that
starts a detached subagent is retried after the instance was queued, the subagent may run
twice. Its report is still stored once, because the report uses the subagent ID as its
`dedupe_key`.

## Message passing

> **Note:** This is a polling workaround. Waymark has no native message queues yet, and
> those are likely the long-term way to pass messages between workflows. Until then,
> tools wait on mailboxes that are read through a callback you provide.

A tool raises `WaitForMessages` to wait for messages on a mailbox. The workflow polls the
agent's `message_reader` every `poll_seconds`, with a durable sleep between polls. It
returns the first non-empty batch to the model as a list of
`{"id", "content", "sender", "sent_at"}` objects. With `timeout_seconds`, the tool returns
`timeout_result` once that much polling time has passed.

```python
from pydantic_ai import RunContext
from pydantic_ai_waymark import WaitForMessages


@support_agent.tool
def wait_for_operator(ctx: RunContext[None]) -> str:
    """Wait for a human operator to reply."""
    raise WaitForMessages(f"operator:{ctx.run_id}", poll_seconds=1, timeout_seconds=3600)
```

Register the hooks alongside the agent's codecs. Like codecs, they can be sync or async
and can declare `Depends(...)` parameters:

```python
from pydantic_ai_waymark import AgentMessage, MessageQuery, OutgoingMessage


async def read_messages(
    query: MessageQuery, db=Depends(get_db_connection)
) -> list[AgentMessage]: ...


async def write_message(message: OutgoingMessage, db=Depends(get_db_connection)) -> None: ...


support_agent = waymark_agent(
    support_agent,
    message_reader=read_messages,
    message_writer=write_message,
)
```

Waymark may rerun a poll or a send, so both hooks must be idempotent:

- A reader must return the same messages again for a `claim_key` it has already
  claimed for. `claim_key` is stable for one waiting tool call.
- A writer must store each `dedupe_key` at most once.

### Iceaxe mailbox

The `iceaxe` extra ships a ready-made mailbox table and hooks that follow these rules:

```bash
uv add "pydantic-ai-waymark[iceaxe]"
```

```python
from pydantic_ai_waymark.iceaxe import (
    AgentMessageRecord,
    message_reader,
    message_writer,
    send_message,
)

support_agent = waymark_agent(
    support_agent,
    message_reader=message_reader(get_db_connection),
    message_writer=message_writer(get_db_connection),
)


async def reply_as_operator(db: DBConnection, run_id: str, text: str) -> None:
    await send_message(db, mailbox=f"operator:{run_id}", content=text, sender="operator")
```

`get_db_connection` is a `mountaineer_di` dependency provider that yields an Iceaxe
`DBConnection`. Import `AgentMessageRecord` into the models module your Iceaxe migrations
scan so the `waymark_agent_messages` table is created. Readers claim rows with
`SELECT ... FOR UPDATE SKIP LOCKED`, so concurrent waiters on the same mailbox each
receive different messages.

## Lifecycle hooks

Override `on_agent_start`, `on_agent_end`, `on_message`, `on_tool_start`, or
`on_tool_end` to report progress. Put database writes and network calls in
Waymark actions so their results are persisted.

This example publishes tool-start events through an application-provided
`websocket_service`:

```python
from typing import Any

from waymark import action, workflow


@action
async def publish_event(event: str, payload: dict[str, Any]) -> None:
    await websocket_service.publish(event, payload)


@workflow
class ObservableSupportWorkflow(PydanticAIWorkflow[SupportRequest]):
    async def on_tool_start(
        self,
        agent_request: SupportRequest,
        tool_id: str,
        tool_args: object,
    ) -> None:
        await self.run_action(
            publish_event(
                event="tool.started",
                payload={"tool_id": tool_id, "args": tool_args},
            )
        )

    async def run(self, request: SupportRequest) -> Reply:
        result = await self.run_agent(request)
        return result.output
```

`on_message` receives each plain-text model message. `on_agent_end` receives the
complete `AgentResult`; `on_tool_end` receives a tool-result dictionary. If the
receiver may see retries, deduplicate tool events by run, event name, and `tool_id`.

## Payload codecs

Use payload codecs to store large values, such as images, outside Waymark
snapshots. Register a serializer and deserializer together to replace those values
with references when saving and restore them when loading.

The database connection and reference-storage helpers below are application code:

```python
from mountaineer_di import Depends
from pydantic_ai_waymark import Payload, SerializedPayload


async def serialize_payload(
    payload: Payload,
    db=Depends(get_db_connection),
) -> SerializedPayload:
    value = await replace_large_values_with_database_refs(db, payload.to_python())
    return payload.serialized(value)


async def deserialize_payload(
    payload: SerializedPayload,
    db=Depends(get_db_connection),
) -> Payload:
    value = await restore_database_refs(db, payload.value)
    return payload.deserialized(value)


support_agent = waymark_agent(
    support_agent,
    serializer=serialize_payload,
    deserializer=deserialize_payload,
)
```

`Payload` covers graph state, messages, agent output, tool output, tool action
results, deferred tool results, and user prompts. Use `payload.kind` to handle
these types separately. `payload.to_python()` gives you the Python tree to
transform; `serialized(...)` and `deserialized(...)` preserve and validate its
kind across the round trip.

Codecs can be sync or async and can declare `mountaineer_di.Depends(...)`
parameters. Generator dependencies stay open for the call and are cleaned up
afterward.

The workflow keeps one current checkpoint. Intermediate model-action results
store only new messages, avoiding growing copies of the full history. Graph-state
payloads exclude message history; `messages` payloads are reassembled before each
agent step. The final result includes the complete history.

Use stable keys or upserts when storing large values so repeated serialization
reuses the same reference.

Waymark actions also accept dependency providers. Use them to inject database or
storage clients without adding those clients to Pydantic AI's per-run `deps`:

```python
from waymark import action


@action
async def save_result(result: dict, db=Depends(get_db_connection)) -> None:
    await db.insert(result)
```

## Checks

The Iceaxe mailbox tests run when `WAYMARK_AI_TEST_DATABASE_URL` points at a PostgreSQL
database, and are skipped otherwise:

```bash
export WAYMARK_AI_TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/waymark_ai_test
uv run pytest -q
uv run ruff check .
uv run ty check
```
