import asyncio
import json
from typing import Any, cast

import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from waymark import action, workflow
from waymark.exceptions import WorkflowFailedError

from pydantic_ai_waymark import (
    AgentMessage,
    AIRequestBase,
    DelegateAgent,
    DurableSleep,
    MessageQuery,
    OutgoingMessage,
    Payload,
    PydanticAIWorkflow,
    SerializedPayload,
    WaitForMessages,
    waymark_agent,
)


@pytest.fixture(autouse=True)
def reset_mutable_test_state() -> None:
    agent_starts.clear()
    mailboxes.clear()
    message_polls.clear()
    empty_polls_before_delivery[0] = 0
    sent_messages.clear()
    codec_kinds.clear()


agent_starts: list[str] = []
mailboxes: dict[str, list[AgentMessage]] = {}
message_polls: list[MessageQuery] = []
empty_polls_before_delivery = [0]
codec_kinds: list[str] = []
sent_messages: dict[str, OutgoingMessage] = {}


def serialize_wrapped(payload: Payload) -> SerializedPayload:
    codec_kinds.append(payload.kind)
    value = payload.to_python()
    if payload.kind == "agent_output":
        value = {"wrapped": value}
    return payload.serialized(value)


def deserialize_wrapped(payload: SerializedPayload) -> Payload:
    value = payload.value
    if payload.kind == "agent_output":
        assert isinstance(value, dict)
        value = value["wrapped"]
    return payload.deserialized(value)


def read_test_mailbox(query: MessageQuery) -> list[AgentMessage]:
    message_polls.append(query)
    if len(message_polls) <= empty_polls_before_delivery[0]:
        return []
    return mailboxes.pop(query.mailbox, [])


def write_test_mailbox(message: OutgoingMessage) -> None:
    if message.dedupe_key in sent_messages:
        return
    sent_messages[message.dedupe_key] = message
    mailboxes.setdefault(message.mailbox, []).append(
        AgentMessage(id=message.dedupe_key, content=message.content, sender=message.sender)
    )


def last_tool_return(messages: list[ModelMessage]) -> ToolReturnPart | None:
    for part in messages[-1].parts:
        if isinstance(part, ToolReturnPart):
            return part
    return None


def drive_detached_manager(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Start a detached subagent, check on it from a later step, then answer."""
    returned = last_tool_return(messages)
    if returned is None:
        return ModelResponse(parts=[ToolCallPart("start_research", {"question": "why"})])
    if returned.tool_name == "start_research":
        handle = cast(dict[str, Any], returned.content)
        assert handle["status"] == "started"
        return ModelResponse(parts=[ToolCallPart("check_research", {"mailbox": handle["mailbox"]})])
    return ModelResponse(parts=[TextPart(json.dumps(returned.content))])


research_agent = waymark_agent(
    Agent(TestModel(custom_output_text="researched"), name="research_agent")
)
pausing_research_agent = waymark_agent(
    Agent(TestModel(call_tools=["pause"], custom_output_text="paused"), name="pausing_research")
)
encoded_research_agent = waymark_agent(
    Agent(TestModel(custom_output_text="encoded"), name="encoded_research"),
    serializer=serialize_wrapped,
    deserializer=deserialize_wrapped,
)
manager_agent = waymark_agent(Agent(TestModel(call_tools=["ask_researcher"]), name="manager_agent"))
parallel_manager_agent = waymark_agent(
    Agent(TestModel(call_tools=["ask_first", "ask_second"]), name="parallel_manager")
)
nested_manager_agent = waymark_agent(
    Agent(TestModel(call_tools=["ask_pausing_researcher"]), name="nested_manager")
)
encoded_manager_agent = waymark_agent(
    Agent(TestModel(call_tools=["ask_encoded_researcher"]), name="encoded_manager")
)
listening_agent = waymark_agent(
    Agent(TestModel(call_tools=["wait_for_reply"]), name="listening_agent"),
    message_reader=read_test_mailbox,
)
impatient_agent = waymark_agent(
    Agent(TestModel(call_tools=["wait_briefly"]), name="impatient_agent"),
    message_reader=read_test_mailbox,
)
prompt_reading_agent = waymark_agent(
    Agent(TestModel(call_tools=["echo_prompt"]), name="prompt_reading_agent")
)
deaf_agent = waymark_agent(Agent(TestModel(call_tools=["wait_unheard"]), name="deaf_agent"))


detached_manager_agent = waymark_agent(
    Agent(FunctionModel(drive_detached_manager), name="detached_manager"),
    message_reader=read_test_mailbox,
    message_writer=write_test_mailbox,
)
writerless_manager_agent = waymark_agent(
    Agent(TestModel(call_tools=["start_unreported"]), name="writerless_manager")
)
failing_research_agent = waymark_agent(
    Agent(TestModel(call_tools=["explode"]), name="failing_research")
)
failing_detached_manager_agent = waymark_agent(
    Agent(TestModel(call_tools=["start_failing"]), name="failing_detached_manager"),
    message_writer=write_test_mailbox,
)


class ResearchRequest(AIRequestBase[None]):
    agent = research_agent


class PausingResearchRequest(AIRequestBase[None]):
    agent = pausing_research_agent


class EncodedResearchRequest(AIRequestBase[None]):
    agent = encoded_research_agent


class ManagerRequest(AIRequestBase[None]):
    agent = manager_agent


class ParallelManagerRequest(AIRequestBase[None]):
    agent = parallel_manager_agent


class NestedManagerRequest(AIRequestBase[None]):
    agent = nested_manager_agent


class EncodedManagerRequest(AIRequestBase[None]):
    agent = encoded_manager_agent


class ListeningRequest(AIRequestBase[None]):
    agent = listening_agent


class ImpatientRequest(AIRequestBase[None]):
    agent = impatient_agent


class PromptReadingRequest(AIRequestBase[None]):
    agent = prompt_reading_agent


class DeafRequest(AIRequestBase[None]):
    agent = deaf_agent


class DetachedManagerRequest(AIRequestBase[None]):
    agent = detached_manager_agent


class WriterlessManagerRequest(AIRequestBase[None]):
    agent = writerless_manager_agent


class FailingResearchRequest(AIRequestBase[None]):
    agent = failing_research_agent


class FailingDetachedManagerRequest(AIRequestBase[None]):
    agent = failing_detached_manager_agent


@detached_manager_agent.tool_plain
def start_research(question: str) -> str:
    """Start the research agent without waiting for it."""
    raise DelegateAgent(ResearchRequest(prompt=question), wait=False)


@detached_manager_agent.tool_plain
def check_research(mailbox: str) -> str:
    """Check once whether the research agent has reported back."""
    raise WaitForMessages(mailbox, timeout_seconds=0, timeout_result="still running")


@writerless_manager_agent.tool_plain
def start_unreported(question: str) -> str:
    raise DelegateAgent(ResearchRequest(prompt=question), wait=False)


@failing_research_agent.tool_plain
def explode() -> str:
    raise ValueError("research broke")


@failing_detached_manager_agent.tool_plain
def start_failing(question: str) -> str:
    raise DelegateAgent(FailingResearchRequest(prompt=question), wait=False)


@manager_agent.tool_plain
def ask_researcher(question: str) -> str:
    """Ask the research agent."""
    raise DelegateAgent(ResearchRequest(prompt=question))


@parallel_manager_agent.tool_plain
def ask_first(question: str) -> str:
    raise DelegateAgent(ResearchRequest(prompt=f"first {question}"))


@parallel_manager_agent.tool_plain
def ask_second(question: str) -> str:
    raise DelegateAgent(ResearchRequest(prompt=f"second {question}"))


@nested_manager_agent.tool_plain
def ask_pausing_researcher(question: str) -> str:
    raise DelegateAgent(PausingResearchRequest(prompt=question))


@pausing_research_agent.tool_plain
def pause() -> str:
    raise DurableSleep(0.001, result="rested")


@encoded_manager_agent.tool_plain
def ask_encoded_researcher(question: str) -> str:
    raise DelegateAgent(EncodedResearchRequest(prompt=question))


@listening_agent.tool_plain
def wait_for_reply() -> str:
    raise WaitForMessages("inbox", poll_seconds=0.001)


@impatient_agent.tool_plain
def wait_briefly() -> str:
    raise WaitForMessages(
        "quiet-inbox",
        poll_seconds=0.001,
        timeout_seconds=0.002,
        timeout_result="nobody answered",
    )


@prompt_reading_agent.tool
def echo_prompt(ctx: RunContext[None]) -> str:
    return f"prompt={ctx.prompt}"


@deaf_agent.tool_plain
def wait_unheard() -> str:
    raise WaitForMessages("inbox")


@action
async def record_agent_start(agent_reference: str) -> None:
    agent_starts.append(agent_reference)


@workflow
class ManagerWorkflow(PydanticAIWorkflow[ManagerRequest | ResearchRequest]):
    async def on_agent_start(self, agent_request: ManagerRequest | ResearchRequest) -> None:
        await self.run_action(record_agent_start(agent_request.agent_reference))

    async def run(self, request: ManagerRequest) -> str:
        result = await self.run_agent(request)
        return result.output


@workflow
class ParallelManagerWorkflow(PydanticAIWorkflow[ParallelManagerRequest | ResearchRequest]):
    async def run(self, request: ParallelManagerRequest) -> str:
        result = await self.run_agent(request)
        return result.output


@workflow
class NestedManagerWorkflow(PydanticAIWorkflow[NestedManagerRequest | PausingResearchRequest]):
    async def run(self, request: NestedManagerRequest) -> str:
        result = await self.run_agent(request)
        return result.output


@workflow
class EncodedManagerWorkflow(PydanticAIWorkflow[EncodedManagerRequest | EncodedResearchRequest]):
    async def run(self, request: EncodedManagerRequest) -> str:
        result = await self.run_agent(request)
        return result.output


@workflow
class ListeningWorkflow(PydanticAIWorkflow[ListeningRequest]):
    async def run(self, request: ListeningRequest) -> str:
        result = await self.run_agent(request)
        return result.output


@workflow
class ImpatientWorkflow(PydanticAIWorkflow[ImpatientRequest]):
    async def run(self, request: ImpatientRequest) -> str:
        result = await self.run_agent(request)
        return result.output


@workflow
class PromptReadingWorkflow(PydanticAIWorkflow[PromptReadingRequest]):
    async def run(self, request: PromptReadingRequest) -> str:
        result = await self.run_agent(request)
        return result.output


@workflow
class DeafWorkflow(PydanticAIWorkflow[DeafRequest]):
    async def run(self, request: DeafRequest) -> str:
        result = await self.run_agent(request)
        return result.output


@workflow
class DetachedManagerWorkflow(PydanticAIWorkflow[DetachedManagerRequest]):
    async def run(self, request: DetachedManagerRequest) -> str:
        result = await self.run_agent(request)
        return result.output


@workflow
class WriterlessManagerWorkflow(PydanticAIWorkflow[WriterlessManagerRequest]):
    async def run(self, request: WriterlessManagerRequest) -> str:
        result = await self.run_agent(request)
        return result.output


@workflow
class FailingDetachedManagerWorkflow(PydanticAIWorkflow[FailingDetachedManagerRequest]):
    async def run(self, request: FailingDetachedManagerRequest) -> str:
        result = await self.run_agent(request)
        return result.output


def test_delegated_subagent_output_becomes_the_tool_result() -> None:
    result = asyncio.run(ManagerWorkflow().run(ManagerRequest(prompt="find it")))

    assert result == '{"ask_researcher":"researched"}'
    assert agent_starts == [
        f"{__name__}:manager_agent",
        f"{__name__}:research_agent",
    ]
    ir_text = str(ManagerWorkflow.workflow_ir())
    assert "pydantic_ai_agent_delegate_result" in ir_text


def test_subagent_steps_run_as_their_own_durable_actions() -> None:
    result = asyncio.run(NestedManagerWorkflow().run(NestedManagerRequest(prompt="find it")))

    # The subagent's DurableSleep only works if its tools run inside the workflow.
    assert result == '{"ask_pausing_researcher":"paused"}'


def test_parallel_tool_calls_run_their_subagents_concurrently() -> None:
    result = asyncio.run(ParallelManagerWorkflow().run(ParallelManagerRequest(prompt="find it")))

    assert result == '{"ask_first":"researched","ask_second":"researched"}'


def test_subagent_output_is_translated_between_payload_codecs() -> None:
    result = asyncio.run(EncodedManagerWorkflow().run(EncodedManagerRequest(prompt="find it")))

    # The manager has no codec, so it must receive the plain output, not the wrapper.
    assert result == '{"ask_encoded_researcher":"encoded"}'
    assert "agent_output" in codec_kinds


def test_waiting_tool_polls_its_mailbox_until_a_message_arrives() -> None:
    empty_polls_before_delivery[0] = 2
    mailboxes["inbox"] = [AgentMessage(id="1", content="hello", sender="human")]

    result = asyncio.run(ListeningWorkflow().run(ListeningRequest(prompt="wait")))

    assert result == '{"wait_for_reply":[{"id":"1","content":"hello","sender":"human"}]}'
    assert len(message_polls) == 3
    assert {poll.claim_key for poll in message_polls} == {message_polls[0].claim_key}
    assert message_polls[0].mailbox == "inbox"
    assert message_polls[0].agent_reference == f"{__name__}:listening_agent"
    assert (
        message_polls[0].claim_key == f"{message_polls[0].run_id}:{message_polls[0].tool_call_id}"
    )


def test_waiting_tool_returns_its_timeout_result_when_no_message_arrives() -> None:
    result = asyncio.run(ImpatientWorkflow().run(ImpatientRequest(prompt="wait")))

    assert result == '{"wait_briefly":"nobody answered"}'
    assert len(message_polls) == 3


def test_waiting_without_a_message_reader_fails_the_workflow() -> None:
    with pytest.raises(WorkflowFailedError, match="registers no message_reader"):
        asyncio.run(DeafWorkflow().run(DeafRequest(prompt="wait")))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"poll_seconds": 0}, "poll interval"),
        ({"poll_seconds": float("inf")}, "poll interval"),
        ({"timeout_seconds": -1}, "message timeout"),
        ({"timeout_seconds": float("nan")}, "message timeout"),
    ],
)
def test_wait_for_messages_rejects_invalid_settings(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        WaitForMessages("inbox", **kwargs)


def test_wait_for_messages_requires_a_mailbox() -> None:
    with pytest.raises(ValueError, match="mailbox"):
        WaitForMessages("")


def test_detached_subagent_returns_a_handle_and_reports_to_its_mailbox() -> None:
    result = asyncio.run(DetachedManagerWorkflow().run(DetachedManagerRequest(prompt="go")))

    [delivered] = json.loads(result)
    report = json.loads(delivered["content"])
    assert report["status"] == "completed"
    assert report["output"] == "researched"
    assert delivered["sender"] == f"{__name__}:research_agent"
    [sent] = sent_messages.values()
    assert sent.mailbox == f"waymark-subagent:{report['subagent_id']}"
    assert sent.dedupe_key == report["subagent_id"]
    # The detached subagent runs in its own workflow, not as nested work of this one.
    ir_text = str(DetachedManagerWorkflow.workflow_ir())
    assert "pydantic_ai_agent_start_detached_subagent" in ir_text


def test_detached_subagent_requires_a_message_writer() -> None:
    with pytest.raises(WorkflowFailedError, match="registers no message_writer"):
        asyncio.run(WriterlessManagerWorkflow().run(WriterlessManagerRequest(prompt="go")))


def test_failed_detached_subagent_reports_its_error() -> None:
    # NOTE: Under pytest Waymark runs the "detached" workflow inline, so its failure also
    # reaches this parent; in production only the subagent's own instance fails.
    with pytest.raises(WorkflowFailedError):
        asyncio.run(
            FailingDetachedManagerWorkflow().run(FailingDetachedManagerRequest(prompt="go"))
        )

    [sent] = sent_messages.values()
    report = json.loads(sent.content)
    assert report["status"] == "failed"
    assert "research broke" in report["error"]


def test_tools_see_the_run_prompt_inside_their_action() -> None:
    result = asyncio.run(PromptReadingWorkflow().run(PromptReadingRequest(prompt="the question")))

    assert result == '{"echo_prompt":"prompt=the question"}'
