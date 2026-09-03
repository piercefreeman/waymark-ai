from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Concatenate, Literal, TypeAlias, TypedDict

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from pydantic_ai import ModelMessagesTypeAdapter, _agent_graph
from pydantic_ai.agent import AbstractAgent
from pydantic_ai.messages import ModelMessage, UserContent
from pydantic_ai.run import AgentRun
from pydantic_ai.tools import DeferredToolResult
from pydantic_core import ErrorDetails

# These three values are deliberately dynamic: users choose the dependency and
# output types when defining an agent, and agents are looked up by name at runtime.
AgentDeps: TypeAlias = Any
AgentOutput: TypeAlias = Any
ToolValue: TypeAlias = Any

RegisteredAgent: TypeAlias = AbstractAgent[AgentDeps, AgentOutput]
RegisteredAgentRun: TypeAlias = AgentRun[AgentDeps, AgentOutput]
PydanticRunNode: TypeAlias = _agent_graph.AgentNode[AgentDeps, AgentOutput]
PersistedPydanticNode: TypeAlias = (
    _agent_graph.ModelRequestNode[AgentDeps, AgentOutput]
    | _agent_graph.CallToolsNode[AgentDeps, AgentOutput]
)
UserPrompt: TypeAlias = str | Sequence[UserContent] | None

type JsonValue = str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]
ToolMetadata: TypeAlias = dict[str, dict[str, JsonValue]]
_json_value_adapter = TypeAdapter(Any)
_graph_state_adapter = TypeAdapter(_agent_graph.GraphAgentState)


class WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class BackoffConfig(WireModel):
    """Bounded exponential backoff for retryable model-request failures."""

    attempts: int = Field(default=3, ge=1)
    initial_seconds: float = Field(default=2.0, ge=0)
    multiplier: float = Field(default=2.0, ge=1)
    max_seconds: float = Field(default=30.0, ge=0)


class UsagePayload(WireModel):
    # Pydantic AI deliberately preserves provider-specific usage counters such as
    # ``output_reasoning_tokens`` alongside its stable fields.
    model_config = ConfigDict(extra="allow")
    __pydantic_extra__: dict[str, JsonValue] = Field(init=False)

    input_tokens: int
    cache_write_tokens: int
    cache_read_tokens: int
    output_tokens: int
    input_audio_tokens: int
    cache_audio_read_tokens: int
    output_audio_tokens: int
    details: dict[str, int]
    requests: int
    tool_calls: int


class AgentResult(WireModel):
    output: AgentOutput
    message_history: str
    new_messages: str
    usage: UsagePayload
    run_id: str
    conversation_id: str


class SerializedToolCall(WireModel):
    tool_name: str
    args: str | dict[str, JsonValue] | None
    tool_call_id: str
    tool_kind: Literal["tool-search", "capability-load"] | None
    id: str | None
    provider_name: str | None
    provider_details: dict[str, JsonValue] | None
    part_kind: Literal["tool-call"]


class ToolCall(WireModel):
    call: SerializedToolCall
    sequential: bool = False


class ModelRequestNodePayload(WireModel):
    kind: Literal["model_request"]
    request: str
    is_resuming_without_prompt: bool
    resume_suspended: str | None


class CallToolsNodePayload(WireModel):
    kind: Literal["call_tools"]
    model_response: str
    tool_call_results: Any
    tool_call_metadata: ToolMetadata | None
    user_prompt: Any


AgentNodePayload: TypeAlias = ModelRequestNodePayload | CallToolsNodePayload


class DepsState(WireModel):
    new_message_index: int
    resumed_request: str | None
    resumed_request_index: int | None
    model_id: str | None
    model_selected_for_step: int | None
    loaded_capability_ids: list[str]
    discovered_tool_names: list[str]
    tool_run_step: int | None
    tool_retries: dict[str, int]
    failed_tools: list[str]
    succeeded_tools: list[str]


class RunningTransition(WireModel):
    result: None = None
    state: str
    node: AgentNodePayload
    deps_state: DepsState
    history_delta: str | None = None
    messages: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    approvals: list[SerializedToolCall] = Field(default_factory=list)
    tool_metadata: ToolMetadata = Field(default_factory=dict)


class NodeTransition(RunningTransition):
    kind: Literal["node"]


class ToolsTransition(RunningTransition):
    kind: Literal["tools"]


class DoneTransition(WireModel):
    kind: Literal["done"]
    result: AgentResult
    state: None = None
    node: None = None
    deps_state: None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    approvals: list[SerializedToolCall] = Field(default_factory=list)
    tool_metadata: ToolMetadata = Field(default_factory=dict)


PendingTransition: TypeAlias = NodeTransition | ToolsTransition
AgentTransition: TypeAlias = PendingTransition | DoneTransition


class AgentCheckpoint(WireModel):
    """Only the state needed to execute the agent's next durable step."""

    state: str | None = None
    node: AgentNodePayload | None = None
    deps_state: DepsState | None = None
    tool_metadata: ToolMetadata = Field(default_factory=dict)
    message_history: list[str] = Field(default_factory=list)


class ToolResultBase(TypedDict):
    tool_call_id: str
    tool_name: str


class ToolReturnResult(ToolResultBase):
    kind: Literal["return"]
    value: ToolValue


class ModelRetryResult(ToolResultBase):
    kind: Literal["model_retry"]
    message: str


class ToolFailedResult(ToolResultBase):
    kind: Literal["tool_failed"]
    message: str


class RetryPromptResult(ToolResultBase):
    kind: Literal["retry_prompt"]
    content: str | list[ErrorDetails]


class DeferredToolResultPayload(ToolResultBase):
    kind: Literal["deferred"]


class DurableSleepResult(ToolResultBase):
    kind: Literal["sleep"]
    seconds: float
    value: ToolValue


ToolActionResult: TypeAlias = (
    ToolReturnResult
    | ModelRetryResult
    | ToolFailedResult
    | RetryPromptResult
    | DeferredToolResultPayload
    | DurableSleepResult
)

PayloadKind: TypeAlias = Literal[
    "graph_state",
    "messages",
    "agent_output",
    "tool_output",
    "tool_action_results",
    "deferred_tool_results",
    "user_prompt",
]


class SerializedPayload(WireModel):
    kind: PayloadKind
    value: JsonValue

    def deserialized(self, value: object) -> "Payload":
        if self.kind == "graph_state":
            value = _graph_state_adapter.validate_python(value)
        elif self.kind == "messages":
            value = ModelMessagesTypeAdapter.validate_python(value)
        return _payload_adapter.validate_python({"kind": self.kind, "value": value})


class PayloadBase(WireModel):
    kind: PayloadKind
    value: object

    def to_python(self) -> object:
        return self.model_dump(mode="python")["value"]

    def serialized(self, value: object) -> SerializedPayload:
        json_value = _json_value_adapter.dump_python(value, mode="json")
        return SerializedPayload(kind=self.kind, value=json_value)


class GraphStatePayload(PayloadBase):
    kind: Literal["graph_state"] = "graph_state"
    value: _agent_graph.GraphAgentState


class MessagesPayload(PayloadBase):
    kind: Literal["messages"] = "messages"
    value: list[ModelMessage]


class AgentOutputPayload(PayloadBase):
    kind: Literal["agent_output"] = "agent_output"
    value: AgentOutput


class ToolOutputPayload(PayloadBase):
    kind: Literal["tool_output"] = "tool_output"
    value: ToolValue


class ToolActionResultsPayload(PayloadBase):
    kind: Literal["tool_action_results"] = "tool_action_results"
    value: list[ToolActionResult]


class DeferredToolResultsPayload(PayloadBase):
    kind: Literal["deferred_tool_results"] = "deferred_tool_results"
    value: dict[str, DeferredToolResult | Literal["skip"]] | None


class UserPromptPayload(PayloadBase):
    kind: Literal["user_prompt"] = "user_prompt"
    value: UserPrompt


Payload: TypeAlias = (
    GraphStatePayload
    | MessagesPayload
    | AgentOutputPayload
    | ToolOutputPayload
    | ToolActionResultsPayload
    | DeferredToolResultsPayload
    | UserPromptPayload
)
_payload_adapter = TypeAdapter(Payload)
PayloadSerializer: TypeAlias = Callable[
    Concatenate[Payload, ...],
    SerializedPayload | Awaitable[SerializedPayload],
]
PayloadDeserializer: TypeAlias = Callable[
    Concatenate[SerializedPayload, ...],
    Payload | Awaitable[Payload],
]
