from collections.abc import Sequence
from typing import Any, Literal, TypeAlias, TypedDict

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import _agent_graph
from pydantic_ai.agent import AbstractAgent
from pydantic_ai.messages import UserContent
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


class WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class UsagePayload(WireModel):
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


class WaymarkToolPolicy(WireModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    attempts: int = Field(default=3, ge=1)
    backoff_seconds: float = Field(default=2.0, ge=0)
    timeout: float = Field(default=120.0, gt=0)


class ToolCall(WireModel):
    call: SerializedToolCall
    waymark: WaymarkToolPolicy
    sequential: bool = False


class ModelRequestNodePayload(WireModel):
    kind: Literal["model_request"]
    request: str
    is_resuming_without_prompt: bool
    resume_suspended: str | None


class CallToolsNodePayload(WireModel):
    kind: Literal["call_tools"]
    model_response: str
    tool_call_results: dict[str, DeferredToolResult | Literal["skip"]] | None
    tool_call_metadata: ToolMetadata | None
    user_prompt: UserPrompt


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
