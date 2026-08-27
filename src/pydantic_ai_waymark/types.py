from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeAlias, TypedDict

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
WorkflowToolArgs: TypeAlias = dict[str, JsonValue]
ToolMetadata: TypeAlias = dict[str, dict[str, JsonValue]]
# Waymark actions receive decoded mappings before handlers refine them into the
# discriminated payloads below.
WirePayload: TypeAlias = Mapping[str, object]


class UsagePayload(TypedDict):
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


class AgentResult(TypedDict):
    output: AgentOutput
    message_history: str
    new_messages: str
    usage: UsagePayload
    run_id: str
    conversation_id: str


class SerializedToolCall(TypedDict):
    tool_name: str
    args: str | dict[str, JsonValue] | None
    tool_call_id: str
    tool_kind: Literal["tool-search", "capability-load"] | None
    id: str | None
    provider_name: str | None
    provider_details: dict[str, JsonValue] | None
    part_kind: Literal["tool-call"]


class WaymarkToolPolicy(TypedDict):
    attempts: int
    backoff_seconds: float
    timeout: float


class ActionToolCall(TypedDict):
    call: SerializedToolCall
    waymark: WaymarkToolPolicy
    workflow_args: None


class WorkflowToolCall(TypedDict):
    call: SerializedToolCall
    waymark: Literal[False]
    workflow_args: WorkflowToolArgs


ToolCall: TypeAlias = ActionToolCall | WorkflowToolCall


class ModelRequestNodePayload(TypedDict):
    kind: Literal["model_request"]
    request: str
    is_resuming_without_prompt: bool
    resume_suspended: str | None


class CallToolsNodePayload(TypedDict):
    kind: Literal["call_tools"]
    model_response: str
    tool_call_results: dict[str, DeferredToolResult | Literal["skip"]] | None
    tool_call_metadata: ToolMetadata | None
    user_prompt: UserPrompt


AgentNodePayload: TypeAlias = ModelRequestNodePayload | CallToolsNodePayload


class DepsState(TypedDict):
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


class RunningTransition(TypedDict):
    result: None
    state: str
    node: AgentNodePayload
    deps_state: DepsState
    tool_calls: list[ToolCall]
    approvals: list[SerializedToolCall]
    tool_metadata: ToolMetadata


class NodeTransition(RunningTransition):
    kind: Literal["node"]


class ToolsTransition(RunningTransition):
    kind: Literal["tools"]


class DoneTransition(TypedDict):
    kind: Literal["done"]
    result: AgentResult
    state: None
    node: None
    deps_state: None
    tool_calls: list[ToolCall]
    approvals: list[SerializedToolCall]
    tool_metadata: ToolMetadata


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


ToolActionResult: TypeAlias = (
    ToolReturnResult
    | ModelRetryResult
    | ToolFailedResult
    | RetryPromptResult
    | DeferredToolResultPayload
)
