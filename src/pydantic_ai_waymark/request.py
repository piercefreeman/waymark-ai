from importlib import import_module
from typing import Any, ClassVar, Generic, Self, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_ai import Agent

from .registry import agent_reference as serialize_agent_reference
from .types import BackoffConfig

AgentDependenciesT = TypeVar("AgentDependenciesT")


class AIRequestBase(BaseModel, Generic[AgentDependenciesT]):
    """Serializable input for a durable Pydantic AI workflow."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    #: Module-level Pydantic AI agent this request runs. Each concrete request
    #: subclass must assign its registered agent here.
    agent: ClassVar[Agent[Any, Any]]

    #: User prompt passed to the agent. Use ``None`` when resuming without a new prompt.
    prompt: str | None

    #: JSON-encoded Pydantic AI message history used to seed the first graph transition.
    message_history: str | None = None

    #: Per-run dependencies passed to agent instructions, tools, and other hooks.
    deps: AgentDependenciesT | None = None

    #: Optional Pydantic AI model override for this run; otherwise the agent's model is used.
    model: str | None = None

    #: Optional stable conversation identifier supplied only when starting the agent run.
    conversation_id: str | None = None

    #: Optional stable run identifier supplied only when starting the agent run.
    run_id: str | None = None

    #: Bounded backoff policy for retryable provider failures during graph-node actions.
    model_retry: BackoffConfig = Field(default_factory=BackoffConfig)

    #: Importable ``module:variable`` reference for ``agent``. It is populated and
    #: validated automatically so workers can restore the registered agent.
    agent_reference: str = ""

    @model_validator(mode="after")
    def bind_agent_reference(self) -> Self:
        expected = serialize_agent_reference(self.agent, type(self).__module__)
        if self.agent_reference and self.agent_reference != expected:
            raise ValueError(f"request agent must be {expected!r}")
        self.agent_reference = expected
        return self


def request_type_reference(request_type: type[AIRequestBase[Any]]) -> str:
    """Return the importable ``module:QualName`` of a request class."""
    return f"{request_type.__module__}:{request_type.__qualname__}"


def load_request(reference: str, value: object) -> AIRequestBase[Any]:
    """Validate ``value`` as the request class that ``reference`` names."""
    module_name, _, qualname = reference.partition(":")
    request_type: object = import_module(module_name)
    for name in qualname.split("."):
        request_type = getattr(request_type, name)
    if not isinstance(request_type, type) or not issubclass(request_type, AIRequestBase):
        raise TypeError(f"{reference!r} is not an AIRequestBase subclass")
    return request_type.model_validate(value)
