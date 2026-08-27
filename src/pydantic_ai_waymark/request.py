from typing import Any, ClassVar, Generic, Self, TypeVar

from pydantic import BaseModel, ConfigDict, model_validator
from pydantic_ai import Agent

from .registry import agent_reference as serialize_agent_reference

AgentDependenciesT = TypeVar("AgentDependenciesT")


class AIRequestBase(BaseModel, Generic[AgentDependenciesT]):
    """Serializable input for a durable Pydantic AI workflow."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    agent: ClassVar[Agent[Any, Any]]
    prompt: str | None
    message_history: str | None = None
    deps: AgentDependenciesT | None = None
    model: str | None = None
    conversation_id: str | None = None
    run_id: str | None = None
    agent_reference: str = ""

    @model_validator(mode="after")
    def bind_agent_reference(self) -> Self:
        expected = serialize_agent_reference(self.agent, type(self).__module__)
        if self.agent_reference and self.agent_reference != expected:
            raise ValueError(f"request agent must be {expected!r}")
        self.agent_reference = expected
        return self

    def to_json(self) -> str:
        return self.model_dump_json()
