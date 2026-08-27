from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, computed_field
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

    @computed_field
    @property
    def agent_reference(self) -> str:
        return serialize_agent_reference(self.agent, type(self).__module__)

    def to_json(self) -> str:
        return self.model_dump_json()
