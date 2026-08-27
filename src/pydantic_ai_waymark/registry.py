import inspect
from importlib import import_module
from typing import TypeVar

from pydantic_ai.agent import AbstractAgent

from .types import RegisteredAgent

AgentT = TypeVar("AgentT", bound=RegisteredAgent)

_agents: dict[str, RegisteredAgent] = {}


def waymark_agent(
    agent: AgentT,
    *,
    name: str | None = None,
) -> AgentT:
    """Register a module-level Pydantic AI agent for Waymark workers."""
    frame = inspect.currentframe()
    caller = frame.f_back if frame is not None else None
    if caller is None or caller.f_code.co_name != "<module>":
        raise RuntimeError("waymark_agent() must be called at module level")

    module_name = caller.f_globals["__name__"]
    agent_name = name or agent.name
    if not isinstance(module_name, str):
        raise RuntimeError("caller module has no valid __name__")
    if not agent_name:
        raise ValueError("the agent needs a name, or pass name= to waymark_agent()")

    _agents[f"{module_name}:{agent_name}"] = agent
    return agent


def registered_agent(reference: str) -> RegisteredAgent:
    if agent := _agents.get(reference):
        return agent
    module_name, separator, variable_name = reference.rpartition(":")
    if separator:
        value = getattr(import_module(module_name), variable_name, None)
        if isinstance(value, AbstractAgent):
            _agents[reference] = value
            return value
    matches = [agent for key, agent in _agents.items() if key.endswith(f":{reference}")]
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise ValueError(f"agent name {reference!r} is ambiguous; use 'module:name'")
    raise KeyError(f"agent {reference!r} is not registered")


def agent_reference(agent: RegisteredAgent, module_name: str) -> str:
    """Return the importable module variable that holds an agent."""
    module = import_module(module_name)
    names = sorted(
        name
        for name, value in vars(module).items()
        if value is agent and not name.startswith("_")
    )
    if not names:
        raise ValueError(f"agent must be assigned to a public module variable in {module_name!r}")
    variable_name = agent.name if agent.name in names else names[0]
    reference = f"{module_name}:{variable_name}"
    _agents[reference] = agent
    return reference
