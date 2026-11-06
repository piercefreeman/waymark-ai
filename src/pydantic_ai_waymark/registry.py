import inspect
from importlib import import_module
from typing import TypeVar

from mountaineer_di import strip_depends_from_signature
from pydantic_ai.agent import AbstractAgent

from .types import (
    MessageReader,
    MessageWriter,
    PayloadDeserializer,
    PayloadSerializer,
    RegisteredAgent,
)

AgentT = TypeVar("AgentT", bound=RegisteredAgent)

_agents: dict[str, RegisteredAgent] = {}
_agent_codecs: dict[
    int,
    tuple[PayloadSerializer, str, PayloadDeserializer, str],
] = {}
_message_readers: dict[int, tuple[MessageReader, str]] = {}
_message_writers: dict[int, tuple[MessageWriter, str]] = {}


def _payload_parameter(
    codec: PayloadSerializer | PayloadDeserializer | MessageReader | MessageWriter,
    name: str,
) -> str:
    parameters = list(strip_depends_from_signature(codec).parameters.values())
    if len(parameters) != 1 or parameters[0].kind not in {
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    }:
        raise TypeError(
            f"{name} must accept one positional argument; other arguments must use Depends(...)"
        )
    return parameters[0].name


def waymark_agent(
    agent: AgentT,
    *,
    name: str | None = None,
    serializer: PayloadSerializer | None = None,
    deserializer: PayloadDeserializer | None = None,
    message_reader: MessageReader | None = None,
    message_writer: MessageWriter | None = None,
) -> AgentT:
    """Register a module-level Pydantic AI agent, its payload codecs, and its message reader.

    ``message_reader`` is called with a ``MessageQuery`` each time a tool waiting through
    ``WaitForMessages`` polls its mailbox. ``message_writer`` is called with an
    ``OutgoingMessage`` when a subagent this agent started with ``DelegateAgent(...,
    wait=False)`` reports back. Like the codecs, both can be sync or async and can declare
    ``Depends(...)`` parameters.

    NOTE: The message hooks are a polling workaround until Waymark has native message
    queues, which are likely the long-term way to pass messages between workflows.
    """
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
    if (serializer is None) != (deserializer is None):
        raise ValueError("serializer and deserializer must be provided together")

    codecs = None
    if serializer is not None and deserializer is not None:
        codecs = (
            serializer,
            _payload_parameter(serializer, "serializer"),
            deserializer,
            _payload_parameter(deserializer, "deserializer"),
        )
    reader = (
        (message_reader, _payload_parameter(message_reader, "message_reader"))
        if message_reader is not None
        else None
    )
    writer = (
        (message_writer, _payload_parameter(message_writer, "message_writer"))
        if message_writer is not None
        else None
    )
    _agents[f"{module_name}:{agent_name}"] = agent
    if codecs is None:
        _agent_codecs.pop(id(agent), None)
    else:
        _agent_codecs[id(agent)] = codecs
    if reader is None:
        _message_readers.pop(id(agent), None)
    else:
        _message_readers[id(agent)] = reader
    if writer is None:
        _message_writers.pop(id(agent), None)
    else:
        _message_writers[id(agent)] = writer
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


def registered_codecs(
    reference: str,
) -> tuple[PayloadSerializer, str, PayloadDeserializer, str] | None:
    return _agent_codecs.get(id(registered_agent(reference)))


def registered_message_reader(reference: str) -> tuple[MessageReader, str] | None:
    return _message_readers.get(id(registered_agent(reference)))


def registered_message_writer(reference: str) -> tuple[MessageWriter, str] | None:
    return _message_writers.get(id(registered_agent(reference)))


def agent_reference(agent: RegisteredAgent, module_name: str) -> str:
    """Return the importable module variable that holds an agent."""
    module = import_module(module_name)
    names = sorted(
        name for name, value in vars(module).items() if value is agent and not name.startswith("_")
    )
    if not names:
        raise ValueError(f"agent must be assigned to a public module variable in {module_name!r}")
    variable_name = agent.name if agent.name in names else names[0]
    reference = f"{module_name}:{variable_name}"
    _agents[reference] = agent
    return reference
