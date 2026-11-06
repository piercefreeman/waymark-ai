import asyncio
import json
import os
from typing import cast

import pytest
from fastapi.testclient import TestClient
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from examples import recurse_agent as example
from examples import web


@pytest.fixture
def scripted_model(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace the example's OpenAI model with a script; returns the prompts it served."""
    prompts: list[str] = []
    monkeypatch.setattr(example.recurse_agent, "model", FunctionModel(script(prompts)))
    return prompts


SUBQUESTIONS = {"root": ["left", "right"], "left": ["left deeper"], "right": []}


def first_prompt(messages: list[ModelMessage]) -> str:
    for part in messages[0].parts:
        if isinstance(part, UserPromptPart) and isinstance(part.content, str):
            return part.content
    raise AssertionError("expected a user prompt")


def script(prompts: list[str]):
    """Delegate ``SUBQUESTIONS[prompt]`` in one turn, then combine whatever comes back."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompt = first_prompt(messages)
        last = messages[-1]
        if len(messages) == 1:
            prompts.append(prompt)
            delegated = SUBQUESTIONS.get(prompt, [])
            if delegated:
                return ModelResponse(
                    parts=[
                        ToolCallPart("delegate_subquestion", {"question": question})
                        for question in delegated
                    ]
                )
        assert isinstance(last, ModelRequest)
        answers = [
            str(cast(dict[str, object], part.content)["answer"])
            for part in last.parts
            if isinstance(part, ToolReturnPart) and isinstance(part.content, dict)
        ]
        # NOTE: Pydantic AI releases carry `ModelRetry` in different part types, so match
        # the example's retry message instead of a part class.
        refused = any(
            "Maximum depth reached" in str(getattr(part, "content", "")) for part in last.parts
        )
        answer = f"{prompt}: " + (" + ".join(answers) if answers else "direct")
        if refused:
            answer += " (depth limit)"
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "final_result",
                    {"answer": answer, "subquestions": SUBQUESTIONS.get(prompt, [])},
                )
            ]
        )

    return respond


def test_recurse_agent_delegates_to_itself_until_the_depth_limit(scripted_model: list[str]) -> None:
    answer = asyncio.run(
        example.RecurseWorkflow().run(
            example.RecurseRequest(
                prompt="root",
                deps=example.RecurseDependencies(max_depth=1),
            )
        )
    )

    # "left" runs one level down, so its own delegation is refused with a ModelRetry.
    assert answer.answer == "root: left: direct (depth limit) + right: direct"
    assert answer.subquestions == ["left", "right"]
    assert sorted(scripted_model) == ["left", "right", "root"]


def test_recurse_agent_answers_directly_at_depth_zero(scripted_model: list[str]) -> None:
    answer = asyncio.run(
        example.RecurseWorkflow().run(
            example.RecurseRequest(prompt="root", deps=example.RecurseDependencies(max_depth=0))
        )
    )

    assert answer.answer == "root: direct (depth limit)"
    assert scripted_model == ["root"]


def restating_script(tools_seen: dict[int, list[str]]):
    """Delegate a rephrasing of the question, then answer with whatever came back."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tools_seen.setdefault(len(messages), [tool.name for tool in info.function_tools])
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "delegate_subquestion",
                        {"question": "What is a safe walking route from Novato to San Rafael?"},
                    )
                ]
            )
        feedback = " ".join(str(getattr(part, "content", "")) for part in messages[-1].parts)
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "final_result",
                    {
                        "answer": "rejected"
                        if "restates the whole question" in feedback
                        else "ran",
                        "subquestions": [],
                    },
                )
            ]
        )

    return respond


def test_recurse_agent_refuses_to_delegate_a_restated_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tools_seen: dict[int, list[str]] = {}
    monkeypatch.setattr(example.recurse_agent, "model", FunctionModel(restating_script(tools_seen)))

    answer = asyncio.run(
        example.RecurseWorkflow().run(
            example.RecurseRequest(prompt="How do I safely walk from Novato to San Rafael?")
        )
    )

    assert answer.answer == "rejected"
    assert "start_fact_check" in tools_seen[1]


def test_fact_check_tools_are_only_offered_at_the_top_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tools_by_prompt: dict[str, list[str]] = {}
    respond = script([])

    def recording(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tools_by_prompt.setdefault(
            first_prompt(messages), [tool.name for tool in info.function_tools]
        )
        return respond(messages, info)

    monkeypatch.setattr(example.recurse_agent, "model", FunctionModel(recording))
    asyncio.run(example.RecurseWorkflow().run(example.RecurseRequest(prompt="root")))

    assert {"start_fact_check", "check_fact_check"} <= set(tools_by_prompt["root"])
    assert "start_fact_check" not in tools_by_prompt["left"]
    assert "delegate_subquestion" in tools_by_prompt["left"]


def test_restatement_check_compares_significant_words() -> None:
    question = "How do I safely walk from Novato to Saint Mark's School in San Rafael?"

    assert example.restates(
        "Provide a safe walking route from Novato to Saint Mark's School in San Rafael", question
    )
    assert not example.restates("Which buses run from Novato to San Rafael?", question)
    # A narrower part shares most words but drops one end of the trip.
    assert not example.restates(
        "What is the safest walking route from San Rafael Transit Center to Saint Mark's School?",
        question,
    )


def test_recurse_example_compiles_its_subagent_paths() -> None:
    ir_text = str(example.RecurseWorkflow.workflow_ir())

    assert "pydantic_ai_agent_delegate_result" in ir_text
    assert "pydantic_ai_agent_start_detached_subagent" in ir_text
    assert "pydantic_ai_agent_read_messages" in ir_text


def test_recurse_form_passes_the_depth_through_dependencies(monkeypatch) -> None:
    async def run(_self, request: example.RecurseRequest) -> example.RecurseAnswer:
        assert request.prompt == "Why is the sky blue?"
        assert request.deps == example.RecurseDependencies(max_depth=1)
        return example.RecurseAnswer(answer="Rayleigh scattering.", subquestions=["What scatters?"])

    monkeypatch.setattr(web.RecurseWorkflow, "run", run)
    client = TestClient(web.app)
    response = client.post(
        "/recurse",
        content="prompt=Why+is+the+sky+blue%3F&max_depth=1",
        headers={"content-type": "application/x-www-form-urlencoded"},
    )

    assert response.status_code == 200
    assert "Rayleigh scattering." in response.text
    assert "What scatters?" in response.text
    assert client.post("/recurse", content="prompt=+&max_depth=1").status_code == 422
    assert client.post("/recurse", content="prompt=hi&max_depth=9").status_code == 422


def fact_checking_script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Start a fact check, read its report from the mailbox, then answer."""
    last = messages[-1]
    returned = [part for part in last.parts if isinstance(part, ToolReturnPart)]
    if len(messages) == 1:
        return ModelResponse(parts=[ToolCallPart("start_fact_check", {"claim": "water is wet"})])
    if returned[0].tool_name == "start_fact_check":
        handle = cast(dict[str, str], returned[0].content)
        return ModelResponse(
            parts=[ToolCallPart("check_fact_check", {"mailbox": handle["mailbox"]})]
        )
    [report_message] = cast(list[dict[str, str]], returned[0].content)
    report = json.loads(report_message["content"])
    return ModelResponse(
        parts=[
            ToolCallPart(
                "final_result",
                {
                    "answer": f"{report['status']}: {report['output']['verdict']}",
                    "subquestions": [],
                },
            )
        ]
    )


def checked_claim(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(
        parts=[
            ToolCallPart(
                "final_result",
                {
                    "claim": "water is wet",
                    "verdict": "plausible",
                    "reasoning": "It is.",
                    "verify_with": "A physics textbook.",
                },
            )
        ]
    )


@pytest.mark.skipif(
    os.environ.get("WAYMARK_AI_TEST_DATABASE_URL") is None,
    reason="set WAYMARK_AI_TEST_DATABASE_URL to run the example's mailbox flow",
)
def test_recurse_agent_reads_a_detached_fact_check_from_its_mailbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WAYMARK_DATABASE_URL", os.environ["WAYMARK_AI_TEST_DATABASE_URL"])
    monkeypatch.setattr(example.recurse_agent, "model", FunctionModel(fact_checking_script))
    monkeypatch.setattr(example.fact_check_agent, "model", FunctionModel(checked_claim))
    asyncio.run(example.create_mailbox_table())
    # Creating it twice must be a no-op, since every web process does it on startup.
    asyncio.run(example.create_mailbox_table())

    answer = asyncio.run(
        example.RecurseWorkflow().run(example.RecurseRequest(prompt="Is water wet?"))
    )

    assert answer.answer == "completed: plausible"
