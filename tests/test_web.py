from fastapi.testclient import TestClient
from pydantic_ai import ModelMessagesTypeAdapter
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart

from examples import web
from examples.support_agent import (
    ParallelSupportWorkflow,
    SleepRequest,
    SleepWorkflow,
    SupportReply,
    SupportRequest,
)
from pydantic_ai_waymark import AgentResult


def test_form_runs_workflow(monkeypatch) -> None:
    history = ModelMessagesTypeAdapter.dump_json(
        [
            ModelResponse(
                parts=[
                    ToolCallPart(
                        "lookup_support_policy", {"topic": "duplicate charge"}, "call-1"
                    ),
                    ToolCallPart("final_result", {"answer": "done"}, "call-2"),
                ]
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        "lookup_support_policy", "Escalate account access.", "call-1"
                    )
                ]
            ),
        ]
    ).decode()

    async def run(_self, request: SupportRequest) -> AgentResult:
        assert request.prompt == "Why was I charged twice?"
        return AgentResult.model_validate(
            {
                "output": SupportReply(
                    answer="I found the duplicate charge.", needs_human=False
                ),
                "message_history": history,
                "new_messages": history,
                "usage": {
                    "input_tokens": 0,
                    "cache_write_tokens": 0,
                    "cache_read_tokens": 0,
                    "output_tokens": 0,
                    "input_audio_tokens": 0,
                    "cache_audio_read_tokens": 0,
                    "output_audio_tokens": 0,
                    "details": {},
                    "requests": 2,
                    "tool_calls": 1,
                },
                "run_id": "run-1",
                "conversation_id": "conversation-1",
            }
        )

    monkeypatch.setattr(web.SupportWorkflow, "run", run)
    response = TestClient(web.app).post(
        "/run",
        content="prompt=Why+was+I+charged+twice%3F",
        headers={"content-type": "application/x-www-form-urlencoded"},
    )

    assert response.status_code == 200
    assert "I found the duplicate charge." in response.text
    assert "lookup_support_policy" in response.text
    assert 'topic&quot;:&quot;duplicate charge' in response.text
    assert "Escalate account access." in response.text
    assert "final_result" not in response.text
    assert TestClient(web.app).post("/run", content="prompt=+").status_code == 422


def test_sleep_form_passes_duration_through_dependencies(monkeypatch) -> None:
    async def run(_self, request: SleepRequest) -> str:
        assert request.deps is not None
        assert request.deps.seconds == 1.5
        return "Processing complete."

    monkeypatch.setattr(web.SleepWorkflow, "run", run)
    client = TestClient(web.app)
    response = client.post(
        "/sleep",
        content="seconds=1.5",
        headers={"content-type": "application/x-www-form-urlencoded"},
    )

    assert response.status_code == 200
    assert "Processing complete." in response.text
    assert client.post("/sleep", content="seconds=61").status_code == 422


def test_parallel_agent_example_compiles_to_parallel_branches() -> None:
    ir_text = str(ParallelSupportWorkflow.workflow_ir())

    assert "parallel_expr" in ir_text
    assert ir_text.count('name: "run_agent"') >= 2
    assert "combine_parallel_support_results" in ir_text


def test_sleep_example_compiles() -> None:
    assert 'name: "run_agent"' in str(SleepWorkflow.workflow_ir())
