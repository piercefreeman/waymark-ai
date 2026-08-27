from fastapi.testclient import TestClient

from examples import web
from examples.support_agent import SupportReply, SupportRequest


def test_form_runs_workflow(monkeypatch) -> None:
    async def run(_self, request: SupportRequest) -> SupportReply:
        assert request.prompt == "Why was I charged twice?"
        return SupportReply(answer="I found the duplicate charge.", needs_human=False)

    monkeypatch.setattr(web.SupportWorkflow, "run", run)
    response = TestClient(web.app).post(
        "/run",
        content="prompt=Why+was+I+charged+twice%3F",
        headers={"content-type": "application/x-www-form-urlencoded"},
    )

    assert response.status_code == 200
    assert "I found the duplicate charge." in response.text
    assert TestClient(web.app).post("/run", content="prompt=+").status_code == 422
