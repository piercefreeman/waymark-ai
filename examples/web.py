from html import escape
from urllib.parse import parse_qs

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

from .support_agent import SupportRequest, SupportWorkflow

app = FastAPI(title="Pydantic AI + Waymark")


def page(*, prompt: str = "", answer: str = "", error: str = "") -> str:
    result = (
        f'<section aria-live="polite"><h2>Answer</h2><p>{escape(answer)}</p></section>'
        if answer
        else ""
    )
    failure = f'<p role="alert">{escape(error)}</p>' if error else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Pydantic AI + Waymark</title>
  <style>
    body {{ font: 16px/1.5 system-ui; max-width: 48rem; margin: 4rem auto; padding: 0 1rem; }}
    textarea, button {{ box-sizing: border-box; font: inherit; width: 100%; padding: .75rem; }}
    textarea {{ min-height: 9rem; margin: .5rem 0 1rem; }}
    button {{ cursor: pointer; }}
    section, [role=alert] {{ margin-top: 2rem; padding: 1rem; background: #f3f4f6; }}
  </style>
</head>
<body>
  <h1>Durable support agent</h1>
  <p>Submit a prompt. The agent can call a workflow-level durable sleep tool.</p>
  <form method="post" action="/run">
    <label for="prompt">Prompt</label>
    <textarea id="prompt" name="prompt" maxlength="10000" required>{escape(prompt)}</textarea>
    <button type="submit">Run agent</button>
  </form>
  {failure}
  {result}
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return page()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/run", response_class=HTMLResponse)
async def run_agent(request: Request) -> HTMLResponse:
    body = await request.body()
    if len(body) > 20_000:
        return HTMLResponse(page(error="Prompt is too long."), status_code=413)
    try:
        values = parse_qs(body.decode(), max_num_fields=1)
    except (UnicodeDecodeError, ValueError):
        return HTMLResponse(page(error="Invalid form data."), status_code=400)
    prompt = values.get("prompt", [""])[0].strip()
    if not prompt:
        return HTMLResponse(page(error="Prompt is required."), status_code=422)

    try:
        reply = await SupportWorkflow().run(SupportRequest(prompt=prompt))
    except Exception as error:
        return HTMLResponse(page(prompt=prompt, error=str(error)), status_code=500)
    return HTMLResponse(page(prompt=prompt, answer=reply.answer))
