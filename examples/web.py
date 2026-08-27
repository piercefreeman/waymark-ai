from html import escape
from urllib.parse import parse_qs

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import ValidationError
from pydantic_ai import ModelMessagesTypeAdapter
from pydantic_ai.messages import ToolCallPart, ToolReturnPart

from .support_agent import (
    SleepDependencies,
    SleepRequest,
    SleepWorkflow,
    SupportReply,
    SupportRequest,
    SupportWorkflow,
)

app = FastAPI(title="Pydantic AI + Waymark")


def support_tool_calls(message_history: str) -> tuple[tuple[str, str, str], ...]:
    messages = ModelMessagesTypeAdapter.validate_json(message_history)
    results = {
        part.tool_call_id: str(part.content)
        for message in messages
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    }
    return tuple(
        (part.tool_name, part.args_as_json_str(), results.get(part.tool_call_id, ""))
        for message in messages
        for part in message.parts
        if isinstance(part, ToolCallPart) and part.tool_name != "final_result"
    )


def page(
    *,
    prompt: str = "",
    answer: str = "",
    tool_calls: tuple[tuple[str, str, str], ...] = (),
    error: str = "",
    sleep_seconds: str = "5",
    sleep_answer: str = "",
    sleep_error: str = "",
) -> str:
    tool_items = "".join(
        f"<li><strong>{escape(name)}</strong>"
        f"<pre>Arguments: {escape(arguments)}</pre>"
        f"<pre>Result: {escape(result)}</pre></li>"
        for name, arguments, result in tool_calls
    )
    tool_history = (
        f"<details open><summary>Tool calls ({len(tool_calls)})</summary>"
        f"<ol>{tool_items}</ol></details>"
        if tool_calls
        else ""
    )
    support_result = (
        f'<section aria-live="polite"><h2>Answer</h2><p>{escape(answer)}</p>'
        f"{tool_history}</section>"
        if answer
        else ""
    )
    support_failure = f'<p role="alert">{escape(error)}</p>' if error else ""
    sleep_result = (
        f'<section aria-live="polite"><h2>Result</h2><p>{escape(sleep_answer)}</p></section>'
        if sleep_answer
        else ""
    )
    sleep_failure = f'<p role="alert">{escape(sleep_error)}</p>' if sleep_error else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Pydantic AI + Waymark</title>
  <style>
    body {{ font: 16px/1.5 system-ui; max-width: 48rem; margin: 4rem auto; padding: 0 1rem; }}
    textarea, input, button {{
      box-sizing: border-box; font: inherit; width: 100%; padding: .75rem;
    }}
    textarea {{ min-height: 9rem; margin: .5rem 0 1rem; }}
    input {{ margin: .5rem 0 1rem; }}
    button {{ cursor: pointer; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; }}
    main > section {{ margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #d1d5db; }}
    section[aria-live], [role=alert] {{ margin-top: 2rem; padding: 1rem; background: #f3f4f6; }}
  </style>
</head>
<body>
  <h1>Durable agent workflows</h1>
  <main>
    <section>
      <h2>Support agent</h2>
      <form method="post" action="/run">
        <label for="prompt">Prompt</label>
        <textarea id="prompt" name="prompt" maxlength="10000" required>{escape(prompt)}</textarea>
        <button type="submit">Run support agent</button>
      </form>
      {support_failure}
      {support_result}
    </section>
    <section>
      <h2>Durable sleep</h2>
      <p>Simulate processing with a dependency-backed durable timer.</p>
      <form method="post" action="/sleep">
        <label for="sleep-seconds">Seconds (0–60)</label>
        <input id="sleep-seconds" name="seconds" type="number" min="0" max="60"
               step="0.1" value="{escape(sleep_seconds)}" required>
        <button type="submit">Simulate processing</button>
      </form>
      {sleep_failure}
      {sleep_result}
    </section>
  </main>
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
        result = await SupportWorkflow().run(SupportRequest(prompt=prompt))
        reply = SupportReply.model_validate(result.output)
        tool_calls = support_tool_calls(result.message_history)
    except Exception as error:
        return HTMLResponse(page(prompt=prompt, error=str(error)), status_code=500)
    return HTMLResponse(page(prompt=prompt, answer=reply.answer, tool_calls=tool_calls))


@app.post("/sleep", response_class=HTMLResponse)
async def run_sleep(request: Request) -> HTMLResponse:
    body = await request.body()
    if len(body) > 1_000:
        return HTMLResponse(page(sleep_error="Invalid duration."), status_code=413)
    try:
        values = parse_qs(body.decode(), max_num_fields=1)
    except (UnicodeDecodeError, ValueError):
        return HTMLResponse(page(sleep_error="Invalid form data."), status_code=400)
    seconds = values.get("seconds", [""])[0].strip()
    try:
        deps = SleepDependencies.model_validate({"seconds": seconds})
    except ValidationError:
        return HTMLResponse(
            page(sleep_seconds=seconds, sleep_error="Seconds must be between 0 and 60."),
            status_code=422,
        )

    try:
        reply = await SleepWorkflow().run(
            SleepRequest(prompt="Simulate processing.", deps=deps)
        )
    except Exception as error:
        return HTMLResponse(
            page(sleep_seconds=seconds, sleep_error=str(error)), status_code=500
        )
    return HTMLResponse(page(sleep_seconds=seconds, sleep_answer=reply))
