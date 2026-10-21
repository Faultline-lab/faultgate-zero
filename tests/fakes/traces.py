"""Build traces directly, for check tests: one chat span per tool call, then a final chat span that
sends every seen result back and gives the answer."""
from __future__ import annotations

import json
from typing import Any

from faultgate_zero.trace import Span, Trace


def step(tool: str, args: Any = None, result: Any = "ok", error: bool = False) -> tuple:
    """``result=None`` means the result never reached the model (the run ended first)."""
    a = args if isinstance(args, str) else json.dumps(args or {})
    r = result if result is None or isinstance(result, str) else json.dumps(result)
    return tool, a, r, error


def chat(i: int, t: int, *, out: list | None = None, inp: list | None = None, error: str | None = None,
         finish: str | None = None) -> Span:
    attrs: dict[str, Any] = {"gen_ai.operation.name": "chat", "gen_ai.input.messages": inp or [],
                             "gen_ai.output.messages": [{"role": "assistant", "parts": out or []}]}
    if finish:
        attrs["gen_ai.response.finish_reasons"] = [finish]
    if error:
        attrs["error.type"] = error.split(":")[0]
        attrs.pop("gen_ai.output.messages")
    return Span(f"chat{i}", "root", "chat", t, t + 1, attrs, error=bool(error), status_message=error or "")


def trace(steps: list[tuple] = (), answer: str | None = "done", *, user_id: str | None = None,
          termination: str = "completed", chats: list[Span] | None = None, finish: str | None = None) -> Trace:
    spans = [Span("root", None, "invoke_agent", 0, 10_000, {"gen_ai.operation.name": "invoke_agent",
                                                         "faultgate.termination": termination})]
    responses = []
    t = 10
    for i, (tool, args, result, err) in enumerate(steps):
        spans.append(chat(i, t, out=[{"type": "tool_call", "id": f"call{i}", "name": tool, "arguments": args}]))
        if result is not None:
            p = {"type": "tool_call_response", "id": f"call{i}", "result": result}
            if err:
                p["is_error"] = True
            responses.append(p)
        t += 10
    spans.extend(chats or [])
    t += 100
    out = [{"type": "text", "content": answer}] if answer else []
    spans.append(chat(len(steps), t, out=out, inp=[{"role": "tool", "parts": responses}], finish=finish))
    return Trace("t1", "scenario", "agent", user_id, spans)
