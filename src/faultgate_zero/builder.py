"""Turn a finished Run into a Trace with GenAI semantic-convention attributes."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from . import adapters
from .adapters._common import drop_none, text_part
from .recorder import Run, ToolCall
from .trace import AGENT, CHAT, TOOL, Span, Trace, answer_from_calls, messages


def span_id(run_id: str, label: str) -> str:
    return hashlib.sha256(f"{run_id}/{label}".encode()).hexdigest()[:16]


def _short(exc: BaseException | None, n: int = 500) -> str:
    return f"{type(exc).__name__}: {exc}"[:n] if exc is not None else ""


def _json(v: Any) -> str:
    if isinstance(v, str):
        return v
    try:
        return json.dumps(v, ensure_ascii=False, default=str, sort_keys=True)
    except (TypeError, ValueError):
        return str(v)


def build_trace(run: Run) -> Trace:
    root_id = span_id(run.id, "root")
    chats: list[Span] = []
    for i, call in enumerate(run.model_calls):
        attrs: dict[str, Any] = {"gen_ai.operation.name": CHAT, "gen_ai.provider.name": call.provider,
                                 "gen_ai.request.model": call.model or None}
        try:
            attrs.update(adapters.get(call.adapter).normalise(call))
        except Exception as exc:
            run.mark_incomplete(f"{call.adapter}.normalise", exc)
        if call.error is not None:
            attrs["error.type"] = type(call.error).__qualname__
            if run.escaped(call.error):
                attrs["faultgate.error_escaped"] = True
        chats.append(Span(span_id(run.id, f"chat-{i}"), root_id, f"chat {call.model}".strip(), call.start_ns,
                          call.end_ns, drop_none(attrs), error=call.error is not None,
                          status_message=_short(call.error)))

    tools = [_tool_span(run, root_id, j, t) for j, t in enumerate(run.tool_calls)]

    first_user = next((m for s in chats for m in messages(s.attrs.get("gen_ai.input.messages"))
                       if m.get("role") == "user"), None)
    answer = answer_from_calls(chats)
    root_attrs = drop_none({
        "gen_ai.operation.name": AGENT,
        "gen_ai.agent.name": run.agent,
        "faultgate.scenario": run.name,
        "faultgate.test": run.test,
        "enduser.id": run.user_id,
        "gen_ai.input.messages": [first_user] if first_user else None,
        "gen_ai.output.messages": [{"role": "assistant", "parts": [text_part(answer)]}] if answer else [],
        "faultgate.termination": run.termination,
        "faultgate.capture_complete": run.capture_complete,
    })
    if run.capture_errors:
        root_attrs["faultgate.capture_errors"] = list(run.capture_errors)
    root = Span(root_id, None, f"{AGENT} {run.agent}", run.start_ns, run.end_ns, root_attrs,
                error=run.error is not None, status_message=_short(run.error))
    spans = sorted([root, *chats, *tools], key=lambda s: s.start_ns)
    return Trace(trace_id=run.id, scenario=run.name, agent=run.agent, user_id=run.user_id, spans=spans)


def _tool_span(run: Run, root_id: str, j: int, t: ToolCall) -> Span:
    attrs = drop_none({
        "gen_ai.operation.name": TOOL,
        "gen_ai.tool.name": t.name,
        "gen_ai.tool.call.id": t.call_id,
        "gen_ai.tool.call.arguments": _json(t.arguments),
        "gen_ai.tool.call.result": None if t.error is not None else _json(t.result),
        "error.type": type(t.error).__qualname__ if t.error is not None else None,
    })
    if t.error is not None:
        attrs["gen_ai.tool.call.result"] = f"Error: {_short(t.error, 300)}"
        if run.escaped(t.error):
            attrs["faultgate.error_escaped"] = True
    return Span(span_id(run.id, f"tool-{j}"), root_id, f"{TOOL} {t.name}", t.start_ns, t.end_ns, attrs,
                error=t.error is not None, status_message=_short(t.error))
