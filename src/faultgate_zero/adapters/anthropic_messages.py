"""Anthropic messages (``client.messages.create`` and ``client.beta.messages.create``, sync and async)."""
from __future__ import annotations

import json
from typing import Any

from ..recorder import ModelCall, Run
from ..trace import payload_text
from ._common import as_dict, drop_none, text_part


def start(args: tuple, kwargs: dict[str, Any]) -> ModelCall:
    msgs = kwargs.get("messages")
    return ModelCall(
        adapter="anthropic_messages", provider="anthropic", api="messages", model=str(kwargs.get("model") or ""),
        request={"messages": list(msgs) if isinstance(msgs, (list, tuple)) else [], "system": kwargs.get("system")},
        streamed=bool(kwargs.get("stream")),
    )


def finish(call: ModelCall, result: Any, run: Run) -> Any:
    call.response = result
    return result


def tool_calls(call: ModelCall) -> list[tuple[str, str]]:
    d = as_dict(call.response)
    return [(str(b.get("id")), str(b.get("name"))) for b in map(as_dict, d.get("content") or [])
            if b.get("type") == "tool_use"]


def normalise(call: ModelCall) -> dict[str, Any]:
    inp: list[dict[str, Any]] = []
    system = call.request.get("system")
    if system:
        inp.append({"role": "system", "parts": _blocks(system)})
    inp += [_message(m) for m in call.request.get("messages", [])]
    attrs: dict[str, Any] = {"gen_ai.input.messages": inp}
    d = as_dict(call.response) if call.response is not None else {}
    if "content" in d:
        attrs["gen_ai.output.messages"] = [{"role": "assistant", "parts": _blocks(d.get("content"))}]
        if d.get("stop_reason"):
            attrs["gen_ai.response.finish_reasons"] = [str(d["stop_reason"])]
        attrs["gen_ai.response.model"] = d.get("model")
        attrs["gen_ai.response.id"] = d.get("id")
        usage = d.get("usage") or {}
        attrs["gen_ai.usage.input_tokens"] = usage.get("input_tokens")
        attrs["gen_ai.usage.output_tokens"] = usage.get("output_tokens")
    return drop_none(attrs)


def _message(m: Any) -> dict[str, Any]:
    d = as_dict(m)
    return {"role": str(d.get("role") or "user"), "parts": _blocks(d.get("content"))}


def _blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [text_part(content)] if content else []
    parts: list[dict[str, Any]] = []
    for b in content or []:
        b = as_dict(b)
        t = str(b.get("type") or "")
        if t == "text":
            parts.append(text_part(b.get("text", "")))
        elif t in ("tool_use", "server_tool_use", "mcp_tool_use"):
            args = b.get("input")
            parts.append({"type": "tool_call", "id": b.get("id"), "name": b.get("name"),
                          "arguments": args if isinstance(args, str) else json.dumps(args, default=str)})
        elif t == "tool_result" or t.endswith("_tool_result"):
            res = b.get("content")
            p = {"type": "tool_call_response", "id": b.get("tool_use_id"), "result": payload_text(res)}
            if b.get("is_error") is True or (isinstance(res, dict) and str(res.get("type", "")).endswith("_error")):
                p["is_error"] = True
            parts.append(p)
        elif t in ("thinking", "redacted_thinking"):
            parts.append({"type": "reasoning"})
        elif t:
            parts.append({"type": t})
    return parts
