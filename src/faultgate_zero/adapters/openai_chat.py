"""OpenAI chat completions (``client.chat.completions.create``, sync and async)."""
from __future__ import annotations

from typing import Any

from ..recorder import ModelCall, Run
from ..trace import payload_text
from ._common import as_dict, drop_none, text_part


def start(args: tuple, kwargs: dict[str, Any]) -> ModelCall:
    msgs = kwargs.get("messages")
    return ModelCall(
        adapter="openai_chat", provider="openai", api="chat", model=str(kwargs.get("model") or ""),
        request={"messages": list(msgs) if isinstance(msgs, (list, tuple)) else []},
        streamed=bool(kwargs.get("stream")),
    )


def finish(call: ModelCall, result: Any, run: Run) -> Any:
    call.response = result
    if call.streamed and not hasattr(result, "http_response"):
        from ..streams import ChatChunks, attach

        acc = ChatChunks()
        return attach(result, run, call, acc.feed, acc.result)
    return result


def tool_calls(call: ModelCall) -> list[tuple[str, str]]:
    """(call id, tool name) for each tool call in this response, in order."""
    choices = as_dict(call.response).get("choices") or []
    msg = as_dict(choices[0].get("message")) if choices else {}
    out = []
    for tc in msg.get("tool_calls") or []:
        tc = as_dict(tc)
        fn = as_dict(tc.get("function") or tc.get("custom"))
        out.append((str(tc.get("id")), str(fn.get("name"))))
    return out


def normalise(call: ModelCall) -> dict[str, Any]:
    attrs: dict[str, Any] = {"gen_ai.input.messages": [_message(m) for m in call.request.get("messages", [])]}
    d = as_dict(call.response) if call.response is not None else {}
    choices = d.get("choices")
    if isinstance(choices, list):
        attrs["gen_ai.response.finish_reasons"] = [str(c["finish_reason"]) for c in choices if c.get("finish_reason")]
        if choices:
            out = _message({**(choices[0].get("message") or {}), "role": "assistant"})
            attrs["gen_ai.output.messages"] = [out]
        attrs["gen_ai.response.model"] = d.get("model")
        attrs["gen_ai.response.id"] = d.get("id")
        usage = d.get("usage") or {}
        attrs["gen_ai.usage.input_tokens"] = usage.get("prompt_tokens")
        attrs["gen_ai.usage.output_tokens"] = usage.get("completion_tokens")
    return drop_none(attrs)


def _message(m: Any) -> dict[str, Any]:
    d = as_dict(m)
    role = str(d.get("role") or "user")
    if role == "tool":
        return {"role": "tool", "parts": [{"type": "tool_call_response", "id": d.get("tool_call_id"),
                                           "result": payload_text(d.get("content"))}]}
    parts: list[dict[str, Any]] = []
    content = d.get("content")
    if isinstance(content, str):
        if content:
            parts.append(text_part(content))
    elif isinstance(content, list):
        for p in content:
            p = as_dict(p)
            if p.get("type") == "text":
                parts.append(text_part(p.get("text", p.get("content", ""))))
            elif p.get("type"):
                parts.append({"type": str(p["type"])})
    if d.get("refusal"):
        parts.append(text_part(d["refusal"]))
    for tc in d.get("tool_calls") or []:
        tc = as_dict(tc)
        if tc.get("type") == "custom" or "custom" in tc:
            fn = as_dict(tc.get("custom"))
            parts.append({"type": "tool_call", "id": tc.get("id"), "name": fn.get("name"), "arguments": fn.get("input")})
        else:
            fn = as_dict(tc.get("function"))
            parts.append({"type": "tool_call", "id": tc.get("id"), "name": fn.get("name"), "arguments": fn.get("arguments")})
    return {"role": role, "parts": parts}
