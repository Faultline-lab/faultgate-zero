"""OpenAI Responses API (``client.responses.create`` / ``parse`` / ``stream``, sync and async).

The OpenAI Agents SDK uses this API by default. Tool calls are ``function_call`` output items; their
results come back as ``function_call_output`` input items, matched by ``call_id``.
"""
from __future__ import annotations

from typing import Any

from ..recorder import ModelCall, Run
from ..trace import payload_text
from ._common import as_dict, drop_none, text_part

TEXT_IN = ("input_text", "text", "output_text")


def start(args: tuple, kwargs: dict[str, Any]) -> ModelCall:
    inp = kwargs.get("input")
    return ModelCall(
        adapter="openai_responses", provider="openai", api="responses", model=str(kwargs.get("model") or ""),
        request={"input": list(inp) if isinstance(inp, (list, tuple)) else inp,
                 "instructions": kwargs.get("instructions")},
        streamed=bool(kwargs.get("stream")),
    )


def finish(call: ModelCall, result: Any, run: Run) -> Any:
    call.response = result
    if call.streamed and not hasattr(result, "http_response"):
        from ..streams import ResponsesEvents, attach

        acc = ResponsesEvents()
        return attach(result, run, call, acc.feed, acc.result)
    return result


def tool_calls(call: ModelCall) -> list[tuple[str, str]]:
    out = []
    for item in map(as_dict, as_dict(call.response).get("output") or []):
        if item.get("type") in ("function_call", "custom_tool_call"):
            out.append((str(item.get("call_id")), str(item.get("name"))))
    return out


def normalise(call: ModelCall) -> dict[str, Any]:
    inp: list[dict[str, Any]] = []
    if call.request.get("instructions"):
        inp.append({"role": "system", "parts": [text_part(call.request["instructions"])]})
    raw = call.request.get("input")
    if isinstance(raw, str):
        inp.append({"role": "user", "parts": [text_part(raw)]})
    else:
        inp += [m for m in map(_input_item, raw or []) if m]
    attrs: dict[str, Any] = {"gen_ai.input.messages": inp}
    d = as_dict(call.response) if call.response is not None else {}
    if "output" in d:
        parts: list[dict[str, Any]] = []
        for item in map(as_dict, d.get("output") or []):
            parts += _output_parts(item)
        attrs["gen_ai.output.messages"] = [{"role": "assistant", "parts": parts}]
        status = d.get("status")
        reason = as_dict(d.get("incomplete_details")).get("reason")
        attrs["gen_ai.response.finish_reasons"] = [str(reason or status)] if (reason or status) else None
        attrs["gen_ai.response.model"] = d.get("model")
        attrs["gen_ai.response.id"] = d.get("id")
        usage = as_dict(d.get("usage"))
        attrs["gen_ai.usage.input_tokens"] = usage.get("input_tokens")
        attrs["gen_ai.usage.output_tokens"] = usage.get("output_tokens")
    return drop_none(attrs)


def _content_parts(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [text_part(content)] if content else []
    parts = []
    for p in map(as_dict, content or []):
        if p.get("type") in TEXT_IN:
            parts.append(text_part(p.get("text", "")))
        elif p.get("type") == "refusal":
            parts.append(text_part(p.get("refusal", "")))
        elif p.get("type"):
            parts.append({"type": str(p["type"])})
    return parts


def _input_item(item: Any) -> dict[str, Any] | None:
    d = as_dict(item)
    t = d.get("type")
    if t == "function_call_output" or t == "custom_tool_call_output":
        return {"role": "tool", "parts": [{"type": "tool_call_response", "id": d.get("call_id"),
                                           "result": payload_text(d.get("output"))}]}
    if t in ("function_call", "custom_tool_call"):
        return {"role": "assistant", "parts": _output_parts(d)}
    if "role" in d:
        return {"role": str(d["role"]), "parts": _content_parts(d.get("content"))}
    return {"role": "user", "parts": [{"type": str(t)}]} if t else None


def _output_parts(item: dict[str, Any]) -> list[dict[str, Any]]:
    t = item.get("type")
    if t == "message":
        return _content_parts(item.get("content"))
    if t == "function_call":
        return [{"type": "tool_call", "id": item.get("call_id"), "name": item.get("name"), "arguments": item.get("arguments")}]
    if t == "custom_tool_call":
        return [{"type": "tool_call", "id": item.get("call_id"), "name": item.get("name"), "arguments": item.get("input")}]
    if t == "reasoning":
        return [{"type": "reasoning"}]
    return [{"type": str(t)}] if t else []
