"""Real Anthropic SDK clients over httpx2.MockTransport."""
from __future__ import annotations

import json
from typing import Any

import anthropic
import httpx2

from fakes.openai_fakes import Script


def message(text: str | None = None, tool_uses: list[tuple[str, str, Any]] | None = None,
            stop: str | None = None, model: str = "claude-test", usage: tuple[int, int] = (12, 6)) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if text:
        content.append({"type": "text", "text": text})
    for tid, name, inp in tool_uses or []:
        content.append({"type": "tool_use", "id": tid, "name": name, "input": inp})
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": model, "content": content,
            "stop_reason": stop or ("tool_use" if tool_uses else "end_turn"), "stop_sequence": None,
            "usage": {"input_tokens": usage[0], "output_tokens": usage[1]}}


def client(script: Script) -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key="sk-ant-test", base_url="http://fake.test", max_retries=0,
                               http_client=httpx2.Client(transport=httpx2.MockTransport(script.handler)))


def async_client(script: Script) -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(api_key="sk-ant-test", base_url="http://fake.test", max_retries=0,
                                    http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(script.handler)))


def agent(c: Any, prompt: str, tools: dict[str, Any], max_steps: int = 12) -> str | None:
    """An Anthropic tool loop: tool errors go back as is_error tool_result blocks."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    for _ in range(max_steps):
        r = c.messages.create(model="claude-test", max_tokens=512, system="You are a support agent.",
                              messages=messages, tools=[{"name": n, "input_schema": {"type": "object"}} for n in tools])
        messages.append({"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in r.content]})
        uses = [b for b in r.content if b.type == "tool_use"]
        if not uses:
            return "".join(b.text for b in r.content if b.type == "text")
        results = []
        for u in uses:
            try:
                results.append({"type": "tool_result", "tool_use_id": u.id, "content": json.dumps(tools[u.name](**u.input))})
            except Exception as exc:
                results.append({"type": "tool_result", "tool_use_id": u.id, "content": str(exc), "is_error": True})
        messages.append({"role": "user", "content": results})
    return None
