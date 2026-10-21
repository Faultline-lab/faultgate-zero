"""Tiny agent loops of the kind teams write, driven by the fakes."""
from __future__ import annotations

import json
from typing import Any, Callable

SCHEMAS = [{"type": "function", "function": {"name": n, "parameters": {"type": "object"}}}
           for n in ("search", "lookup_order", "issue_refund", "verify_identity")]


def openai_agent(client: Any, prompt: str, tools: dict[str, Callable[..., Any]], max_steps: int = 12,
                 model: str = "gpt-test") -> str | None:
    messages: list[Any] = [{"role": "system", "content": "You are a support agent."},
                           {"role": "user", "content": prompt}]
    for _ in range(max_steps):
        r = client.chat.completions.create(model=model, messages=messages, tools=SCHEMAS)
        msg = r.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls:
            return msg.content
        for tc in msg.tool_calls:
            try:
                content = json.dumps(tools[tc.function.name](**json.loads(tc.function.arguments)))
            except Exception as exc:
                content = f"Error: {exc}"
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
    return None


async def async_openai_agent(client: Any, prompt: str, tools: dict[str, Callable[..., Any]], max_steps: int = 12,
                             model: str = "gpt-test") -> str | None:
    messages: list[Any] = [{"role": "user", "content": prompt}]
    for _ in range(max_steps):
        r = await client.chat.completions.create(model=model, messages=messages, tools=SCHEMAS)
        msg = r.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls:
            return msg.content
        for tc in msg.tool_calls:
            content = json.dumps(tools[tc.function.name](**json.loads(tc.function.arguments)))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
    return None
