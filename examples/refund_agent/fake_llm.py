"""An OpenAI client whose model answers are scripted, so the example runs offline and gives the same
result every time. The real OpenAI SDK runs; only the HTTP transport is replaced."""
from __future__ import annotations

import json
from typing import Any

import openai

try:  # openai 3.x uses httpx2; older versions use httpx
    import httpx2 as httpx
except ImportError:  # pragma: no cover
    import httpx  # type: ignore[no-redef]


def say(text: str, finish: str = "stop") -> dict[str, Any]:
    return _completion({"role": "assistant", "content": text}, finish)


def call(*calls: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    """The model asks for tools: ``call(("lookup_order", {"order_id": "1042"}))``."""
    tool_calls = [{"id": f"call_{name}_{i}_{abs(hash(json.dumps(args, sort_keys=True))) % 10_000}", "type": "function",
                   "function": {"name": name, "arguments": json.dumps(args)}} for i, (name, args) in enumerate(calls)]
    return _completion({"role": "assistant", "content": None, "tool_calls": tool_calls}, "tool_calls")


def provider_down() -> tuple[int, dict[str, Any]]:
    return 503, {"error": {"message": "The server is overloaded.", "type": "server_error"}}


def _completion(message: dict[str, Any], finish: str) -> dict[str, Any]:
    return {"id": "chatcmpl-x", "object": "chat.completion", "created": 0, "model": "gpt-4o-mini",
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 20, "total_tokens": 140}}


def scripted_client(*answers: Any) -> openai.OpenAI:
    queue = list(answers)

    def handler(request: Any) -> Any:
        answer = queue.pop(0) if queue else say("(script exhausted)")
        status, body = answer if isinstance(answer, tuple) else (200, answer)
        return httpx.Response(status, json=body)

    return openai.OpenAI(api_key="sk-example", base_url="http://llm.example/v1", max_retries=0,
                         http_client=httpx.Client(transport=httpx.MockTransport(handler)))
