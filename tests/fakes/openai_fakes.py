"""Real OpenAI SDK clients over httpx2.MockTransport: no network, the SDK's own request and response code runs."""
from __future__ import annotations

import json
from typing import Any

import httpx2
import openai


def completion(content: str | None = None, tool_calls: list[tuple[str, str, Any]] | None = None,
               finish: str | None = None, model: str = "gpt-test", usage: tuple[int, int] = (10, 5),
               id: str = "chatcmpl-1") -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = [{"id": cid, "type": "function",
                              "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)}}
                             for cid, name, args in tool_calls]
    return {"id": id, "object": "chat.completion", "created": 0, "model": model,
            "choices": [{"index": 0, "message": msg, "finish_reason": finish or ("tool_calls" if tool_calls else "stop")}],
            "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1], "total_tokens": sum(usage)}}


class Script:
    """Answers each request with the next scripted response: a dict (HTTP 200) or (status, dict)."""

    def __init__(self, *responses: Any) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(request.content or b"{}"))
        if not self.responses:
            return httpx2.Response(500, json={"error": {"message": "script exhausted", "type": "server_error"}})
        r = self.responses.pop(0)
        if isinstance(r, SSE):
            return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=r.body.encode())
        status, body = r if isinstance(r, tuple) else (200, r)
        return httpx2.Response(status, json=body)


class SSE:
    """A server-sent-events body: ``events`` are (event name or None, data dict or "[DONE]")."""

    def __init__(self, events: list[tuple[str | None, Any]]) -> None:
        out = []
        for name, data in events:
            if name:
                out.append(f"event: {name}")
            out.append(f"data: {data if isinstance(data, str) else json.dumps(data)}")
            out.append("")
        self.body = "\n".join(out) + "\n"


def chat_stream(content: str = "", tool_calls: list[tuple[str, str, Any]] | None = None, finish: str | None = None,
                model: str = "gpt-test", usage: tuple[int, int] = (10, 5)) -> SSE:
    """The same completion as ``completion(...)``, cut into chunks: text 3 characters at a time, tool
    arguments split in two."""
    base = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 0, "model": model}

    def chunk(delta: dict[str, Any], fin: str | None = None) -> tuple[None, dict[str, Any]]:
        return None, {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": fin}]}

    events = [chunk({"role": "assistant"})]
    for i in range(0, len(content), 3):
        events.append(chunk({"content": content[i:i + 3]}))
    for n, (cid, name, args) in enumerate(tool_calls or []):
        a = args if isinstance(args, str) else json.dumps(args)
        half = len(a) // 2
        events.append(chunk({"tool_calls": [{"index": n, "id": cid, "type": "function",
                                             "function": {"name": name, "arguments": a[:half]}}]}))
        events.append(chunk({"tool_calls": [{"index": n, "function": {"arguments": a[half:]}}]}))
    events.append(chunk({}, finish or ("tool_calls" if tool_calls else "stop")))
    events.append((None, {**base, "choices": [], "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1],
                                                         "total_tokens": sum(usage)}}))
    events.append((None, "[DONE]"))
    return SSE(events)


def client(script: Script) -> openai.OpenAI:
    return openai.OpenAI(api_key="sk-test", base_url="http://fake.test/v1", max_retries=0,
                         http_client=httpx2.Client(transport=httpx2.MockTransport(script.handler)))


def async_client(script: Script) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(api_key="sk-test", base_url="http://fake.test/v1", max_retries=0,
                              http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(script.handler)))


def error_body(message: str, type_: str = "invalid_request_error", code: str | None = None) -> dict[str, Any]:
    return {"error": {"message": message, "type": type_, "code": code}}
