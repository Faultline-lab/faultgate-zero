"""Record streamed responses without changing them.

The SDK's own stream object is returned to the caller (``result is`` what the SDK built). Its class is
swapped for a subclass that feeds each chunk to an accumulator as it passes through unchanged and,
when the stream ends, is closed or its context exits, stores the assembled response on the model call.
A stream the agent stops reading early keeps what it read.
"""
from __future__ import annotations

import inspect
import json
from typing import Any, Callable

from .adapters._common import as_dict
from .recorder import ModelCall, Run

_classes: dict[type, type] = {}


class Recorder:
    def __init__(self, run: Run, call: ModelCall, feed: Callable[[Any], None], result: Callable[[], Any]) -> None:
        self.run, self.call, self._feed, self._result = run, call, feed, result
        self.iterating = False
        self.finished = False
        self.stream: Any = None  # for a manager: the stream its context opened

    def feed(self, item: Any) -> None:
        if self.finished:
            return
        try:
            self._feed(item)
        except Exception as exc:
            self.finished = True
            self.run.mark_incomplete(f"{self.call.adapter} stream", exc)

    def done(self) -> None:
        if self.finished:
            return
        self.finished = True
        try:
            self.call.response = self._result()
            self.call.end_ns = self.run.now()
            self.run._trace = None
        except Exception as exc:
            self.run.mark_incomplete(f"{self.call.adapter} stream", exc)


def _rec(obj: Any) -> Recorder | None:
    return obj.__dict__.get("_fgz_rec") if hasattr(obj, "__dict__") else None


# The recording subclass has the SDK class as its only base (Python 3.13 refuses __class__ assignment
# to a class with a mixin ahead of a Generic-based base), so its methods are closures over that base.


def _done(self: Any) -> None:
    rec = _rec(self)
    if rec:
        rec.done()


def _methods(base: type) -> dict[str, Any]:
    ns: dict[str, Any] = {}
    if inspect.isgeneratorfunction(getattr(base, "__iter__", None)):
        def __iter__(self):  # type: ignore[no-untyped-def]
            rec = _rec(self)
            it = base.__iter__(self)
            if rec is None:
                yield from it
                return
            rec.iterating = True
            try:
                for item in it:
                    rec.feed(item)
                    yield item
            finally:
                rec.iterating = False
                rec.done()
        ns["__iter__"] = __iter__
    if inspect.isasyncgenfunction(getattr(base, "__aiter__", None)):
        async def __aiter__(self):  # type: ignore[no-untyped-def]
            rec = _rec(self)
            it = base.__aiter__(self)
            if rec is None:
                async for item in it:
                    yield item
                return
            rec.iterating = True
            try:
                async for item in it:
                    rec.feed(item)
                    yield item
            finally:
                rec.iterating = False
                rec.done()
        ns["__aiter__"] = __aiter__
    if hasattr(base, "__next__"):
        def __next__(self):  # type: ignore[no-untyped-def]
            rec = _rec(self)
            try:
                item = base.__next__(self)
            except StopIteration:
                _done(self)
                raise
            if rec and not rec.iterating:
                rec.feed(item)
            return item
        ns["__next__"] = __next__
    if hasattr(base, "__anext__"):
        async def __anext__(self):  # type: ignore[no-untyped-def]
            rec = _rec(self)
            try:
                item = await base.__anext__(self)
            except StopAsyncIteration:
                _done(self)
                raise
            if rec and not rec.iterating:
                rec.feed(item)
            return item
        ns["__anext__"] = __anext__
    close = getattr(base, "close", None)
    if inspect.iscoroutinefunction(close):
        async def aclose(self):  # type: ignore[no-untyped-def]
            try:
                return await close(self)
            finally:
                _done(self)
        ns["close"] = aclose
    elif close is not None:
        def sclose(self):  # type: ignore[no-untyped-def]
            try:
                return close(self)
            finally:
                _done(self)
        ns["close"] = sclose
    if hasattr(base, "__enter__"):
        def __enter__(self):  # type: ignore[no-untyped-def]
            inner = base.__enter__(self)
            rec = _rec(self)
            if rec is not None and inner is not self:  # a manager: remember the stream it opened
                rec.stream = inner
            return inner
        ns["__enter__"] = __enter__
    if hasattr(base, "__exit__"):
        def __exit__(self, *exc):  # type: ignore[no-untyped-def]
            try:
                return base.__exit__(self, *exc)
            finally:
                _done(self)
        ns["__exit__"] = __exit__
    if hasattr(base, "__aenter__"):
        async def __aenter__(self):  # type: ignore[no-untyped-def]
            inner = await base.__aenter__(self)
            rec = _rec(self)
            if rec is not None and inner is not self:
                rec.stream = inner
            return inner
        ns["__aenter__"] = __aenter__
    if hasattr(base, "__aexit__"):
        async def __aexit__(self, *exc):  # type: ignore[no-untyped-def]
            try:
                return await base.__aexit__(self, *exc)
            finally:
                _done(self)
        ns["__aexit__"] = __aexit__
    return ns


def _subclass(cls: type) -> type:
    """A subclass of ``cls`` that records, overriding only the methods ``cls`` has."""
    if cls not in _classes:
        sub = type(cls.__name__, (cls,), _methods(cls))
        sub.__qualname__, sub.__module__ = cls.__qualname__, cls.__module__
        _classes[cls] = sub
    return _classes[cls]


def attach_manager(manager: Any, run: Run, call: ModelCall, result: Callable[[Any], Any]) -> Any:
    """Record a stream *manager* (``with client.messages.stream(...) as stream:``): when its context exits,
    ``result(inner_stream)`` gives the assembled response."""
    rec = Recorder(run, call, lambda item: None, lambda: result(rec.stream))
    rec.stream = None
    manager.__class__ = _subclass(type(manager))
    manager.__dict__["_fgz_rec"] = rec
    return manager


def attach(stream: Any, run: Run, call: ModelCall, feed: Callable[[Any], None], result: Callable[[], Any]) -> Any:
    """Record ``stream`` in place and return it. Raises if the object cannot be recorded."""
    rec = Recorder(run, call, feed, result)
    stream.__class__ = _subclass(type(stream))
    stream.__dict__["_fgz_rec"] = rec
    return stream


# accumulators


class ChatChunks:
    """OpenAI-format chat completion chunks (OpenAI, LiteLLM) -> one completion dict."""

    def __init__(self) -> None:
        self.content: list[str] = []
        self.refusal: list[str] = []
        self.tools: dict[int, dict[str, Any]] = {}
        self.finish: str | None = None
        self.usage: dict[str, Any] | None = None
        self.model = self.id = None

    def feed(self, chunk: Any) -> None:
        d = as_dict(chunk)
        self.model = d.get("model") or self.model
        self.id = d.get("id") or self.id
        if d.get("usage"):
            self.usage = d["usage"]
        for c in d.get("choices") or []:
            if c.get("index", 0) != 0:
                continue
            delta = c.get("delta") or {}
            if delta.get("content"):
                self.content.append(delta["content"])
            if delta.get("refusal"):
                self.refusal.append(delta["refusal"])
            for tc in delta.get("tool_calls") or []:
                slot = self.tools.setdefault(tc.get("index", len(self.tools)),
                                             {"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["function"]["name"] += fn["name"]
                if fn.get("arguments"):
                    slot["function"]["arguments"] += fn["arguments"]
            if c.get("finish_reason"):
                self.finish = c["finish_reason"]

    def result(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": "assistant", "content": "".join(self.content) or None}
        if self.refusal:
            msg["refusal"] = "".join(self.refusal)
        if self.tools:
            msg["tool_calls"] = [self.tools[i] for i in sorted(self.tools)]
        out: dict[str, Any] = {"id": self.id, "model": self.model,
                               "choices": [{"index": 0, "message": msg, "finish_reason": self.finish}]}
        if self.usage:
            out["usage"] = self.usage
        return out


class AnthropicEvents:
    """Anthropic message stream events -> one message dict."""

    def __init__(self) -> None:
        self.msg: dict[str, Any] = {}
        self.blocks: dict[int, dict[str, Any]] = {}
        self.json: dict[int, str] = {}

    def feed(self, event: Any) -> None:
        e = as_dict(event)
        t = e.get("type")
        if t == "message_start":
            self.msg = dict(e.get("message") or {})
        elif t == "content_block_start":
            self.blocks[e.get("index", 0)] = dict(e.get("content_block") or {})
        elif t == "content_block_delta":
            i, d = e.get("index", 0), e.get("delta") or {}
            b = self.blocks.setdefault(i, {"type": "text", "text": ""})
            if d.get("type") == "text_delta":
                b["text"] = (b.get("text") or "") + d.get("text", "")
            elif d.get("type") == "input_json_delta":
                self.json[i] = self.json.get(i, "") + d.get("partial_json", "")
        elif t == "content_block_stop":
            i = e.get("index", 0)
            if i in self.json:
                try:
                    self.blocks[i]["input"] = json.loads(self.json[i] or "{}")
                except ValueError:
                    self.blocks[i]["input"] = self.json[i]
        elif t == "message_delta":
            d = e.get("delta") or {}
            if d.get("stop_reason"):
                self.msg["stop_reason"] = d["stop_reason"]
            if e.get("usage"):
                self.msg["usage"] = {**(self.msg.get("usage") or {}), **{k: v for k, v in e["usage"].items() if v is not None}}

    def result(self) -> dict[str, Any]:
        for i, buf in self.json.items():  # a stream cut before content_block_stop
            b = self.blocks.get(i)
            if b is not None and "input" not in b:
                try:
                    b["input"] = json.loads(buf or "{}")
                except ValueError:
                    b["input"] = buf
        return {**self.msg, "content": [self.blocks[i] for i in sorted(self.blocks)]}


class ResponsesEvents:
    """OpenAI Responses stream events -> the final response (sent whole in the last event)."""

    def __init__(self) -> None:
        self.final: Any = None
        self.partial: Any = None

    def feed(self, event: Any) -> None:
        e = as_dict(event)
        t = str(e.get("type", ""))
        if t in ("response.completed", "response.incomplete", "response.failed"):
            self.final = e.get("response")
        elif t in ("response.created", "response.in_progress") and e.get("response"):
            self.partial = e["response"]

    def result(self) -> Any:
        return self.final if self.final is not None else self.partial
