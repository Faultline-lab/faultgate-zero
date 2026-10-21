"""Faultgate-Zero's trace model: one agent run as spans, and the tool steps read from them.

One trace = one agent run: a root ``invoke_agent`` span, one ``chat`` span per model call and one
``execute_tool`` span per ``@fz.tool`` call. Attributes follow the OpenTelemetry GenAI semantic
conventions, so the same model is built from a live run and from an OTLP/JSON file.

Steps (the tool calls the agent made) are read from ``tool_call`` parts in each chat span's
``gen_ai.output.messages`` (that response only, never the re-sent history) and matched by call id to
``tool_call_response`` parts in later ``gen_ai.input.messages``. An ``execute_tool`` span with the
same call id replaces the message-derived result; one without a call id is a step of its own.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

AGENT, CHAT, TOOL = "invoke_agent", "chat", "execute_tool"


@dataclass
class Span:
    span_id: str
    parent_id: str | None
    name: str
    start_ns: int
    end_ns: int = 0
    attrs: dict[str, Any] = field(default_factory=dict)
    error: bool = False
    status_message: str = ""

    @property
    def operation(self) -> str:
        return str(self.attrs.get("gen_ai.operation.name", self.name.split(" ")[0]))


@dataclass
class Step:
    index: int
    tool: str
    call_id: str
    arguments: str
    result: str
    is_error: bool
    result_seen: bool
    span_id: str = ""


@dataclass
class Trace:
    trace_id: str
    scenario: str
    agent: str
    user_id: str | None
    spans: list[Span]

    @property
    def root(self) -> Span | None:
        for s in self.spans:
            if s.operation == AGENT:
                return s
        return next((s for s in self.spans if s.parent_id is None), None)

    @property
    def model_calls(self) -> list[Span]:
        return [s for s in self.spans if s.operation == CHAT]

    @property
    def prompt(self) -> str:
        r = self.root
        text = text_of(r.attrs.get("gen_ai.input.messages"), "user") if r else ""
        if text:
            return text
        for s in self.model_calls:
            text = text_of(s.attrs.get("gen_ai.input.messages"), "user")
            if text:
                return text
        return ""

    @property
    def answer(self) -> str:
        """The run's final answer: assistant text from the last model response that asked for no tools.

        A run whose last response still asks for tools has no answer, whatever text came before it.
        """
        r = self.root
        if r and "gen_ai.output.messages" in r.attrs:
            return text_of(r.attrs["gen_ai.output.messages"], "assistant")
        return answer_from_calls(self.model_calls)

    @property
    def termination(self) -> str:
        r = self.root
        return str(r.attrs.get("faultgate.termination", "")) if r else ""

    @property
    def capture_complete(self) -> bool:
        r = self.root
        return bool(r.attrs.get("faultgate.capture_complete", True)) if r else True

    @property
    def usage(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in self.model_calls:
            for k in ("input_tokens", "output_tokens"):
                v = s.attrs.get(f"gen_ai.usage.{k}")
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    out[k] = out.get(k, 0) + int(v)
        return out

    @property
    def latency_ms(self) -> float | None:
        r = self.root
        if r and r.end_ns and r.end_ns >= r.start_ns:
            return (r.end_ns - r.start_ns) / 1e6
        return None

    @property
    def steps(self) -> list[Step]:
        results: dict[str, tuple[str, bool]] = {}
        for s in self.model_calls:
            for m in messages(s.attrs.get("gen_ai.input.messages")):
                for p in parts(m):
                    if p.get("type") == "tool_call_response" and p.get("id"):
                        results[str(p["id"])] = (payload_text(p.get("result")), _flag_error(p))

        timed: list[tuple[int, int, Step]] = []
        by_id: dict[str, Step] = {}
        seq = 0
        for s in self.model_calls:
            for m in messages(s.attrs.get("gen_ai.output.messages")):
                for p in parts(m):
                    if p.get("type") != "tool_call":
                        continue
                    cid = str(p.get("id") or "")
                    res, err = results.get(cid, ("", False))
                    st = Step(0, str(p.get("name") or "unknown"), cid, canonical_args(p.get("arguments")),
                              res, err, cid in results, s.span_id)
                    if cid and cid not in by_id:
                        by_id[cid] = st
                    timed.append((s.end_ns or s.start_ns, seq, st))
                    seq += 1

        for s in self.spans:
            if s.operation != TOOL:
                continue
            cid = str(s.attrs.get("gen_ai.tool.call.id") or "")
            result = str(s.attrs.get("gen_ai.tool.call.result", ""))
            if cid and cid in by_id:
                st = by_id[cid]
                st.result, st.is_error, st.result_seen, st.span_id = result, s.error, True, s.span_id
                continue
            st = Step(0, str(s.attrs.get("gen_ai.tool.name", s.name)), cid,
                      canonical_args(s.attrs.get("gen_ai.tool.call.arguments", "")), result, s.error, True, s.span_id)
            timed.append((s.start_ns, seq, st))
            seq += 1

        timed.sort(key=lambda x: (x[0], x[1]))
        out = []
        for i, (_, _, st) in enumerate(timed, start=1):
            st.index = i
            out.append(st)
        return out


def answer_from_calls(calls: list[Span]) -> str:
    ok = [s for s in calls if not s.error]
    if not ok:
        return ""
    last = messages(ok[-1].attrs.get("gen_ai.output.messages"))
    if any(p.get("type") == "tool_call" for m in last for p in parts(m)):
        return ""
    return text_of(last, "assistant")


def _flag_error(p: dict[str, Any]) -> bool:
    return p.get("is_error") is True or bool(p.get("error")) or str(p.get("status", "")).lower() in ("error", "failed")


def messages(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return []
    return [m for m in raw if isinstance(m, dict)] if isinstance(raw, list) else []


def parts(m: dict[str, Any]) -> list[dict[str, Any]]:
    return [p for p in m.get("parts") or [] if isinstance(p, dict)]


def text_of(raw: Any, role: str) -> str:
    out = []
    for m in messages(raw):
        if m.get("role") != role:
            continue
        for p in parts(m):
            if p.get("type") == "text":
                out.append(str(p.get("content", "")))
    return "\n".join(t for t in out if t)


def canonical_args(args: Any) -> str:
    """Arguments as compact JSON with sorted keys when they parse, else the raw text."""
    if args is None:
        return ""
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return args
    try:
        return json.dumps(args, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(args)


def payload_text(val: Any) -> str:
    """A tool result as text: strings as they are, text-part lists joined, anything else as JSON."""
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    if isinstance(val, list) and val and all(isinstance(x, dict) and x.get("type") == "text" for x in val):
        return "".join(str(x.get("text", x.get("content", ""))) for x in val)
    try:
        return json.dumps(val, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(val)
