"""OTLP/JSON export and load (``resourceSpans``), so captured runs can be kept and read back.

Message attributes are written as JSON strings, as the GenAI semantic conventions do for span
attributes. ``load_otlp(export(traces))`` gives traces whose steps, answer and verdicts equal the
in-memory ones.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import __version__
from .trace import CHAT, Span, Trace

JSON_ATTRS = ("gen_ai.input.messages", "gen_ai.output.messages")
_KIND = {CHAT: 3}  # SPAN_KIND_CLIENT; everything else SPAN_KIND_INTERNAL (1)


def _encode(v: Any) -> dict[str, Any]:
    if isinstance(v, bool):
        return {"boolValue": v}
    if isinstance(v, int):
        return {"intValue": str(v)}
    if isinstance(v, float):
        return {"doubleValue": v}
    if isinstance(v, str):
        return {"stringValue": v}
    if isinstance(v, (list, tuple)) and all(isinstance(x, (str, int, float, bool)) for x in v) and v:
        return {"arrayValue": {"values": [_encode(x) for x in v]}}
    return {"stringValue": json.dumps(v, ensure_ascii=False, default=str)}


def _decode(v: dict[str, Any]) -> Any:
    if "stringValue" in v:
        return v["stringValue"]
    if "intValue" in v:
        return int(v["intValue"])
    if "doubleValue" in v:
        return float(v["doubleValue"])
    if "boolValue" in v:
        return bool(v["boolValue"])
    if "arrayValue" in v:
        return [_decode(x) for x in v["arrayValue"].get("values", [])]
    return None


def _span(t: Trace, s: Span) -> dict[str, Any]:
    out: dict[str, Any] = {
        "traceId": t.trace_id,
        "spanId": s.span_id,
        "name": s.name,
        "kind": _KIND.get(s.operation, 1),
        "startTimeUnixNano": str(s.start_ns),
        "endTimeUnixNano": str(s.end_ns),
        "attributes": [{"key": k, "value": _encode(v)} for k, v in s.attrs.items()],
        "status": {"code": 2, "message": s.status_message} if s.error else {"code": 0},
    }
    if s.parent_id:
        out["parentSpanId"] = s.parent_id
    return out


def export(traces: list[Trace]) -> dict[str, Any]:
    return {"resourceSpans": [{
        "resource": {"attributes": [{"key": "service.name", "value": {"stringValue": "faultgate-zero"}}]},
        "scopeSpans": [{"scope": {"name": "faultgate_zero", "version": __version__},
                        "spans": [_span(t, s) for t in traces for s in t.spans]}],
    }]}


def write(traces: list[Trace], path: str | Path) -> None:
    Path(path).write_text(json.dumps(export(traces), ensure_ascii=False, indent=1), encoding="utf-8")


def _status_error(st: dict[str, Any]) -> bool:
    code = st.get("code", 0)
    return code == 2 or str(code) in ("2", "STATUS_CODE_ERROR")


def from_dict(doc: dict[str, Any]) -> list[Trace]:
    by_trace: dict[str, list[Span]] = {}
    for rs in doc.get("resourceSpans", []):
        for ss in rs.get("scopeSpans", []):
            for s in ss.get("spans", []):
                attrs = {a["key"]: _decode(a.get("value", {})) for a in s.get("attributes", [])}
                for k in JSON_ATTRS:
                    if isinstance(attrs.get(k), str):
                        try:
                            attrs[k] = json.loads(attrs[k])
                        except ValueError:
                            pass
                st = s.get("status") or {}
                by_trace.setdefault(s["traceId"], []).append(Span(
                    span_id=s["spanId"], parent_id=s.get("parentSpanId") or None, name=s.get("name", ""),
                    start_ns=int(s.get("startTimeUnixNano", 0)), end_ns=int(s.get("endTimeUnixNano", 0)),
                    attrs=attrs, error=_status_error(st), status_message=str(st.get("message", "")),
                ))
    traces = []
    for tid, spans in by_trace.items():
        spans.sort(key=lambda x: x.start_ns)
        t = Trace(tid, "", "agent", None, spans)
        r = t.root
        if r:
            t.scenario = str(r.attrs.get("faultgate.scenario", "") or tid)
            t.agent = str(r.attrs.get("gen_ai.agent.name", "agent"))
            t.user_id = r.attrs.get("enduser.id")
        traces.append(t)
    traces.sort(key=lambda t: t.spans[0].start_ns if t.spans else 0)
    return traces


def load_otlp(path: str | Path) -> list[Trace]:
    return from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
