"""Read any trace export as faultgate-zero traces: ``load(path, fmt="auto")``.

OTLP/JSON goes through faultgate-zero's own loader. Every other format (OpenInference, Langfuse,
LangSmith, Braintrust) is read by faultgate's loaders, imported only when needed (install
``faultgate-zero[formats]``), then converted: both packages use the GenAI attribute names, so the
conversion copies spans as they are.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import otlp
from .trace import Span, Trace

FORMATS = ("auto", "otlp", "openinference", "langfuse", "langsmith", "braintrust")


class FormatError(ValueError):
    """An export that cannot be read: unknown format, bad JSON, or a format that needs faultgate."""


MESSAGE_ATTRS = ("gen_ai.input.messages", "gen_ai.output.messages")


def _part_from_openai(p: Any) -> dict[str, Any] | None:
    if isinstance(p, str):
        return {"type": "text", "content": p}
    if isinstance(p, dict) and p.get("type") in ("text", "input_text", "output_text"):
        return {"type": "text", "content": str(p.get("text", p.get("content", "")))}
    return None


def normalize_messages(raw: Any) -> Any:
    """Messages as GenAI ``parts``, which is what faultgate-zero reads. faultgate writes plain chat
    messages (``{"role", "content"}``); their text becomes text parts. Tool calls are read the way
    faultgate reads them, from parts and tool spans only, so a call is never counted twice.
    Messages that already have parts are kept."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return raw
    if not isinstance(raw, list):
        return raw
    out = []
    for m in raw:
        if not isinstance(m, dict) or isinstance(m.get("parts"), list):
            out.append(m)
            continue
        content = m.get("content")
        parts: list[dict[str, Any]] = []
        if isinstance(content, list):
            parts = [x for x in map(_part_from_openai, content) if x]
        elif content not in (None, ""):
            parts = [{"type": "text", "content": str(content)}]
        out.append({**{k: v for k, v in m.items() if k != "content"}, "parts": parts})
    return out


def to_zero(t: Any) -> Trace:
    """Convert a faultgate Trace to a faultgate-zero Trace: spans as they are, messages as parts."""
    spans = []
    for s in t.spans:
        attrs = dict(s.attrs)
        for k in MESSAGE_ATTRS:
            if k in attrs:
                attrs[k] = normalize_messages(attrs[k])
        spans.append(Span(span_id=s.span_id, parent_id=s.parent_id, name=s.name, start_ns=s.start_ns,
                          end_ns=s.end_ns, attrs=attrs, error=s.status_code == 2, status_message=s.status_message))
    root = t.root
    attrs = root.attrs if root else {}
    return Trace(trace_id=t.trace_id, scenario=str(attrs.get("faultgate.scenario") or t.key),
                 agent=t.agent, user_id=attrs.get("enduser.id"), spans=spans)


def _is_plain_otlp(doc: Any) -> bool:
    if not (isinstance(doc, dict) and "resourceSpans" in doc):
        return False
    for rs in _dicts(doc.get("resourceSpans")):
        for ss in _dicts(rs.get("scopeSpans")):
            for s in _dicts(ss.get("spans")):
                if any(a.get("key") == "openinference.span.kind" for a in _dicts(s.get("attributes"))):
                    return False
    return True


def _dicts(v: Any) -> list[dict[str, Any]]:
    return [x for x in v if isinstance(x, dict)] if isinstance(v, list) else []


def detect(doc: Any) -> str:
    if _is_plain_otlp(doc):
        return "otlp"
    return _faultgate().detect(doc)


def _faultgate() -> Any:
    try:
        from faultgate import formats as fg
    except ImportError:
        raise FormatError("this trace format needs faultgate's loaders: pip install 'faultgate-zero[formats]'") from None
    return fg


def load(path: str | Path, fmt: str = "auto") -> list[Trace]:
    """Every trace in one export file. Raises FormatError (a ValueError) on input it cannot read."""
    if fmt not in FORMATS:
        raise FormatError(f"unknown format {fmt!r}; available: {', '.join(FORMATS)}")
    p = Path(path)
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except OSError as exc:
        raise FormatError(f"{p}: cannot read it ({type(exc).__name__})") from None
    except ValueError:
        raise FormatError(f"{p}: not valid JSON") from None
    if fmt == "auto":
        try:
            fmt = detect(doc)
        except FormatError:
            raise
        except ValueError as exc:
            raise FormatError(f"{p}: {exc}") from None
        except (KeyError, TypeError, AttributeError):
            raise FormatError(f"{p}: unrecognised trace format") from None
    if fmt == "otlp":
        try:
            return otlp.from_dict(doc)
        except (KeyError, TypeError, AttributeError, ValueError) as exc:
            raise FormatError(f"{p}: not a valid OTLP/JSON export ({type(exc).__name__})") from None
    try:
        return [to_zero(t) for t in _faultgate().load(p, fmt)]
    except FormatError:
        raise
    except ValueError as exc:
        raise FormatError(f"{p}: {exc}") from None
    except (KeyError, TypeError, AttributeError):
        raise FormatError(f"{p}: not a valid {fmt} export") from None
