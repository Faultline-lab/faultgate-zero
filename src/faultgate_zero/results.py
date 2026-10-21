"""What a tool result means: an error, or nothing at all."""
from __future__ import annotations

import json
from typing import Any

from .trace import Step

ERROR_PREFIXES = ("error:", "exception:", "traceback (most recent call last)")
COUNT_KEYS = ("count", "total", "n_results", "total_count", "num_results")


def _json(s: str) -> Any:
    s = s.strip()
    if not s or s[0] not in "[{":
        return None
    try:
        return json.loads(s)
    except ValueError:
        return None


def error_of(step: Step) -> str | None:
    """Why this step's result is an error, or None. Only top-level signals count: an ``error`` key deep
    inside a successful payload is data, not a failure."""
    if step.is_error:
        return "the tool reported an error"
    s = step.result.strip()
    low = s.lower()
    for p in ERROR_PREFIXES:
        if low.startswith(p):
            return f"result starts with {s[:len(p)]!r}"
    d = _json(s)
    if isinstance(d, dict):
        err = d.get("error")
        if err not in (None, "", False, [], {}):
            return "result has an 'error' field"
        if str(d.get("status", "")).lower() in ("error", "failed", "failure"):
            return f"result status is {d['status']!r}"
    return None


def is_empty(result: str) -> bool:
    """A result that carried nothing back: empty, [], {}, null, or JSON whose lists are all empty or whose count is 0."""
    s = result.strip()
    if s in ("", "[]", "{}", "null", "None", '""'):
        return True
    d = _json(s)
    if isinstance(d, list):
        return not d
    if isinstance(d, dict):
        for k in COUNT_KEYS:
            if k in d and isinstance(d[k], (int, float)) and not isinstance(d[k], bool):
                return d[k] == 0
        lists = [v for v in d.values() if isinstance(v, list)]
        return bool(lists) and all(not v for v in lists) and all(
            isinstance(v, list) or v in (None, "", 0, False) or k in ("query", "q", "page", "status")
            for k, v in d.items())
    return False


def arguments_of(step: Step) -> dict[str, Any] | None:
    d = _json(step.arguments)
    return d if isinstance(d, dict) else None


def get_path(d: dict[str, Any], path: str) -> tuple[bool, Any]:
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return False, None
        cur = cur[key]
    return True, cur
