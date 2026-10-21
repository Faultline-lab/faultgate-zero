"""Reports over checked runs: one entry per run, as JSON (``--fgz-json``) and Markdown (``--fgz-markdown``).

Entries are plain dicts so xdist workers can send them to the controller. Reasons in them are
already redacted.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from . import __version__
from .runner import RunResult

RUN_KEYS = ("test", "scenario", "steps", "model_calls", "tokens", "latency_ms", "capture_complete", "skipped",
            "checked_by_test", "verdicts")


def entry(test: str, result: RunResult, test_passed: bool, checked_by_test: bool = False) -> dict[str, Any]:
    """Test ids and run names are masked like reasons: parametrize ids and run names can carry data."""
    t = result.trace
    u = t.usage
    return {
        "test": result.mask(test),
        "scenario": result.scenario,
        "checked_by_test": checked_by_test,
        "steps": len(t.steps),
        "model_calls": len(t.model_calls),
        "tokens": {"input": u.get("input_tokens"), "output": u.get("output_tokens")},
        "latency_ms": round(t.latency_ms, 3) if t.latency_ms is not None else None,
        "capture_complete": t.capture_complete,
        "skipped": result.skipped,
        "verdicts": [{"check": v.check, "fired": v.fired, "reason": v.reason, "step": v.step} for v in result.verdicts],
        # internal: for the terminal summary, not written to the JSON report
        "_fired": [v.check for v in result.fired],
        "_summary": result.summary(),
        "_test_passed": test_passed,
    }


def summary(entries: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [e for e in entries if e["_fired"] and not e.get("checked_by_test")]
    return {"runs": len(entries), "failed_runs": len(failed),
            "skipped_runs": sum(1 for e in entries if e["skipped"]),
            "fired": dict(sorted(Counter(c for e in failed for c in e["_fired"]).items()))}


def to_dict(entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {"tool": "faultgate-zero", "version": __version__,
            "runs": [{k: e[k] for k in RUN_KEYS} for e in entries], "summary": summary(entries)}


def write_json(entries: list[dict[str, Any]], path: str | Path) -> None:
    Path(path).write_text(json.dumps(to_dict(entries), indent=2, ensure_ascii=False), encoding="utf-8")


def _cell(s: Any) -> str:
    return str("" if s is None else s).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def markdown(entries: list[dict[str, Any]]) -> str:
    s = summary(entries)
    lines = ["## faultgate-zero", "",
             f"**{s['runs']} runs checked, {s['failed_runs']} with failures**"
             + (f", {s['skipped_runs']} skipped (capture incomplete)" if s["skipped_runs"] else ""), ""]
    rows = [(e, v) for e in entries if not e.get("checked_by_test") for v in e["verdicts"] if v["fired"]]
    if not rows:
        lines.append("All checked runs are clean.")
        return "\n".join(lines) + "\n"
    lines += ["| Test | Run | Check | Step | Reason |", "|---|---|---|---|---|"]
    for e, v in rows:
        lines.append(f"| {_cell(e['test'])} | {_cell(e['scenario'])} | {_cell(v['check'])} | {_cell(v['step'])} | {_cell(v['reason'])} |")
    return "\n".join(lines) + "\n"


def write_markdown(entries: list[dict[str, Any]], path: str | Path) -> None:
    Path(path).write_text(markdown(entries), encoding="utf-8")
