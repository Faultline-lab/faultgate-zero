"""Faultgate-Zero's checks: deterministic rules over one Trace. No network, no LLM judge.

Each check returns one Verdict. Reasons name the step and the tool so a failing test points at the
exact call; they are redacted before anything prints them.
"""
from __future__ import annotations

import re
import warnings
from collections import defaultdict
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Callable, Iterable

from .policy import Policy
from .results import error_of, is_empty
from .trace import Trace


@dataclass(frozen=True)
class Verdict:
    check: str
    fired: bool
    reason: str
    step: int | None = None


@dataclass(frozen=True)
class Options:
    loop_min_steps: int = 5
    loop_empty_share: float = 0.5
    repeat_threshold: int = 3


CUTOFF_REASONS = {"length", "max_tokens", "max_output_tokens", "model_context_window_exceeded"}
OVERFLOW = re.compile(
    r"context[ _-]?length[ _-]?exceeded|maximum context length|context[ _-]?window[ _-]?(?:exceeded|overflow)"
    r"|prompt is too long|too many (?:input )?tokens|exceeds?\s+the\s+(?:model'?s\s+)?(?:context|token)\s+(?:window|limit)"
    r"|input is too long for (?:the )?requested model|exceeds the maximum number of tokens",
    re.I,
)
ABSTAIN = re.compile(
    r"\b(?:could ?n[o']t|can ?n[o']t|cannot|unable to|was unable to|did not|didn't) (?:find|locate|retrieve|determine)\b"
    r"|\bno (?:information|results|data|records?) (?:was |were )?(?:found|available)\b|\bno information (?:about|on)\b",
    re.I,
)


def _short(s: str, n: int = 80) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def search_loop(t: Trace, o: Options) -> Verdict:
    steps = t.steps
    if t.answer.strip() or len(steps) < o.loop_min_steps:
        return Verdict("search_loop", False, "answered" if t.answer.strip() else f"{len(steps)} tool calls")
    seen = [s for s in steps if s.result_seen]
    empty = sum(is_empty(s.result) for s in seen)
    if not seen or empty / len(seen) < o.loop_empty_share:
        return Verdict("search_loop", False, f"{empty} of {len(seen)} results empty")
    tools = sorted({s.tool for s in steps})
    return Verdict("search_loop", True,
                   f"no answer after {len(steps)} tool calls ({', '.join(tools)}); {empty} of {len(seen)} results were empty",
                   steps[-1].index)


def repeated_call(t: Trace, o: Options) -> Verdict:
    groups: dict[tuple[str, str], list] = defaultdict(list)
    for s in t.steps:
        groups[(s.tool, s.arguments)].append(s)
    for (tool, args), calls in groups.items():
        if len(calls) < o.repeat_threshold:
            continue
        seen = [c for c in calls if c.result_seen]  # an unseen result is not "the same result"
        if len(seen) >= o.repeat_threshold and len({c.result for c in seen}) == 1:
            at = seen[o.repeat_threshold - 1].index
            return Verdict("repeated_call", True,
                           f"step {at}: {tool} called {len(calls)}x with identical "
                           f"arguments {_short(args)} and identical results", at)
    return Verdict("repeated_call", False, "no identical repeated calls")


def unrecovered_tool_error(t: Trace, o: Options) -> Verdict:
    steps = t.steps
    for i, s in enumerate(steps):
        why = error_of(s)
        if why is None or s.escaped:
            continue
        if any(later.tool == s.tool and later.result_seen and error_of(later) is None for later in steps[i + 1:]):
            continue
        return Verdict("unrecovered_tool_error", True,
                       f"step {s.index}: {s.tool} failed ({why}: {_short(s.result, 120)!r}) and was never called "
                       f"successfully again", s.index)
    return Verdict("unrecovered_tool_error", False, "no unrecovered tool errors")


def unrecovered_model_error(t: Trace, o: Options) -> Verdict:
    calls = t.model_calls
    for i, c in enumerate(calls):
        if c.attrs.get("faultgate.error_escaped"):
            continue  # it left the agent: the run or the test dealt with it
        if c.error and not any(not later.error for later in calls[i + 1:]):
            kind = c.attrs.get("error.type", "error")
            return Verdict("unrecovered_model_error", True,
                           f"model call {i + 1} of {len(calls)} raised {kind} ({_short(c.status_message, 160)}) "
                           f"and no later model call succeeded")
    return Verdict("unrecovered_model_error", False, "no unrecovered model errors")


def context_overflow(t: Trace, o: Options) -> Verdict:
    calls = t.model_calls
    for i, c in enumerate(calls, start=1):
        raw = c.attrs.get("gen_ai.response.finish_reasons")
        reasons = raw if isinstance(raw, (list, tuple)) else ([] if raw is None else [raw])
        cut = next((str(r) for r in reasons if str(r).lower() in CUTOFF_REASONS), None)
        if cut:
            return Verdict("context_overflow", True, f"model call {i} stopped on finish reason {cut!r}: output cut at the token limit")
        for text in (c.status_message, str(c.attrs.get("error.type", ""))):
            m = OVERFLOW.search(text or "")
            if m:
                return Verdict("context_overflow", True, f"model call {i} hit the context limit: {m.group(0)!r}")
    m = OVERFLOW.search(t.termination)
    if m:
        return Verdict("context_overflow", True, f"run ended on a context-limit error: {m.group(0)!r}")
    return Verdict("context_overflow", False, "no token cutoff")


def abstention(t: Trace, o: Options) -> Verdict:
    m = ABSTAIN.search(t.answer)
    if m:
        return Verdict("abstention", True, f"final answer gives up: {m.group(0)!r}")
    return Verdict("abstention", False, "answer does not abstain")


def policy_check(t: Trace, policy: Policy) -> Verdict:
    bad = policy.violations(t)
    if bad:
        return Verdict("policy", True, "; ".join(msg for _, msg in bad), next((s for s, _ in bad if s), None))
    return Verdict("policy", False, f"{len(t.steps)} tool calls within policy")


RULES: dict[str, Callable[[Trace, Options], Verdict]] = {
    "search_loop": search_loop,
    "repeated_call": repeated_call,
    "unrecovered_tool_error": unrecovered_tool_error,
    "unrecovered_model_error": unrecovered_model_error,
    "context_overflow": context_overflow,
    "abstention": abstention,
}
BUILTIN_DEFAULTS = ("search_loop", "repeated_call", "unrecovered_tool_error", "unrecovered_model_error",
                    "context_overflow", "policy")

# Check plugins: other packages add checks through the ``faultgate_zero.checks`` entry point group.
# Each entry point loads an object with ``name: str``, ``default: bool`` and
# ``run(trace: Trace, options: Options) -> Verdict``. Discovery happens once, when this module loads.
PLUGIN_GROUP = "faultgate_zero.checks"
CRASHED = "check crashed: "


class PluginError(ValueError):
    """A check plugin that cannot be used: its name clashes with a built-in check, or it is malformed."""


def discover(eps: Iterable[Any] | None = None) -> tuple[dict[str, Any], list[str]]:
    """Load check plugins from entry points. Returns (plugins by name, errors). Never raises."""
    if eps is None:
        try:
            eps = metadata.entry_points(group=PLUGIN_GROUP)
        except Exception as exc:  # a broken install must not stop faultgate-zero from loading
            return {}, [f"cannot read {PLUGIN_GROUP} entry points: {type(exc).__name__}"]
    found: dict[str, Any] = {}
    errors: list[str] = []
    builtin = (*RULES, "policy")
    for ep in eps:
        where = f"check plugin {getattr(ep, 'name', '?')!r} ({getattr(ep, 'value', '?')})"
        try:
            obj = ep.load()
        except Exception as exc:
            errors.append(f"{where}: cannot load it: {type(exc).__name__}: {exc}")
            continue
        name = getattr(obj, "name", None)
        if not isinstance(name, str) or not name or not callable(getattr(obj, "run", None))                 or not isinstance(getattr(obj, "default", None), bool):
            errors.append(f"{where}: needs name (str), default (bool) and run(trace, options)")
            continue
        if name in builtin:
            errors.append(f"{where}: name {name!r} clashes with a built-in check")
            continue
        if name in found:
            errors.append(f"{where}: name {name!r} is used by two plugins")
            continue
        found[name] = obj
    return found, errors


PLUGINS, PLUGIN_ERRORS = discover()
CHECK_NAMES = (*RULES, "policy", *PLUGINS)
DEFAULT_CHECKS = (*BUILTIN_DEFAULTS, *(n for n, p in PLUGINS.items() if p.default))


def plugin_errors() -> list[str]:
    return list(PLUGIN_ERRORS)


def _raise_plugin_errors() -> None:
    if PLUGIN_ERRORS:
        raise PluginError(PLUGIN_ERRORS[0])


def resolve(names: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    """Check names from a comma list or sequence; None means the default set. Unknown names are a usage error."""
    _raise_plugin_errors()
    if names is None:
        return DEFAULT_CHECKS
    wanted = [n.strip() for n in (names.split(",") if isinstance(names, str) else names) if n.strip()]
    unknown = [n for n in wanted if n not in CHECK_NAMES]
    if unknown:
        raise ValueError(f"unknown check {unknown[0]!r}; available: {', '.join(CHECK_NAMES)}")
    return tuple(dict.fromkeys(wanted))


def run_plugin(plugin: Any, t: Trace, options: Options) -> Verdict:
    """Run one plugin check. A plugin that raises, or returns something else than a Verdict, gives a
    verdict that did not fire and a warning: the user's test never fails because of a plugin bug."""
    from .recorder import FaultgateZeroWarning

    name = plugin.name
    try:
        v = plugin.run(t, options)
        if not isinstance(v, Verdict):
            raise TypeError(f"run() returned {type(v).__name__}, not Verdict")
    except Exception as exc:
        try:  # under -W error the warning raises; it must not fail the test either
            warnings.warn(f"faultgate-zero: check {name!r} crashed ({type(exc).__name__}); it did not fire",
                          FaultgateZeroWarning, stacklevel=2)
        except Exception:
            pass
        return Verdict(name, False, f"{CRASHED}{type(exc).__name__}")
    return v if v.check == name else Verdict(name, v.fired, v.reason, v.step)


def crashed(v: Verdict) -> bool:
    return not v.fired and v.reason.startswith(CRASHED)


def run_checks(t: Trace, names: tuple[str, ...], policy: Policy, options: Options) -> list[Verdict]:
    out = []
    for n in names:
        if n == "policy":
            out.append(policy_check(t, policy))
        elif n in RULES:
            out.append(RULES[n](t, options))
        else:
            out.append(run_plugin(PLUGINS[n], t, options))
    return out
