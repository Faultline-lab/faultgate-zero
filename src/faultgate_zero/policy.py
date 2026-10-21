"""Policy file (JSON, spec v1.1 section D): what an agent's tool calls are allowed to do."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # the stdlib regex parser (Python 3.11+); used only to vet deny patterns
    import re._constants as _c
    import re._parser as _parser
except ImportError:  # pragma: no cover
    import sre_constants as _c  # type: ignore[no-redef]
    import sre_parse as _parser  # type: ignore[no-redef]

from .results import arguments_of, error_of, get_path
from .trace import Trace

DEFAULT_DENY =(r"\.\./", r"\.\.\\", r"file://", r"~/\.ssh", r"/etc/passwd", r"id_rsa")
TOP_KEYS = {"allow", "deny_patterns", "max_arg_chars", "max_calls", "require_complete", "tools"}
TOOL_KEYS = {"max_calls", "limits", "allowed_values", "required_args", "identity_arg", "requires"}


class PolicyError(ValueError):
    """A policy file that cannot be used. The message names the file, the key and the problem."""


@dataclass
class ToolRule:
    max_calls: int | None = None
    limits: dict[str, float] = field(default_factory=dict)
    allowed_values: dict[str, list[Any]] = field(default_factory=dict)
    required_args: list[str] = field(default_factory=list)
    identity_arg: str | None = None
    requires: list[str] = field(default_factory=list)


@dataclass
class Policy:
    allow: list[str] | None = None
    deny: list[re.Pattern[str]] = field(default_factory=lambda: [re.compile(p, re.I) for p in DEFAULT_DENY])
    max_arg_chars: int | None = None
    max_calls: int | None = None
    require_complete: bool = False
    tools: dict[str, ToolRule] = field(default_factory=dict)
    source: str = "default"

    @classmethod
    def load(cls, path: str | Path) -> Policy:
        p = Path(path)
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise PolicyError(f"policy file {p} not found") from None
        except ValueError as exc:
            raise PolicyError(f"policy file {p} is not valid JSON: {exc}") from None
        return cls.from_dict(raw, source=str(p))

    @classmethod
    def from_dict(cls, d: Any, source: str = "<dict>") -> Policy:
        def fail(key: str, problem: str) -> PolicyError:
            return PolicyError(f"policy {source}: {key}: {problem}")

        if not isinstance(d, dict):
            raise fail("(top level)", "must be a JSON object")
        unknown = set(d) - TOP_KEYS
        if unknown:
            raise fail(sorted(unknown)[0], f"unknown key; allowed keys: {', '.join(sorted(TOP_KEYS))}")
        pol = cls(source=source)
        if "allow" in d:
            pol.allow = _str_list(d["allow"], "allow", fail)
        if "deny_patterns" in d:
            pats = _str_list(d["deny_patterns"], "deny_patterns", fail)
            slow = next((x for x in pats if _can_backtrack_exponentially(x)), None)
            if slow is not None:
                raise fail("deny_patterns", f"{slow!r} repeats a group that itself repeats or has alternatives, "
                                            "which can take exponential time on model-written arguments; "
                                            "rewrite it without that, e.g. as several simpler patterns")
            try:
                pol.deny = [re.compile(x, re.I) for x in pats]
            except re.error as exc:
                raise fail("deny_patterns", f"bad regex: {exc}") from None
        for k in ("max_arg_chars", "max_calls"):
            if k in d:
                setattr(pol, k, _pos_int(d[k], k, fail))
        if "require_complete" in d:
            if not isinstance(d["require_complete"], bool):
                raise fail("require_complete", "must be true or false")
            pol.require_complete = d["require_complete"]
        tools = d.get("tools", {})
        if not isinstance(tools, dict):
            raise fail("tools", "must be an object of tool name -> rules")
        for name, rule in tools.items():
            key = f"tools.{name}"
            if not isinstance(rule, dict):
                raise fail(key, "must be an object")
            bad = set(rule) - TOOL_KEYS
            if bad:
                raise fail(f"{key}.{sorted(bad)[0]}", f"unknown key; allowed keys: {', '.join(sorted(TOOL_KEYS))}")
            tr = ToolRule()
            if "max_calls" in rule:
                tr.max_calls = _pos_int(rule["max_calls"], f"{key}.max_calls", fail)
            limits = rule.get("limits", {})
            if not isinstance(limits, dict) or not all(_num(v) for v in limits.values()):
                raise fail(f"{key}.limits", "must map argument names to numbers")
            tr.limits = {k: float(v) for k, v in limits.items()}
            allowed = rule.get("allowed_values", {})
            if not isinstance(allowed, dict) or not all(isinstance(v, list) for v in allowed.values()):
                raise fail(f"{key}.allowed_values", "must map argument names to lists of values")
            tr.allowed_values = allowed
            tr.required_args = _str_list(rule.get("required_args", []), f"{key}.required_args", fail)
            if "identity_arg" in rule:
                if not isinstance(rule["identity_arg"], str) or not rule["identity_arg"]:
                    raise fail(f"{key}.identity_arg", "must be an argument name")
                tr.identity_arg = rule["identity_arg"]
            tr.requires = _str_list(rule.get("requires", []), f"{key}.requires", fail)
            pol.tools[name] = tr
        return pol

    def violations(self, t: Trace) -> list[tuple[int | None, str]]:
        steps = t.steps
        out: list[tuple[int | None, str]] = []
        counts: dict[str, int] = {}
        for i, st in enumerate(steps):
            counts[st.tool] = counts.get(st.tool, 0) + 1
            where = f"step {st.index}"
            if self.allow is not None and st.tool not in self.allow:
                out.append((st.index, f"{where}: tool {st.tool!r} is not in the allow list"))
            too_long = self.max_arg_chars is not None and len(st.arguments) > self.max_arg_chars
            if too_long:
                out.append((st.index, f"{where}: {st.tool} arguments are {len(st.arguments)} chars > max_arg_chars {self.max_arg_chars}"))
            else:  # all of it: a cap would let a long argument hide a denied path at its end
                for rx in self.deny:
                    if rx.search(st.arguments):
                        out.append((st.index, f"{where}: {st.tool} arguments match denied pattern {rx.pattern!r}"))
                        break
            rule = self.tools.get(st.tool)
            if rule is None:
                continue
            if rule.max_calls is not None and counts[st.tool] == rule.max_calls + 1:
                out.append((st.index, f"{where}: {st.tool} called {counts[st.tool]}x > max_calls {rule.max_calls}"))
            args = arguments_of(st) or {}
            for path, mx in rule.limits.items():
                ok, v = get_path(args, path)
                if ok and _as_float(v) is not None and _as_float(v) > mx:
                    out.append((st.index, f"{where}: {st.tool}.{path} {_val(v)} > max {_fmt(mx)}"))
            for path, allowed in rule.allowed_values.items():
                ok, v = get_path(args, path)
                if ok and v not in allowed:
                    out.append((st.index, f"{where}: {st.tool}.{path} {_val(v, True)} is not one of {_val(allowed, True)}"))
            for path in rule.required_args:
                if not get_path(args, path)[0]:
                    out.append((st.index, f"{where}: {st.tool} is missing required argument {path!r}"))
            if rule.identity_arg:
                ok, v = get_path(args, rule.identity_arg)
                if t.user_id is None:
                    out.append((st.index, f"{where}: {st.tool}.{rule.identity_arg}: no user id for identity check (set fz.run(user_id=...))"))
                elif not ok or str(v) != str(t.user_id):
                    out.append((st.index, f"{where}: {st.tool}.{rule.identity_arg} {_val(v, True)} does not match the run's user id {_val(t.user_id, True)}"))
            for req in rule.requires:
                if not any(p.tool == req and p.result_seen and error_of(p) is None for p in steps[:i]):
                    out.append((st.index, f"{where}: {st.tool} called before a successful {req}"))
        if self.max_calls is not None and len(steps) > self.max_calls:
            out.append((None, f"{len(steps)} tool calls > max_calls {self.max_calls}"))
        if self.require_complete:
            if not t.answer.strip():
                out.append((None, "run incomplete: no final answer"))
            elif t.termination and t.termination != "completed":
                out.append((None, f"run incomplete: {t.termination}"))
        return out


_REPEATS = {_c.MAX_REPEAT, _c.MIN_REPEAT, getattr(_c, "POSSESSIVE_REPEAT", _c.MAX_REPEAT)}


def _children(av: Any) -> list[Any]:
    """Every sub-pattern inside a node's argument, whatever the node type (groups, atomic groups,
    lookarounds, conditionals, repeats, branches)."""
    if isinstance(av, _parser.SubPattern):
        return [av]
    if isinstance(av, (list, tuple)):
        return [s for x in av for s in _children(x)]
    return []


def _literal_alternatives(av: Any) -> bool:
    """A branch of plain literal strings with different first characters, e.g. (foo|bar): it can match a
    given text only one way."""
    alts = av[1]
    if not all(alt and all(op is _c.LITERAL for op, _ in alt) for alt in alts):
        return False
    firsts = [alt[0][1] for alt in alts]
    return len(set(firsts)) == len(firsts)


def _ambiguous(sub: Any) -> bool:
    """The pattern can match one text in more than one way: a variable-count repeat or a branch, anywhere."""
    for op, av in sub:
        if op in _REPEATS and av[0] != av[1]:
            return True
        if op is _c.BRANCH and not _literal_alternatives(av):
            return True
        if any(_ambiguous(c) for c in _children(av)):
            return True
    return False


def _risky(sub: Any) -> bool:
    for op, av in sub:
        if op in _REPEATS and av[1] > 1 and _ambiguous(av[2]):
            return True
        if any(_risky(c) for c in _children(av)):
            return True
    return False


def _can_backtrack_exponentially(pattern: str) -> bool:
    """A group repeated more than once whose body can match one text in several ways, e.g. (a+)+,
    (a|aa)+, ((a)+)+, (?>(a+)+), (?:(a+){1})+, (?:ab?)+. Character classes, single repeats and literal
    alternatives (``[ab]+``, ``\\w+``, ``(ab)+``, ``(foo|bar)+``) are fine. Unparseable patterns are left
    to re.compile. Conservative by design: refusing a safe pattern is a clear usage error, accepting an
    exponential one can hang a test run on a model-written argument."""
    try:
        return _risky(_parser.parse(pattern, re.I))
    except Exception:
        return False


def _val(v: Any, quote: bool = False, n: int = 120) -> str:
    """A value for a reason: model-written values can be any length."""
    s = repr(v) if quote else str(v)
    return s if len(s) <= n else s[: n - 1] + "…"


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _as_float(v: Any) -> float | None:
    if _num(v):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.replace(",", "").strip().lstrip("$"))
        except ValueError:
            return None
    return None


def _fmt(x: float) -> str:
    return str(int(x)) if x == int(x) else str(x)


def _str_list(v: Any, key: str, fail) -> list[str]:
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        raise fail(key, "must be a list of strings")
    return list(v)


def _pos_int(v: Any, key: str, fail) -> int:
    if not isinstance(v, int) or isinstance(v, bool) or v < 0:
        raise fail(key, "must be a non-negative integer")
    return v
