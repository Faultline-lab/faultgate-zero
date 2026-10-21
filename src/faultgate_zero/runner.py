"""Run the checks on a trace and turn the verdicts into a result a test can assert on."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

from . import redact as _redact
from .checks import DEFAULT_CHECKS, Options, Verdict, resolve, run_checks
from .policy import Policy
from .trace import Trace


@dataclass
class Settings:
    """What to check and how. The pytest plugin fills one from marker, CLI, env and pyproject."""
    checks: tuple[str, ...] = DEFAULT_CHECKS
    policy: Policy = field(default_factory=Policy)
    options: Options = field(default_factory=Options)
    redact: Callable[[str], str] = _redact.builtin


_default = Settings()


def default_settings() -> Settings:
    return _default


def set_default_settings(s: Settings) -> None:
    global _default
    _default = s


class AgentCheckFailed(AssertionError):
    """Raised by ``run.assert_clean()`` when a check fired. The message lists each check and its reason."""

    def __init__(self, result: RunResult) -> None:
        self.result = result
        super().__init__(result.summary())


@dataclass
class RunResult:
    trace: Trace
    verdicts: list[Verdict]
    skipped: str | None = None

    @property
    def fired(self) -> list[Verdict]:
        return [v for v in self.verdicts if v.fired]

    @property
    def passed(self) -> bool:
        return not self.fired

    def summary(self) -> str:
        t = self.trace
        head = f"{len(t.steps)} steps, {len(t.model_calls)} model calls"
        if self.skipped:
            return f"faultgate-zero: checks skipped on run {t.scenario!r} ({self.skipped})"
        if not self.fired:
            return f"faultgate-zero: run {t.scenario!r} is clean ({head})"
        n = len(self.fired)
        lines = [f"faultgate-zero: {n} check{'s' if n > 1 else ''} fired on run {t.scenario!r} ({head})"]
        width = max(len(v.check) for v in self.fired)
        for v in self.fired:
            for j, part in enumerate(v.reason.split("; ") if v.check == "policy" else [v.reason]):
                lines.append(f"  {(v.check if j == 0 else '').ljust(width)}  {part}")
        return "\n".join(lines)


def check(trace: Trace, settings: Settings | None = None, *, checks: str | list[str] | None = None,
          skip: list[str] | tuple[str, ...] = (), policy: Policy | None = None) -> RunResult:
    s = settings or default_settings()
    names = resolve(checks) if checks is not None else s.checks
    if skip:
        resolve(list(skip))  # unknown names are a usage error, not a silent no-op
    names = tuple(n for n in names if n not in set(skip))
    if not trace.capture_complete:
        return RunResult(trace, [], skipped="capture incomplete")
    verdicts = run_checks(trace, names, policy or s.policy, s.options)
    return RunResult(trace, [replace(v, reason=s.redact(v.reason)) for v in verdicts])
