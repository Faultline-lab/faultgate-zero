"""Run recording. The current run lives in a ContextVar, so async tasks and threads started with a
copied context record into the run that was current when they started.

A run is opened by ``fz.run()`` (sync or async context manager) or, in pytest, by the plugin around
each test. Model and tool calls made while it is current are appended to it; nothing is sent
anywhere and nothing is written to disk.
"""
from __future__ import annotations

import contextvars
import sys
import time
import uuid
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable


class FaultgateZeroWarning(UserWarning):
    """A capture problem that did not stop the agent: the run's checks are skipped."""


@dataclass
class ModelCall:
    adapter: str
    provider: str
    api: str
    model: str
    request: dict[str, Any]
    response: Any = None
    error: BaseException | None = None
    start_ns: int = 0
    end_ns: int = 0
    streamed: bool = False


@dataclass
class ToolCall:
    name: str
    arguments: Any
    result: Any = None
    error: BaseException | None = None
    call_id: str | None = None
    start_ns: int = 0
    end_ns: int = 0


_current: contextvars.ContextVar[Run | None] = contextvars.ContextVar("faultgate_zero_run", default=None)
_listeners: list[Callable[[Run], None]] = []


def current_run() -> Run | None:
    return _current.get()


def add_listener(fn: Callable[[Run], None]) -> None:
    """Call ``fn(run)`` whenever a run closes (the pytest plugin collects runs this way)."""
    _listeners.append(fn)


def remove_listener(fn: Callable[[Run], None]) -> None:
    if fn in _listeners:
        _listeners.remove(fn)


class Run:
    """One agent run. Use as ``with fz.run("name") as run:`` or ``async with fz.run(...) as run:``."""

    def __init__(self, name: str | None = None, *, user_id: str | None = None, agent: str | None = None,
                 test: str | None = None, implicit: bool = False) -> None:
        self.id = uuid.uuid4().hex
        self.name = name or test or f"run-{self.id[:8]}"
        self.user_id = user_id
        self.agent = agent or "agent"
        self.test = test
        self.implicit = implicit
        self.model_calls: list[ModelCall] = []
        self.tool_calls: list[ToolCall] = []
        self._open_tools: list[ToolCall] = []
        self.capture_errors: list[str] = []
        self.checked = False  # set when the test checks this run itself; the plugin then leaves it alone
        # Errors that left the agent: caught by pytest.raises, or reaching the frame that opened the run.
        self.expected_errors: list[BaseException] = []
        self.owner_codes: set[Any] = set()
        self.error: BaseException | None = None
        self.start_ns = 0
        self.end_ns = 0
        self.closed = False
        self._last_ns = 0
        self._token: contextvars.Token[Run | None] | None = None
        self._trace = None

    def now(self) -> int:
        """Wall-clock nanoseconds, strictly increasing within the run (coarse clocks must not reorder steps)."""
        t = time.time_ns()
        self._last_ns = t if t > self._last_ns else self._last_ns + 1
        return self._last_ns

    @property
    def capture_complete(self) -> bool:
        return not self.capture_errors

    def mark_incomplete(self, where: str, exc: BaseException) -> None:
        msg = f"{where}: {type(exc).__name__}: {exc}"
        self.capture_errors.append(msg)
        warnings.warn(f"faultgate-zero: capture incomplete for run {self.name!r} ({msg}); its checks are skipped",
                      FaultgateZeroWarning, stacklevel=2)

    def open(self) -> Run:
        if self._token is not None:
            raise RuntimeError("this run is already open")
        self.start_ns = self.now()
        self._token = _current.set(self)
        return self

    def close(self, exc: BaseException | None = None) -> None:
        if self._token is None:
            return
        if exc is not None:  # BaseException too: a cancelled or timed-out run did not complete
            self.error = exc
        self.end_ns = self.now()
        try:
            _current.reset(self._token)
        except ValueError:  # closed from another context: leave the var as it is there
            pass
        self._token = None
        self.closed = True
        self._trace = None
        for fn in list(_listeners):
            fn(self)

    def escaped(self, exc: BaseException) -> bool:
        """True when ``exc`` left the agent: it is (or caused) the run's own error or an exception a
        ``pytest.raises`` block caught, or it reached the frame that opened the run."""
        targets = [e for e in (self.error, *self.expected_errors) if e is not None]
        if any(exc is e for t in targets for e in _chain(t)):
            return True
        tb = exc.__traceback__
        return tb is not None and tb.tb_frame.f_code in self.owner_codes

    def __enter__(self) -> Run:
        self.owner_codes.add(sys._getframe(1).f_code)
        return self.open()

    def __exit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> bool:
        self.close(exc)
        return False

    async def __aenter__(self) -> Run:
        self.owner_codes.add(sys._getframe(1).f_code)
        return self.open()

    async def __aexit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> bool:
        self.close(exc)
        return False

    @property
    def termination(self) -> str:
        if self.error is None:
            return "completed"
        return f"error: {type(self.error).__name__}: {self.error}"[:200]

    @property
    def trace(self):
        from .builder import build_trace

        if self._trace is None or not self.closed:
            self._trace = build_trace(self)
        return self._trace

    @property
    def steps(self):
        return self.trace.steps

    def check(self, *, checks: Any = None, skip: Any = (), policy: Any = None):
        """Run the checks on this run now. ``policy`` may be a Policy or a path to a policy file."""
        from .policy import Policy
        from .runner import check

        if policy is not None and not isinstance(policy, Policy):
            policy = Policy.load(policy)
        self.checked = True
        return check(self.trace, checks=checks, skip=tuple(skip), policy=policy)

    @property
    def report(self):
        """The checks' result with the session's settings (a ``RunResult``: ``passed``, ``fired``, ``verdicts``)."""
        return self.check()

    def assert_clean(self, *, checks: Any = None, skip: Any = (), policy: Any = None):
        """Raise ``AgentCheckFailed`` with step-level reasons if any check fired; return the result otherwise."""
        from .runner import AgentCheckFailed

        result = self.check(checks=checks, skip=skip, policy=policy)
        if not result.passed:
            raise AgentCheckFailed(result)
        return result

    def __repr__(self) -> str:
        return f"<faultgate_zero.Run {self.name!r} model_calls={len(self.model_calls)} tool_calls={len(self.tool_calls)}>"


def _chain(exc: BaseException, limit: int = 100):
    """``exc`` and every exception linked to it by ``__cause__``, ``__context__`` or an exception group."""
    seen: set[int] = set()
    todo = [exc]
    while todo and len(seen) < limit:
        e = todo.pop()
        if e is None or id(e) in seen:
            continue
        seen.add(id(e))
        yield e
        todo += [e.__cause__, e.__context__, *getattr(e, "exceptions", ())]


def note_expected(exc: BaseException | None) -> None:
    """Called when ``pytest.raises`` catches ``exc``: errors in its chain left the agent on purpose."""
    r = _current.get()
    if r is not None and exc is not None:
        r.expected_errors.append(exc)


def run(name: str | None = None, *, user_id: str | None = None, agent: str | None = None) -> Run:
    """Open a named agent run: ``with fz.run("refund-double-charge", user_id="cust_42") as run:``.

    ``name`` becomes the scenario key (``faultgate.scenario``), ``user_id`` the root span's
    ``enduser.id`` (read by the policy's identity rule), ``agent`` the ``gen_ai.agent.name``.
    """
    return Run(name, user_id=user_id, agent=agent)
