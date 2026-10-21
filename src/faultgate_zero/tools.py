"""``@fz.tool``: record a tool call where it happens, with its exact result or exception.

The model's tool-call id is not visible to the tool, so the recorder assigns it: the oldest pending
tool call with the same name in the latest model response. That id lets the trace replace the
message-derived step instead of counting the call twice. With no match the call is its own step.
"""
from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, TypeVar

from . import adapters
from .recorder import Run, ToolCall, current_run

F = TypeVar("F", bound=Callable[..., Any])


def _match_call_id(run: Run, name: str) -> str | None:
    used = {t.call_id for t in run.tool_calls if t.call_id} | {t.call_id for t in run._open_tools if t.call_id}
    for call in reversed(run.model_calls):
        if call.response is None:
            continue
        fn = getattr(adapters.get(call.adapter), "tool_calls", None)
        if fn is None:
            return None
        for cid, tname in fn(call):
            if tname == name and cid not in used:
                return cid
        return None  # only the latest model response with a result counts
    return None


def _begin(run: Run, name: str, sig: inspect.Signature | None, args: tuple, kwargs: dict) -> ToolCall | None:
    try:
        if sig is not None:
            bound = sig.bind_partial(*args, **kwargs)
            arguments = {k: v for k, v in bound.arguments.items() if k not in ("self", "cls")}
        else:
            arguments = {"args": list(args), **kwargs}
        tc = ToolCall(name=name, arguments=arguments, call_id=_match_call_id(run, name), start_ns=run.now())
        run._open_tools.append(tc)
        return tc
    except Exception as exc:
        run.mark_incomplete(f"tool {name}", exc)
        return None


def _end(run: Run, tc: ToolCall | None, result: Any, error: BaseException | None) -> None:
    if tc is None:
        return
    try:
        tc.end_ns = run.now()
        tc.result, tc.error = result, error
        if tc in run._open_tools:
            run._open_tools.remove(tc)
        run.tool_calls.append(tc)
    except Exception as exc:
        run.mark_incomplete(f"tool {tc.name}", exc)


def tool(fn: F | None = None, *, name: str | None = None) -> Any:
    """Decorate a tool function (sync or async). ``name`` must match the tool name the model calls."""

    def decorate(f: F) -> F:
        tname = name or f.__name__
        try:
            sig: inspect.Signature | None = inspect.signature(f)
        except (TypeError, ValueError):
            sig = None

        if inspect.iscoroutinefunction(f):
            @functools.wraps(f)
            async def awrapper(*args: Any, **kwargs: Any) -> Any:
                run = current_run()
                if run is None:
                    return await f(*args, **kwargs)
                tc = _begin(run, tname, sig, args, kwargs)
                try:
                    result = await f(*args, **kwargs)
                except BaseException as exc:
                    _end(run, tc, None, exc)
                    raise
                _end(run, tc, result, None)
                return result
            return awrapper  # type: ignore[return-value]

        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            run = current_run()
            if run is None:
                return f(*args, **kwargs)
            tc = _begin(run, tname, sig, args, kwargs)
            try:
                result = f(*args, **kwargs)
            except BaseException as exc:
                _end(run, tc, None, exc)
                raise
            _end(run, tc, result, None)
            return result
        return wrapper  # type: ignore[return-value]

    return decorate(fn) if fn is not None else decorate
