"""Install and remove the SDK wrappers.

Rules (spec v1 section 4): a wrapper never changes what the SDK returns or raises; it records only
while a run is open; ``install()`` is idempotent; it wraps whatever is on the class now (another
instrumentor's wrapper included); an SDK is patched only if it is importable.
"""
from __future__ import annotations

import functools
import importlib
import importlib.util
import inspect
import warnings
import weakref
from dataclasses import dataclass
from typing import Any, Callable

from . import adapters
from .recorder import FaultgateZeroWarning, ModelCall, Run, current_run


@dataclass(frozen=True)
class Target:
    module: str
    owner: str
    method: str
    is_async: bool
    adapter: str

    @property
    def path(self) -> str:
        return f"{self.module}.{self.owner}.{self.method}"


TARGETS: list[Target] = [
    Target("openai.resources.chat.completions", "Completions", "create", False, "openai_chat"),
    Target("openai.resources.chat.completions", "AsyncCompletions", "create", True, "openai_chat"),
    Target("anthropic.resources.messages", "Messages", "create", False, "anthropic_messages"),
    Target("anthropic.resources.messages", "AsyncMessages", "create", True, "anthropic_messages"),
    Target("anthropic.resources.beta.messages", "Messages", "create", False, "anthropic_messages"),
    Target("anthropic.resources.beta.messages", "AsyncMessages", "create", True, "anthropic_messages"),
]

_installed: list[tuple[Any, str, Callable[..., Any]]] = []
# Ours by identity: functools.wraps copies our attributes onto any wrapper put on top of ours, and
# wrapt-style proxies forward attribute reads, so an attribute check would find their wrapper instead.
_wrappers: weakref.WeakSet[Callable[..., Any]] = weakref.WeakSet()


def _ours(fn: Any) -> Callable[..., Any] | None:
    """Our wrapper in ``fn``'s ``__wrapped__`` chain, if any (another instrumentor may sit on top)."""
    seen = 0
    while fn is not None and seen < 50:
        try:
            if fn in _wrappers:
                return fn
        except TypeError:  # not weak-referenceable, so not ours
            pass
        fn = getattr(fn, "__wrapped__", None)
        seen += 1
    return None


def _start(run: Run, adapter: Any, args: tuple, kwargs: dict) -> ModelCall | None:
    try:
        call = adapter.start(args, kwargs)
        call.start_ns = run.now()
        run.model_calls.append(call)
        return call
    except Exception as exc:  # a Zero bug must never break the agent
        run.mark_incomplete(f"{adapter.__name__}.start", exc)
        return None


def _end(run: Run, adapter: Any, call: ModelCall | None, result: Any, error: BaseException | None) -> Any:
    if call is None:
        return result
    try:
        call.end_ns = run.now()
        if error is not None:
            call.error = error
            return result
        return adapter.finish(call, result, run)
    except Exception as exc:
        run.mark_incomplete(f"{adapter.__name__}.finish", exc)
        return result


def _wrap(original: Callable[..., Any], target: Target) -> Callable[..., Any]:
    adapter = adapters.get(target.adapter)

    if target.is_async or inspect.iscoroutinefunction(original):
        @functools.wraps(original)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            run = current_run()
            if run is None or wrapper._fgz_disabled:
                return await original(*args, **kwargs)
            call = _start(run, adapter, args, kwargs)
            try:
                result = await original(*args, **kwargs)
            except BaseException as exc:
                _end(run, adapter, call, None, exc)
                raise
            return _end(run, adapter, call, result, None)
    else:
        @functools.wraps(original)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            run = current_run()
            if run is None or wrapper._fgz_disabled:
                return original(*args, **kwargs)
            call = _start(run, adapter, args, kwargs)
            try:
                result = original(*args, **kwargs)
            except BaseException as exc:
                _end(run, adapter, call, None, exc)
                raise
            return _end(run, adapter, call, result, None)

    wrapper._fgz_wrapper = True
    wrapper._fgz_original = original
    wrapper._fgz_disabled = False
    wrapper._fgz_target = target.path
    _wrappers.add(wrapper)
    return wrapper


def install(targets: list[Target] | None = None) -> list[str]:
    """Patch every importable SDK method in ``targets``. Returns the patched paths. Safe to call twice."""
    patched = []
    for t in targets if targets is not None else TARGETS:
        top = t.module.split(".")[0]
        if importlib.util.find_spec(top) is None:
            continue
        try:
            mod = importlib.import_module(t.module)
        except Exception as exc:
            warnings.warn(f"faultgate-zero: cannot import {t.module} ({exc}); its calls are not captured",
                          FaultgateZeroWarning, stacklevel=2)
            continue
        owner = getattr(mod, t.owner, None)
        # getattr_static: the plain function, also when inherited from a base class
        current = inspect.getattr_static(owner, t.method, None) if owner is not None else None
        if current is None:
            warnings.warn(f"faultgate-zero: {t.path} not found in this {top} version; its calls are not captured",
                          FaultgateZeroWarning, stacklevel=2)
            continue
        mine = _ours(current)
        if mine is not None:
            if mine._fgz_disabled:
                mine._fgz_disabled = False
                _installed.append((owner, t.method, mine))
            patched.append(t.path)
            continue
        wrapper = _wrap(current, t)
        wrapper._fgz_inherited = t.method not in owner.__dict__
        setattr(owner, t.method, wrapper)
        _installed.append((owner, t.method, wrapper))
        patched.append(t.path)
    return patched


def uninstall() -> None:
    """Restore the methods ``install()`` replaced. A wrapper that another instrumentor wrapped since is
    switched off instead, so their wrapper keeps working."""
    while _installed:
        owner, name, wrapper = _installed.pop()
        if owner.__dict__.get(name) is wrapper:
            if getattr(wrapper, "_fgz_inherited", False):
                delattr(owner, name)  # the base class's method shows through again
            else:
                setattr(owner, name, wrapper._fgz_original)
        else:
            wrapper._fgz_disabled = True


def installed() -> list[str]:
    return [w._fgz_target for _, _, w in _installed]
