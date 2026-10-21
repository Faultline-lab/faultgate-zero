"""Install and remove the SDK wrappers.

Rules (spec v1 section 4): a wrapper never changes what the SDK returns or raises; it records only
while a run is open; ``install()`` is idempotent; it wraps whatever is on the class now (another
instrumentor's wrapper included); an SDK is patched only if it is importable.
"""
from __future__ import annotations

import contextvars
import functools
import importlib
import importlib.abc
import importlib.util
import inspect
import sys
import threading
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
    starter: str = "start"  # the adapter function that builds the ModelCall

    @property
    def path(self) -> str:
        return ".".join(x for x in (self.module, self.owner, self.method) if x)


TARGETS: list[Target] = [
    Target("openai.resources.chat.completions", "Completions", "create", False, "openai_chat"),
    Target("openai.resources.chat.completions", "AsyncCompletions", "create", True, "openai_chat"),
    Target("anthropic.resources.messages", "Messages", "create", False, "anthropic_messages"),
    Target("anthropic.resources.messages", "AsyncMessages", "create", True, "anthropic_messages"),
    Target("anthropic.resources.beta.messages", "Messages", "create", False, "anthropic_messages"),
    Target("anthropic.resources.beta.messages", "AsyncMessages", "create", True, "anthropic_messages"),
    Target("openai.resources.chat.completions", "Completions", "parse", False, "openai_chat"),
    Target("openai.resources.chat.completions", "AsyncCompletions", "parse", True, "openai_chat"),
    Target("openai.resources.responses", "Responses", "create", False, "openai_responses"),
    Target("openai.resources.responses", "AsyncResponses", "create", True, "openai_responses"),
    Target("openai.resources.responses", "Responses", "parse", False, "openai_responses"),
    Target("openai.resources.responses", "AsyncResponses", "parse", True, "openai_responses"),
    # messages.stream() posts directly instead of calling create(); it returns a context manager
    Target("anthropic.resources.messages", "Messages", "stream", False, "anthropic_messages", "start_stream_helper"),
    Target("anthropic.resources.messages", "AsyncMessages", "stream", False, "anthropic_messages", "start_stream_helper"),
    Target("anthropic.resources.beta.messages", "Messages", "stream", False, "anthropic_messages", "start_stream_helper"),
    Target("anthropic.resources.beta.messages", "AsyncMessages", "stream", False, "anthropic_messages", "start_stream_helper"),
    Target("litellm", "", "completion", False, "litellm_completion"),
    Target("litellm", "", "acompletion", True, "litellm_completion"),
]

_installed: list[tuple[Any, str, Callable[..., Any]]] = []
# Set while a recorded call runs: a patched SDK that call uses underneath (LiteLLM over the OpenAI SDK,
# parse() over create()) is not recorded a second time. The outermost call wins.
_in_call: contextvars.ContextVar[bool] = contextvars.ContextVar("faultgate_zero_in_call", default=False)
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


def _start(run: Run, adapter: Any, starter: str, args: tuple, kwargs: dict) -> ModelCall | None:
    try:
        call = getattr(adapter, starter)(args, kwargs)
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
    starter = target.starter
    # method targets get the SDK resource as args[0]; module functions (litellm) do not
    skip_self = 1 if target.owner else 0

    if target.is_async or inspect.iscoroutinefunction(original):
        @functools.wraps(original)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            run = current_run()
            if run is None or wrapper._fgz_disabled or _in_call.get():
                return await original(*args, **kwargs)
            call = _start(run, adapter, starter, args[skip_self:], kwargs)
            token = _in_call.set(True)
            try:
                result = await original(*args, **kwargs)
            except BaseException as exc:
                _end(run, adapter, call, None, exc)
                raise
            finally:
                _in_call.reset(token)
            return _end(run, adapter, call, result, None)
    else:
        @functools.wraps(original)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            run = current_run()
            if run is None or wrapper._fgz_disabled or _in_call.get():
                return original(*args, **kwargs)
            call = _start(run, adapter, starter, args[skip_self:], kwargs)
            token = _in_call.set(True)
            try:
                result = original(*args, **kwargs)
            except BaseException as exc:
                _end(run, adapter, call, None, exc)
                raise
            finally:
                _in_call.reset(token)
            return _end(run, adapter, call, result, None)

    wrapper._fgz_wrapper = True
    wrapper._fgz_original = original
    wrapper._fgz_disabled = False
    wrapper._fgz_target = target.path
    _wrappers.add(wrapper)
    return wrapper


_lock = threading.RLock()  # install, uninstall and patching on import
_generation = 0  # bumped by every uninstall()


def _warn(msg: str) -> None:
    """Warn without ever raising: with -W error a warning must not break the agent's import or call."""
    try:
        warnings.warn(msg, FaultgateZeroWarning, stacklevel=3)
    except Exception:
        pass


def _patch_one(t: Target) -> list[str]:
    top = t.module.split(".")[0]
    try:
        mod = importlib.import_module(t.module)
    except Exception as exc:
        _warn(f"faultgate-zero: cannot import {t.module} ({exc}); its calls are not captured")
        return []
    owner = getattr(mod, t.owner, None) if t.owner else mod
    # getattr_static: the plain function, also when inherited from a base class
    current = inspect.getattr_static(owner, t.method, None) if owner is not None else None
    if current is None:
        _warn(f"faultgate-zero: {t.path} not found in this {top} version; its calls are not captured")
        return []
    mine = _ours(current)
    if mine is not None:
        if mine._fgz_disabled:
            mine._fgz_disabled = False
            _installed.append((owner, t.method, mine))
        return [t.path]
    wrapper = _wrap(current, t)
    wrapper._fgz_inherited = t.method not in owner.__dict__
    setattr(owner, t.method, wrapper)
    _installed.append((owner, t.method, wrapper))
    return [t.path]


# SDKs not imported yet are patched the moment the agent's code imports them. install() never imports
# an SDK itself: importing one can have side effects of its own (LiteLLM fetches a model price list).
_pending: dict[str, list[Target]] = {}


class _PatchingLoader(importlib.abc.Loader):
    def __init__(self, inner: Any, name: str) -> None:
        self._inner, self._name = inner, name

    def create_module(self, spec: Any) -> Any:
        return self._inner.create_module(spec)

    def exec_module(self, module: Any) -> None:
        self._inner.exec_module(module)
        with _lock:  # uninstall() from another thread waits for this; a cleared _pending means it ran first
            gen = _generation
            for t in _pending.pop(self._name, []):
                if gen != _generation:
                    break
                try:
                    _patch_one(t)
                except Exception as exc:  # never break the agent's import
                    _warn(f"faultgate-zero: could not patch {t.path} ({exc}); its calls are not captured")
            if gen != _generation:  # uninstall() ran during the patching itself (same thread): honour it
                uninstall()

    def __getattr__(self, name: str) -> Any:  # resource readers and the rest come from the real loader
        return getattr(self._inner, name)


class _ImportHook(importlib.abc.MetaPathFinder):
    def __init__(self) -> None:
        self._busy = threading.local()

    def find_spec(self, name: str, path: Any = None, target: Any = None) -> Any:
        if name not in _pending or getattr(self._busy, "on", False):
            return None
        self._busy.on = True
        try:
            spec = None
            for finder in sys.meta_path:
                if finder is self or not hasattr(finder, "find_spec"):
                    continue
                spec = finder.find_spec(name, path, target)
                if spec is not None:
                    break
        finally:
            self._busy.on = False
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return None
        spec.loader = _PatchingLoader(spec.loader, name)
        return spec


_hook = _ImportHook()


def install(targets: list[Target] | None = None) -> list[str]:
    """Patch every SDK method in ``targets`` whose SDK is already imported, and the rest as soon as the
    agent's code imports their SDK. Returns the paths patched now. Safe to call twice."""
    patched: list[str] = []
    with _lock:
        for t in targets if targets is not None else TARGETS:
            top = t.module.split(".")[0]
            if top in sys.modules:
                patched += _patch_one(t)
            elif importlib.util.find_spec(top) is not None:  # a top-level find_spec does not import anything
                waiting = _pending.setdefault(top, [])
                if t not in waiting:
                    waiting.append(t)
        if _pending and _hook not in sys.meta_path:
            sys.meta_path.insert(0, _hook)
    return patched


def uninstall() -> None:
    """Restore the methods ``install()`` replaced. A wrapper that another instrumentor wrapped since is
    switched off instead, so their wrapper keeps working."""
    global _generation
    with _lock:
        _generation += 1
        _pending.clear()
        if _hook in sys.meta_path:
            sys.meta_path.remove(_hook)
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
