import functools
import importlib.machinery
import sys
import types
import warnings

import pytest

import faultgate_zero as fz
from faultgate_zero import patch
from faultgate_zero.adapters import openai_chat
from faultgate_zero.patch import Target

SENTINEL = object()


class Boom(Exception):
    pass


@pytest.fixture
def dummy():
    """A fake SDK module with one sync and one async create method, patched through a private target list."""
    mod = types.ModuleType("fgz_dummy_sdk")
    mod.__spec__ = importlib.machinery.ModuleSpec("fgz_dummy_sdk", None)

    class Client:
        def create(self, **kw):
            if kw.get("fail"):
                raise Boom("sdk failed")
            return SENTINEL

        async def acreate(self, **kw):
            return SENTINEL

    mod.Client = Client
    sys.modules["fgz_dummy_sdk"] = mod
    targets = [Target("fgz_dummy_sdk", "Client", "create", False, "openai_chat"),
               Target("fgz_dummy_sdk", "Client", "acreate", True, "openai_chat")]
    original = Client.__dict__["create"]
    yield Client, targets, original
    for owner, name, w in list(patch._installed):
        if owner is Client:
            patch._installed.remove((owner, name, w))
    del sys.modules["fgz_dummy_sdk"]


def test_returns_the_exact_object(dummy):
    Client, targets, _ = dummy
    patch.install(targets)
    with fz.run() as run:
        assert Client().create(model="m", messages=[]) is SENTINEL
    assert run.model_calls[0].response is SENTINEL


def test_async_wrapper_returns_the_exact_object(dummy):
    import asyncio

    Client, targets, _ = dummy
    patch.install(targets)

    async def go():
        async with fz.run() as run:
            return await Client().acreate(model="m"), run

    out, run = asyncio.run(go())
    assert out is SENTINEL and len(run.model_calls) == 1


def test_exception_type_and_traceback_kept(dummy):
    Client, targets, _ = dummy
    patch.install(targets)
    with fz.run() as run:
        with pytest.raises(Boom, match="sdk failed") as info:
            Client().create(fail=True)
    frames = []
    tb = info.value.__traceback__
    while tb:
        frames.append(tb.tb_frame.f_code.co_name)
        tb = tb.tb_next
    assert "create" in frames  # the SDK's own frame is still in the traceback
    assert isinstance(run.model_calls[0].error, Boom)


def test_install_is_idempotent_and_uninstall_restores(dummy):
    Client, targets, original = dummy
    assert patch.install(targets) == [t.path for t in targets]
    first = Client.__dict__["create"]
    patch.install(targets)
    assert Client.__dict__["create"] is first
    assert first.__wrapped__ is original
    for t in list(patch._installed):
        if t[0] is not Client:
            patch._installed.remove(t)
    patch.uninstall()
    assert Client.__dict__["create"] is original


def test_wraps_another_instrumentor_and_steps_aside_on_uninstall(dummy):
    Client, targets, original = dummy
    seen = []

    def other(fn):
        @functools.wraps(fn)
        def w(*a, **k):
            seen.append(1)
            return fn(*a, **k)
        return w

    Client.create = other(original)          # an instrumentor got there first
    patch.install(targets)
    ours = Client.__dict__["create"]
    Client.create = other(ours)              # and another one wraps us afterwards
    patch.install(targets)                   # still idempotent: finds us in the chain
    assert Client.__dict__["create"].__wrapped__ is ours
    with fz.run() as run:
        assert Client().create() is SENTINEL
    assert len(run.model_calls) == 1 and len(seen) == 2
    for t in list(patch._installed):
        if t[0] is not Client:
            patch._installed.remove(t)
    patch.uninstall()                        # cannot unhook from under the outer wrapper: switch off instead
    with fz.run() as run2:
        assert Client().create() is SENTINEL
    assert run2.model_calls == [] and len(seen) == 4
    patch.install(targets)                   # switching back on
    with fz.run() as run3:
        Client().create()
    assert len(run3.model_calls) == 1


def test_internal_error_never_breaks_the_call(dummy, monkeypatch):
    Client, targets, _ = dummy
    patch.install(targets)

    def broken(args, kwargs):
        raise RuntimeError("zero bug")

    monkeypatch.setattr(openai_chat, "start", broken)
    with fz.run() as run:
        with pytest.warns(fz.FaultgateZeroWarning, match="capture incomplete"):
            assert Client().create() is SENTINEL
    assert run.capture_complete is False
    assert run.trace.capture_complete is False
    assert "zero bug" in run.capture_errors[0]


def test_internal_error_in_finish_never_breaks_the_call(dummy, monkeypatch):
    Client, targets, _ = dummy
    patch.install(targets)
    monkeypatch.setattr(openai_chat, "finish", lambda call, result, run: 1 / 0)
    with fz.run() as run:
        with pytest.warns(fz.FaultgateZeroWarning):
            assert Client().create() is SENTINEL
    assert not run.capture_complete


def test_missing_sdk_is_skipped_silently_and_missing_method_warns():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert patch.install([Target("no_such_sdk_fgz.x", "C", "m", False, "openai_chat")]) == []
    with pytest.warns(fz.FaultgateZeroWarning, match="not found"):
        assert patch.install([Target("json", "JSONDecoder", "no_such_method", False, "openai_chat")]) == []


def test_records_only_inside_a_run(dummy):
    Client, targets, _ = dummy
    patch.install(targets)
    Client().create()  # no run open: nothing to record into, nothing breaks
    with fz.run() as run:
        Client().create()
    assert len(run.model_calls) == 1
