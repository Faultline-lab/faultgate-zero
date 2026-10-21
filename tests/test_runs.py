import asyncio
import threading
import time
import contextvars

import faultgate_zero as fz
from faultgate_zero.builder import build_trace
from faultgate_zero.recorder import ModelCall

from fakes.openai_fakes import Script, async_client, client, completion


def test_two_async_tasks_in_their_own_runs_never_mix():
    async def agent(name, model, n):
        c = async_client(Script(*[completion("x", model=model) for _ in range(n)]))
        async with fz.run(name) as run:
            for _ in range(n):
                await c.chat.completions.create(model=model, messages=[{"role": "user", "content": name}])
                await asyncio.sleep(0)
        return run

    async def main():
        return await asyncio.gather(agent("a", "model-a", 4), agent("b", "model-b", 3))

    ra, rb = asyncio.run(main())
    assert [c.model for c in ra.model_calls] == ["model-a"] * 4
    assert [c.model for c in rb.model_calls] == ["model-b"] * 3


def test_nested_run_takes_the_calls_made_inside_it():
    c = client(Script(completion("1"), completion("2"), completion("3")))
    msgs = [{"role": "user", "content": "x"}]
    with fz.run("outer") as outer:
        c.chat.completions.create(model="m", messages=msgs)
        with fz.run("inner") as inner:
            c.chat.completions.create(model="m", messages=msgs)
        c.chat.completions.create(model="m", messages=msgs)
    assert len(outer.model_calls) == 2 and len(inner.model_calls) == 1
    assert fz.current_run() is None


def test_thread_with_copied_context_records_into_the_run():
    c = client(Script(completion("1"), completion("2")))
    msgs = [{"role": "user", "content": "x"}]
    with fz.run() as run:
        ctx = contextvars.copy_context()
        t = threading.Thread(target=ctx.run, args=(c.chat.completions.create,), kwargs={"model": "m", "messages": msgs})
        t.start()
        t.join()
        plain = threading.Thread(target=c.chat.completions.create, kwargs={"model": "m", "messages": msgs})
        plain.start()
        plain.join()
    assert len(run.model_calls) == 1  # a plain thread does not inherit the run (documented)


def test_run_names_and_scenario():
    with fz.run("refund-double-charge", user_id="u1", agent="support") as run:
        pass
    t = run.trace
    assert t.scenario == "refund-double-charge"
    assert t.root.attrs["gen_ai.agent.name"] == "support"
    assert t.root.attrs["enduser.id"] == "u1"
    assert "gen_ai.input.messages" not in t.root.attrs  # left out, not None


def test_overhead_of_capture_and_build_is_under_2ms_per_call():
    """Wrapper bookkeeping plus trace build and step extraction, per model call, against prebuilt responses."""
    import openai

    resp = openai.types.chat.ChatCompletion.model_validate(
        completion(tool_calls=[("c1", "search", {"q": "x"})]))
    from faultgate_zero.patch import _end, _start
    from faultgate_zero.adapters import openai_chat

    n = 300
    msgs = [{"role": "user", "content": "x" * 200}] + [
        {"role": "tool", "tool_call_id": f"c{i}", "content": "r" * 100} for i in range(20)]
    t0 = time.perf_counter()
    with fz.run() as run:
        for _ in range(n):
            call = _start(run, openai_chat, "start", (), {"model": "m", "messages": msgs})
            _end(run, openai_chat, call, resp, None)
    trace = build_trace(run)
    steps = trace.steps
    per_call_ms = (time.perf_counter() - t0) * 1000 / n
    assert len(steps) == n
    assert per_call_ms < 2.0, per_call_ms


def test_cancelled_or_timed_out_run_is_not_completed():
    async def slow_agent():
        async with fz.run("slow") as run:
            holder.append(run)
            await asyncio.sleep(10)

    holder = []

    async def main():
        try:
            await asyncio.wait_for(slow_agent(), timeout=0.01)
        except asyncio.TimeoutError:
            pass

    asyncio.run(main())
    t = holder[0].trace
    assert t.termination.startswith("error: CancelledError")
    assert t.root.error
