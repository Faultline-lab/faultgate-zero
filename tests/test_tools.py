import asyncio
import json

import pytest

import faultgate_zero as fz
from fakes.agents import openai_agent
from fakes.openai_fakes import Script, client, completion


@fz.tool
def lookup_order(order_id: str) -> dict:
    return {"order_id": order_id, "charges": 2}


@fz.tool(name="issue_refund")
def refund(order_id: str, amount: float) -> dict:
    if amount > 100:
        raise ValueError("amount over limit")
    return {"ok": True}


def test_decorated_tool_is_never_counted_twice():
    script = Script(completion(tool_calls=[("c1", "lookup_order", {"order_id": "1"}),
                                           ("c2", "lookup_order", {"order_id": "2"})]),
                    completion("done"))
    with fz.run() as run:
        openai_agent(client(script), "q", {"lookup_order": lookup_order})
    assert [t.call_id for t in run.tool_calls] == ["c1", "c2"]
    steps = run.steps
    assert [(s.tool, s.call_id) for s in steps] == [("lookup_order", "c1"), ("lookup_order", "c2")]
    assert json.loads(steps[0].result) == {"order_id": "1", "charges": 2}


def test_decorated_tool_that_raised_is_an_error_step_and_reraises():
    script = Script(completion(tool_calls=[("c1", "issue_refund", {"order_id": "1", "amount": 900})]),
                    completion("Sorry, I could not refund."))
    with fz.run() as run:
        openai_agent(client(script), "q", {"issue_refund": refund})  # the loop catches and reports it
    [st] = run.steps
    assert st.is_error and st.result.startswith("Error: ValueError: amount over limit")
    assert [v.check for v in run.check().fired] == ["unrecovered_tool_error"]


def test_last_tool_result_is_seen_with_the_decorator():
    script = Script(*[completion(tool_calls=[(f"c{i}", "lookup_order", {"order_id": "1"})]) for i in range(2)])
    with fz.run() as run:
        openai_agent(client(script), "q", {"lookup_order": lookup_order}, max_steps=2)
    assert all(s.result_seen for s in run.steps)


def test_unmatched_decorated_tool_is_its_own_step():
    with fz.run() as run:
        lookup_order("9")  # called outside any model tool call
    [st] = run.steps
    assert st.tool == "lookup_order" and st.call_id == "" and st.arguments == '{"order_id":"9"}'


def test_name_mismatch_does_not_steal_another_tools_id():
    @fz.tool
    def search(q):
        return []

    script = Script(completion(tool_calls=[("c1", "lookup_order", {"order_id": "1"})]), completion("x"))
    with fz.run() as run:
        c = client(script)
        r = c.chat.completions.create(model="m", messages=[{"role": "user", "content": "q"}])
        search("unrelated")
    assert run.tool_calls[0].call_id is None
    assert len(run.steps) == 2


def test_async_tool_and_outside_a_run():
    @fz.tool
    async def search(q):
        return [q]

    assert asyncio.run(search("a")) == ["a"]

    async def main():
        async with fz.run() as run:
            await search("b")
        return run

    assert asyncio.run(main()).tool_calls[0].result == ["b"]
