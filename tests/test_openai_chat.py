import asyncio
import json

import openai
import pytest

import faultgate_zero as fz
from fakes.agents import async_openai_agent, openai_agent
from fakes.openai_fakes import Script, async_client, client, completion, error_body


def test_records_a_tool_loop_through_the_real_sdk():
    script = Script(
        completion(tool_calls=[("call_1", "lookup_order", {"order_id": "1042"})]),
        completion(tool_calls=[("call_2", "issue_refund", {"order_id": "1042", "amount": 20})]),
        completion("Refunded $20 for order 1042."),
    )
    tools = {"lookup_order": lambda order_id: {"order_id": order_id, "charges": 2},
             "issue_refund": lambda order_id, amount: {"ok": True}}
    with fz.run("refund", user_id="cust_42") as run:
        answer = openai_agent(client(script), "Refund one charge on order 1042", tools)

    assert answer == "Refunded $20 for order 1042."
    t = run.trace
    assert [s.tool for s in t.steps] == ["lookup_order", "issue_refund"]
    assert t.steps[0].arguments == '{"order_id":"1042"}'
    assert json.loads(t.steps[0].result) == {"order_id": "1042", "charges": 2}
    assert all(s.result_seen for s in t.steps)
    assert t.answer == "Refunded $20 for order 1042."
    assert t.prompt == "Refund one charge on order 1042"
    assert t.scenario == "refund" and t.user_id == "cust_42"
    assert t.termination == "completed"
    assert t.usage == {"input_tokens": 30, "output_tokens": 15}
    assert len(t.model_calls) == 3
    assert t.model_calls[0].attrs["gen_ai.request.model"] == "gpt-test"
    assert t.model_calls[0].attrs["gen_ai.response.finish_reasons"] == ["tool_calls"]


def test_async_client_is_recorded():
    script = Script(completion(tool_calls=[("c1", "search", {"q": "x"})]), completion("done"))

    async def main():
        async with fz.run("async") as run:
            out = await async_openai_agent(async_client(script), "find x", {"search": lambda q: []})
        return run, out

    run, out = asyncio.run(main())
    assert out == "done"
    assert [s.tool for s in run.steps] == ["search"]
    assert run.steps[0].result == "[]"


def test_returns_the_sdk_object_and_records_nothing_outside_a_run():
    script = Script(completion("hi"), completion("hi again"))
    c = client(script)
    r = c.chat.completions.create(model="gpt-test", messages=[{"role": "user", "content": "x"}])
    assert isinstance(r, openai.types.chat.ChatCompletion)
    with fz.run() as run:
        r2 = c.chat.completions.create(model="gpt-test", messages=[{"role": "user", "content": "x"}])
    assert isinstance(r2, openai.types.chat.ChatCompletion)
    assert run.model_calls[0].response is r2
    assert len(run.model_calls) == 1


def test_sdk_error_keeps_type_and_is_recorded_on_the_chat_span():
    script = Script((400, error_body("This model's maximum context length is 8192 tokens", code="context_length_exceeded")))
    c = client(script)
    with fz.run() as run:
        with pytest.raises(openai.BadRequestError) as info:
            c.chat.completions.create(model="gpt-test", messages=[{"role": "user", "content": "x"}])
    assert info.value.__traceback__ is not None
    chat = run.trace.model_calls[0]
    assert chat.error and chat.attrs["error.type"] == "BadRequestError"
    assert "maximum context length" in chat.status_message
    assert run.trace.termination == "completed"  # the error did not leave the run


def test_exception_leaving_the_run_sets_termination_and_root_error():
    script = Script((500, error_body("boom", "server_error")))
    with pytest.raises(openai.InternalServerError):
        with fz.run() as run:
            client(script).chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}])
    assert run.trace.termination.startswith("error: InternalServerError: ")
    assert run.trace.root.error


def test_pydantic_message_objects_in_history_are_normalised():
    script = Script(completion(tool_calls=[("c1", "search", {"q": "a"})]), completion("ok"))
    c = client(script)
    with fz.run() as run:
        msgs = [{"role": "user", "content": [{"type": "text", "text": "find a"}]}]
        r = c.chat.completions.create(model="m", messages=msgs)
        msgs.append(r.choices[0].message)  # the SDK object itself, as many agents do
        msgs.append({"role": "tool", "tool_call_id": "c1", "content": [{"type": "text", "text": "found"}]})
        c.chat.completions.create(model="m", messages=msgs)
    assert run.trace.prompt == "find a"
    assert [(s.tool, s.result) for s in run.steps] == [("search", "found")]
    assert run.trace.answer == "ok"


def test_refusal_is_the_answer():
    body = completion(None)
    body["choices"][0]["message"]["refusal"] = "I can't help with that."
    with fz.run() as run:
        client(Script(body)).chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}])
    assert run.trace.answer == "I can't help with that."


def test_messages_list_mutated_after_the_call_does_not_change_the_record():
    script = Script(completion("a"))
    msgs = [{"role": "user", "content": "first"}]
    with fz.run() as run:
        client(script).chat.completions.create(model="m", messages=msgs)
        msgs.append({"role": "user", "content": "later"})
    assert len(run.model_calls[0].request["messages"]) == 1
