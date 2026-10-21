import asyncio

import anthropic
import pytest

import faultgate_zero as fz
from fakes import anthropic_fakes as af
from fakes.openai_fakes import Script, error_body


def test_records_a_tool_loop_with_an_is_error_result():
    script = Script(af.message("Let me check.", tool_uses=[("tu_1", "lookup_order", {"order_id": "1042"})]),
                    af.message(tool_uses=[("tu_2", "issue_refund", {"order_id": "1042", "amount": 20})]),
                    af.message("Refund failed, please contact support."))

    def refund(order_id, amount):
        raise RuntimeError("payment provider unavailable")

    with fz.run("anthropic-refund") as run:
        out = af.agent(af.client(script), "Refund order 1042", {"lookup_order": lambda order_id: {"ok": 1},
                                                                "issue_refund": refund})
    assert out == "Refund failed, please contact support."
    t = run.trace
    assert [(s.tool, s.is_error) for s in t.steps] == [("lookup_order", False), ("issue_refund", True)]
    assert t.steps[1].result == "payment provider unavailable"
    assert t.steps[0].arguments == '{"order_id":"1042"}'
    assert t.answer == "Refund failed, please contact support."
    assert t.prompt == "Refund order 1042"
    assert t.model_calls[0].attrs["gen_ai.response.finish_reasons"] == ["tool_use"]
    assert t.model_calls[0].attrs["gen_ai.input.messages"][0] == {
        "role": "system", "parts": [{"type": "text", "content": "You are a support agent."}]}
    assert t.usage == {"input_tokens": 36, "output_tokens": 18}
    assert [v.check for v in run.check().fired] == ["unrecovered_tool_error"]


def test_async_and_max_tokens_cutoff():
    script = Script(af.message("The policy says", stop="max_tokens"))

    async def main():
        async with fz.run() as run:
            await af.async_client(script).messages.create(model="m", max_tokens=5,
                                                          messages=[{"role": "user", "content": "policy?"}])
        return run

    run = asyncio.run(main())
    assert [v.check for v in run.check().fired] == ["context_overflow"]


def test_prompt_too_long_error():
    script = Script((400, {"type": "error", "error": {"type": "invalid_request_error",
                                                      "message": "prompt is too long: 210000 tokens > 200000 maximum"}}))
    def agent():
        try:
            af.client(script).messages.create(model="m", max_tokens=5, messages=[{"role": "user", "content": "x"}])
        except anthropic.BadRequestError:
            return "Sorry."  # the agent swallows it

    with fz.run() as run:
        agent()
    fired = {v.check: v.reason for v in run.check().fired}
    assert set(fired) == {"context_overflow", "unrecovered_model_error"}


def test_error_handled_by_the_test_itself_is_not_reported():
    script = Script((400, {"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}}))
    with fz.run() as run:
        try:
            af.client(script).messages.create(model="m", max_tokens=5, messages=[{"role": "user", "content": "x"}])
        except anthropic.BadRequestError:
            pass  # reached the frame that opened the run: the test dealt with it
    assert run.check().passed


def test_beta_messages_are_recorded_once():
    script = Script(af.message("hi"))
    with fz.run() as run:
        af.client(script).beta.messages.create(model="m", max_tokens=5, messages=[{"role": "user", "content": "x"}])
    assert len(run.model_calls) == 1


def test_server_tool_results_in_the_same_response():
    body = af.message("Found it.")
    body["content"] = [{"type": "server_tool_use", "id": "srv_1", "name": "web_search", "input": {"query": "x"}},
                       {"type": "web_search_tool_result", "tool_use_id": "srv_1",
                        "content": {"type": "web_search_tool_result_error", "error_code": "unavailable"}},
                       {"type": "text", "text": "Found it."}]
    with fz.run() as run:
        af.client(Script(body)).messages.create(model="m", max_tokens=5, messages=[{"role": "user", "content": "x"}])
    [st] = run.steps
    assert st.tool == "web_search" and st.is_error and st.result_seen
