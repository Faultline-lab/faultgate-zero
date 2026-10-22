import json

import openai
import pytest

import faultgate_zero as fz
from fakes.openai_fakes import SSE, Script, client


def response(output, status="completed", incomplete=None, usage=(10, 5)):
    return {"id": "resp_1", "object": "response", "created_at": 0, "model": "gpt-test", "status": status,
            "output": output, "incomplete_details": {"reason": incomplete} if incomplete else None,
            "usage": {"input_tokens": usage[0], "output_tokens": usage[1], "total_tokens": sum(usage),
                      "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}},
            "parallel_tool_calls": True, "tool_choice": "auto", "tools": [], "error": None, "instructions": None,
            "metadata": {}, "temperature": 1.0, "top_p": 1.0, "text": {"format": {"type": "text"}}}


def fcall(call_id, name, args):
    return {"type": "function_call", "id": f"fc_{call_id}", "call_id": call_id, "name": name,
            "arguments": json.dumps(args), "status": "completed"}


def text(t):
    return {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": t, "annotations": []}]}


def agent(c, prompt, tools, max_steps=8):
    """A Responses API tool loop, as the OpenAI Agents SDK runs one (input grows with each turn)."""
    items = [{"role": "user", "content": prompt}]
    for _ in range(max_steps):
        r = c.responses.create(model="gpt-test", instructions="You are a support agent.", input=items)
        calls = [o for o in r.output if o.type == "function_call"]
        if not calls:
            return r.output_text
        for o in calls:
            items.append({"type": "function_call", "call_id": o.call_id, "name": o.name, "arguments": o.arguments})
            items.append({"type": "function_call_output", "call_id": o.call_id,
                          "output": json.dumps(tools[o.name](**json.loads(o.arguments)))})
    return None


def test_responses_tool_loop():
    script = Script(response([fcall("call_1", "lookup_order", {"order_id": "1042"})]),
                    response([text("Order 1042 shipped.")]))
    with fz.run("responses") as run:
        out = agent(client(script), "Where is order 1042?", {"lookup_order": lambda order_id: {"status": "shipped"}})
    assert out == "Order 1042 shipped."
    t = run.trace
    assert [(s.tool, s.arguments, s.result) for s in t.steps] == [
        ("lookup_order", '{"order_id":"1042"}', '{"status": "shipped"}')]
    assert t.answer == "Order 1042 shipped." and t.prompt == "Where is order 1042?"
    assert t.usage == {"input_tokens": 20, "output_tokens": 10}
    assert run.check().passed


def test_responses_loop_and_cutoff_fire():
    script = Script(*[response([fcall(f"c{i}", "search", {"q": f"q{i}"})]) for i in range(6)])
    with fz.run() as run:
        agent(client(script), "find", {"search": lambda q: []}, max_steps=6)
    assert [v.check for v in run.check().fired] == ["search_loop"]
    with fz.run() as run2:
        client(Script(response([text("The policy says")], status="incomplete", incomplete="max_output_tokens"))) \
            .responses.create(model="m", input="policy?")
    assert [v.check for v in run2.check().fired] == ["context_overflow"]


def test_responses_stream_uses_the_completed_event():
    final = response([text("streamed answer")])
    events = [("response.created", {"type": "response.created", "sequence_number": 0,
                                    "response": {**final, "status": "in_progress", "output": []}}),
              ("response.output_text.delta", {"type": "response.output_text.delta", "sequence_number": 1,
                                              "item_id": "msg_1", "output_index": 0, "content_index": 0,
                                              "delta": "streamed answer", "logprobs": []}),
              ("response.completed", {"type": "response.completed", "sequence_number": 2, "response": final})]
    with fz.run() as run:
        s = client(Script(SSE(events))).responses.create(model="m", input="x", stream=True)
        assert isinstance(s, openai.Stream)
        for _ in s:
            pass
    assert len(run.model_calls) == 1 and run.trace.answer == "streamed answer"


@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("recovered", [False, True])
def test_failed_response_is_checked_without_changing_sdk_behavior(streamed, recovered):
    failed = {**response([], status="failed"),
              "error": {"code": "server_error", "message": "Provider unavailable"}}
    payload = SSE([("response.failed", {"type": "response.failed", "sequence_number": 0,
                                        "response": failed})]) if streamed else failed
    c = client(Script(payload, response([text("recovered")])))
    with fz.run() as run:
        result = c.responses.create(model="m", input="x", stream=streamed)
        if streamed:
            assert [event.type for event in result] == ["response.failed"]
        else:
            assert result.status == "failed"
        if recovered:
            assert c.responses.create(model="m", input="retry").output_text == "recovered"
    assert run.capture_complete
    assert run.trace.model_calls[0].error
    assert bool(run.check().fired) is not recovered
    if not recovered:
        assert run.check().fired[0].check == "unrecovered_model_error"


def test_responses_stream_keeps_tool_arguments_before_final_event():
    events = [
        ("response.output_item.added", {"type": "response.output_item.added", "sequence_number": 0,
          "output_index": 0, "item": {**fcall("c1", "read_file", {}), "arguments": ""}}),
        ("response.function_call_arguments.delta", {"type": "response.function_call_arguments.delta",
          "sequence_number": 1, "output_index": 0, "item_id": "fc_c1", "delta": '{"path":"/etc/passwd"}'}),
    ]
    with fz.run() as run:
        stream = client(Script(SSE(events))).responses.create(model="m", input="x", stream=True)
        for event in stream:
            if event.type == "response.function_call_arguments.delta":
                break
    [step] = run.steps
    assert step.tool == "read_file" and json.loads(step.arguments) == {"path": "/etc/passwd"}
    assert run.check().fired[0].check == "policy"


def test_responses_stream_keeps_partial_text():
    events = [("response.output_text.delta", {"type": "response.output_text.delta", "sequence_number": 0,
               "item_id": "msg_1", "output_index": 0, "content_index": 0, "delta": "Hello", "logprobs": []})]
    with fz.run() as run:
        stream = client(Script(SSE(events))).responses.create(model="m", input="x", stream=True)
        next(stream)
    assert run.trace.answer == "Hello"
