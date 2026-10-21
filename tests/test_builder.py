from faultgate_zero.trace import Span, Trace

from fakes.agents import openai_agent
from fakes.openai_fakes import Script, client, completion

import faultgate_zero as fz


def _chat(i, inp, out, t):
    return Span(f"c{i}", "r", "chat", t, t + 1, {"gen_ai.operation.name": "chat",
                                                   "gen_ai.input.messages": inp, "gen_ai.output.messages": out})


def _tool_call(cid, name, args="{}"):
    return {"type": "tool_call", "id": cid, "name": name, "arguments": args}


def test_chat_output_holds_only_that_response_and_root_holds_text_only():
    script = Script(completion(tool_calls=[("c1", "search", {"q": "a"})]),
                    completion(tool_calls=[("c2", "search", {"q": "b"})]),
                    completion("answer"))
    with fz.run() as run:
        openai_agent(client(script), "q", {"search": lambda q: [q]})
    t = run.trace
    outs = [s.attrs["gen_ai.output.messages"] for s in t.model_calls]
    assert [len(o) for o in outs] == [1, 1, 1]
    assert [p["type"] for o in outs for m in o for p in m["parts"]] == ["tool_call", "tool_call", "text"]
    root_parts = [p for m in t.root.attrs["gen_ai.output.messages"] for p in m["parts"]]
    assert root_parts == [{"type": "text", "content": "answer"}]
    # the full history is re-sent on every request, yet each tool call is one step
    assert len(t.steps) == 2


def test_run_that_ends_while_asking_for_tools_has_no_answer():
    script = Script(*[completion("thinking...", tool_calls=[(f"c{i}", "search", {"q": "x"})]) for i in range(3)])
    with fz.run() as run:
        out = openai_agent(client(script), "q", {"search": lambda q: []}, max_steps=3)
    assert out is None
    assert run.trace.answer == ""
    assert run.trace.root.attrs["gen_ai.output.messages"] == []
    assert run.trace.steps[-1].result_seen is False  # its result never reached the model


def test_results_matched_by_id_and_error_flag_carried():
    inp2 = [{"role": "tool", "parts": [{"type": "tool_call_response", "id": "b", "result": "rb"},
                                       {"type": "tool_call_response", "id": "a", "result": "denied", "is_error": True}]}]
    t = Trace("t", "s", "agent", None, [
        Span("r", None, "invoke_agent", 0, 10, {"gen_ai.operation.name": "invoke_agent"}),
        _chat(1, [], [{"role": "assistant", "parts": [_tool_call("a", "x"), _tool_call("b", "y")]}], 1),
        _chat(2, inp2, [{"role": "assistant", "parts": [{"type": "text", "content": "done"}]}], 3),
    ])
    st = t.steps
    assert [(s.tool, s.result, s.is_error) for s in st] == [("x", "denied", True), ("y", "rb", False)]
    assert t.answer == "done"


def test_tool_span_with_call_id_replaces_message_step_and_without_id_is_its_own_step():
    t = Trace("t", "s", "agent", None, [
        Span("r", None, "invoke_agent", 0, 20, {"gen_ai.operation.name": "invoke_agent"}),
        _chat(1, [], [{"role": "assistant", "parts": [_tool_call("a", "x", '{"k": 1}')]}], 1),
        Span("t1", "r", "execute_tool x", 3, 4, {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "x",
                                                 "gen_ai.tool.call.id": "a", "gen_ai.tool.call.result": "exact"}),
        Span("t2", "r", "execute_tool z", 5, 6, {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "z",
                                                 "gen_ai.tool.call.arguments": "{}"}, error=True),
    ])
    st = t.steps
    assert [(s.index, s.tool, s.result, s.result_seen, s.is_error) for s in st] == [
        (1, "x", "exact", True, False), (2, "z", "", True, True)]
    assert st[0].arguments == '{"k":1}'


def test_failed_model_call_is_not_the_answer():
    t = Trace("t", "s", "agent", None, [
        Span("r", None, "invoke_agent", 0, 20, {"gen_ai.operation.name": "invoke_agent"}),
        _chat(1, [], [{"role": "assistant", "parts": [{"type": "text", "content": "ok"}]}], 1),
        Span("c2", "r", "chat", 5, 6, {"gen_ai.operation.name": "chat"}, error=True),
    ])
    assert t.answer == "ok"
