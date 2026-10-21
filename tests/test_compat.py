"""Framework compatibility: only what passes here is named as supported in the README."""
import json

import pytest

import faultgate_zero as fz
from fakes import anthropic_fakes as af
from fakes.openai_fakes import Script, client, completion


def test_litellm_mock_response_is_recorded_once():
    litellm = pytest.importorskip("litellm")
    with fz.run() as run:
        r = litellm.completion(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}], mock_response="hello")
    assert r.choices[0].message.content == "hello"
    assert [c.adapter for c in run.model_calls] == ["litellm_completion"]
    assert run.trace.answer == "hello"


def test_litellm_over_the_openai_sdk_is_not_counted_twice():
    litellm = pytest.importorskip("litellm")
    script = Script(completion(tool_calls=[("c1", "lookup_order", {"order_id": "1"})]), completion("done"))
    c = client(script)
    with fz.run() as run:
        for _ in range(2):
            litellm.completion(model="openai/gpt-test", messages=[{"role": "user", "content": "q"}], client=c,
                               api_key="sk-test", base_url="http://fake.test/v1")
    assert len(script.requests) == 2  # both went through the (patched) OpenAI SDK underneath
    assert [c.adapter for c in run.model_calls] == ["litellm_completion", "litellm_completion"]
    assert [s.tool for s in run.steps] == ["lookup_order"]


def test_langchain_openai_tool_loop():
    pytest.importorskip("langchain_openai")
    import httpx2
    from langchain_core.messages import HumanMessage, ToolMessage
    from langchain_openai import ChatOpenAI

    script = Script(completion(tool_calls=[("c1", "lookup_order", {"order_id": "1042"})]),
                    completion("Order 1042 shipped."))
    llm = ChatOpenAI(model="gpt-test", api_key="sk-test", base_url="http://fake.test/v1", max_retries=0,
                     http_client=httpx2.Client(transport=httpx2.MockTransport(script.handler)))

    def lookup_order(order_id: str) -> str:
        """Look up an order."""
        return json.dumps({"status": "shipped"})

    bound = llm.bind_tools([lookup_order])
    with fz.run("langchain") as run:
        msgs = [HumanMessage("Where is 1042?")]
        ai = bound.invoke(msgs)
        msgs += [ai, ToolMessage(lookup_order(**ai.tool_calls[0]["args"]), tool_call_id=ai.tool_calls[0]["id"])]
        final = bound.invoke(msgs)
    assert final.content == "Order 1042 shipped."
    assert len(run.model_calls) == 2
    assert [(s.tool, s.result) for s in run.steps] == [("lookup_order", '{"status": "shipped"}')]
    assert run.trace.answer == "Order 1042 shipped."


def test_langchain_anthropic_tool_loop(monkeypatch):
    lca = pytest.importorskip("langchain_anthropic")
    import httpx2
    from langchain_core.messages import HumanMessage, ToolMessage

    script = Script(af.message(tool_uses=[("tu_1", "lookup_order", {"order_id": "1042"})]),
                    af.message("Order 1042 shipped."))
    monkeypatch.setattr(lca.chat_models, "_get_default_httpx_client",
                        lambda **kw: httpx2.Client(transport=httpx2.MockTransport(script.handler), base_url=kw["base_url"]))
    llm = lca.ChatAnthropic(model="claude-test", api_key="sk-ant-test", base_url="http://fake.test", max_retries=0)

    def lookup_order(order_id: str) -> str:
        """Look up an order."""
        return "shipped"

    bound = llm.bind_tools([lookup_order])
    with fz.run() as run:
        msgs = [HumanMessage("Where is 1042?")]
        ai = bound.invoke(msgs)
        msgs += [ai, ToolMessage("shipped", tool_call_id=ai.tool_calls[0]["id"])]
        final = bound.invoke(msgs)
    assert "shipped" in str(final.content)
    assert len(run.model_calls) == 2
    assert [(s.tool, s.result) for s in run.steps] == [("lookup_order", "shipped")]


def test_coexists_with_the_opentelemetry_openai_instrumentor():
    otel = pytest.importorskip("opentelemetry.instrumentation.openai_v2")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    inst = otel.OpenAIInstrumentor()
    fz.uninstall()
    inst.instrument(tracer_provider=provider)   # the instrumentor first, then us on top
    try:
        fz.install()
        script = Script(completion("one"), completion("two"))
        c = client(script)
        with fz.run() as run:
            r1 = c.chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}])
        assert r1.choices[0].message.content == "one"
        assert len(run.model_calls) == 1
        assert len(exporter.get_finished_spans()) == 1
        fz.uninstall()                          # and we step aside without breaking it
        c.chat.completions.create(model="m", messages=[{"role": "user", "content": "y"}])
        assert len(exporter.get_finished_spans()) == 2
    finally:
        inst.uninstrument()
        fz.install()


def test_record_model_call_for_raw_http_providers():
    with fz.run() as run:
        fz.record_model_call(messages=[{"role": "user", "content": "find x"}], model="in-house",
                             response={"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
                                 {"id": "t1", "type": "function", "function": {"name": "search", "arguments": "{}"}}]},
                                 "finish_reason": "tool_calls"}]})
        fz.record_model_call(messages=[{"role": "tool", "tool_call_id": "t1", "content": "[]"}],
                             response={"choices": [{"message": {"role": "assistant", "content": "nothing"},
                                                    "finish_reason": "stop"}]})
    assert [(s.tool, s.result) for s in run.steps] == [("search", "[]")]
    assert run.trace.answer == "nothing"
    fz.record_model_call(messages=[])  # outside a run: nothing happens
