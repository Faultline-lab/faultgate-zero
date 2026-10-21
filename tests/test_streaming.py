"""Streamed and non-streamed versions of the same response give identical traces (spec M4)."""
import asyncio

import anthropic
import openai

import faultgate_zero as fz
from fakes import anthropic_fakes as af
from fakes.openai_fakes import Script, async_client, chat_stream, client, completion

COMPARE = ("gen_ai.output.messages", "gen_ai.response.finish_reasons", "gen_ai.usage.input_tokens",
           "gen_ai.usage.output_tokens", "gen_ai.input.messages")


def _chat_view(run):
    return [{k: s.attrs.get(k) for k in COMPARE} for s in run.trace.model_calls]


def _openai_pair(**kw):
    msgs = [{"role": "user", "content": "find order 1042"}]
    with fz.run() as plain:
        client(Script(completion(**kw))).chat.completions.create(model="m", messages=msgs)
    with fz.run() as streamed:
        s = client(Script(chat_stream(**{("content" if k == "content" else k): v for k, v in kw.items()}))) \
            .chat.completions.create(model="m", messages=msgs, stream=True, stream_options={"include_usage": True})
        assert isinstance(s, openai.Stream)  # the SDK's own stream object comes back
        chunks = list(s)
    return plain, streamed, chunks


def test_openai_stream_with_text_matches_plain():
    plain, streamed, chunks = _openai_pair(content="Order 1042 shipped on Monday.")
    assert "".join(c.choices[0].delta.content or "" for c in chunks if c.choices) == "Order 1042 shipped on Monday."
    assert _chat_view(plain) == _chat_view(streamed)
    assert streamed.trace.answer == "Order 1042 shipped on Monday."


def test_openai_stream_with_split_tool_arguments_matches_plain():
    plain, streamed, _ = _openai_pair(tool_calls=[("c1", "lookup_order", {"order_id": "1042", "verbose": True}),
                                                  ("c2", "search", {"q": "refund policy"})])
    assert _chat_view(plain) == _chat_view(streamed)
    assert [(s.tool, s.arguments) for s in streamed.steps] == [
        ("lookup_order", '{"order_id":"1042","verbose":true}'), ("search", '{"q":"refund policy"}')]


def test_openai_stream_read_with_next_and_stopped_early_keeps_what_was_read():
    with fz.run() as run:
        s = client(Script(chat_stream("abcdefghi"))).chat.completions.create(
            model="m", messages=[{"role": "user", "content": "x"}], stream=True)
        next(s), next(s), next(s)  # role chunk, "abc", "def"
        s.close()
    assert run.trace.answer == "abcdef"
    assert run.trace.model_calls[0].attrs.get("gen_ai.response.finish_reasons") == []


def test_openai_stream_length_cutoff_fires():
    with fz.run() as run:
        for _ in client(Script(chat_stream("The refund pol", finish="length"))).chat.completions.create(
                model="m", messages=[{"role": "user", "content": "x"}], stream=True):
            pass
    assert [v.check for v in run.check().fired] == ["context_overflow"]


def test_openai_async_stream_and_stream_helper():
    async def main():
        async with fz.run() as run:
            s = await async_client(Script(chat_stream("hello"))).chat.completions.create(
                model="m", messages=[{"role": "user", "content": "x"}], stream=True)
            assert isinstance(s, openai.AsyncStream)
            async for _ in s:
                pass
        return run

    assert asyncio.run(main()).trace.answer == "hello"
    with fz.run() as run:
        with client(Script(chat_stream("helper text"))).chat.completions.stream(
                model="m", messages=[{"role": "user", "content": "x"}]) as stream:
            for _ in stream:
                pass
    assert len(run.model_calls) == 1 and run.trace.answer == "helper text"


def _anthropic_view(run):
    return [{k: s.attrs.get(k) for k in COMPARE} for s in run.trace.model_calls]


def test_anthropic_stream_matches_plain():
    kw = dict(text="Checking the order.", tool_uses=[("tu_1", "lookup_order", {"order_id": "1042"})])
    msgs = [{"role": "user", "content": "find 1042"}]
    with fz.run() as plain:
        af.client(Script(af.message(**kw))).messages.create(model="m", max_tokens=50, messages=msgs)
    with fz.run() as streamed:
        s = af.client(Script(af.message_stream(**kw))).messages.create(model="m", max_tokens=50, messages=msgs, stream=True)
        assert isinstance(s, anthropic.Stream)
        for _ in s:
            pass
    assert _anthropic_view(plain) == _anthropic_view(streamed)
    assert [s.arguments for s in streamed.steps] == ['{"order_id":"1042"}']


def test_anthropic_stream_helper_sync_and_async():
    msgs = [{"role": "user", "content": "x"}]
    with fz.run() as run:
        with af.client(Script(af.message_stream("Final words.", stop="max_tokens"))).messages.stream(
                model="m", max_tokens=5, messages=msgs) as stream:
            text = "".join(stream.text_stream)
    assert text == "Final words."
    assert len(run.model_calls) == 1 and run.trace.answer == "Final words."
    assert [v.check for v in run.check().fired] == ["context_overflow"]

    async def main():
        async with fz.run() as r:
            async with af.async_client(Script(af.message_stream("async words"))).messages.stream(
                    model="m", max_tokens=5, messages=msgs) as stream:
                await stream.get_final_message()
        return r

    assert asyncio.run(main()).trace.answer == "async words"
