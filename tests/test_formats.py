"""formats.load: every trace format as faultgate-zero traces (0.2)."""
import builtins
import json
from pathlib import Path

import pytest

from faultgate_zero import formats, otlp

DATA = Path(__file__).parent / "data" / "formats"


def test_otlp_loads_without_faultgate(monkeypatch):
    real_import = builtins.__import__

    def no_faultgate(name, *a, **k):
        if name == "faultgate" or name.startswith("faultgate."):
            raise ImportError(name)
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_faultgate)
    traces = formats.load(DATA / "otlp.json")
    assert traces
    assert [t.steps for t in traces] == [t.steps for t in otlp.load_otlp(DATA / "otlp.json")]
    with pytest.raises(formats.FormatError, match=r"pip install 'faultgate-zero\[formats\]'"):
        formats.load(DATA / "langfuse.json")


@pytest.mark.parametrize("fmt", ["openinference", "langfuse", "langsmith", "braintrust", "otlp"])
def test_every_faultgate_example_format_loads(fmt):
    pytest.importorskip("faultgate")
    path = DATA / f"{fmt}.json"
    auto = formats.load(path)
    assert auto, fmt
    assert [t.trace_id for t in formats.load(path, fmt)] == [t.trace_id for t in auto]
    assert any(t.steps for t in auto), f"{fmt}: no tool steps read"


def test_converted_traces_keep_steps_and_results():
    fg = pytest.importorskip("faultgate.formats")
    path = DATA / "langsmith.json"
    zero = formats.load(path)
    assert len(zero) == len(fg.load(path))
    steps = [s for t in zero for s in t.steps]
    assert steps and all(s.tool for s in steps)
    assert any(s.result_seen for s in steps)


def test_bad_input_is_a_format_error(tmp_path):
    assert issubclass(formats.FormatError, ValueError)
    with pytest.raises(formats.FormatError, match="unknown format 'csv'"):
        formats.load(DATA / "otlp.json", "csv")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(formats.FormatError, match="not valid JSON"):
        formats.load(bad)
    with pytest.raises(formats.FormatError, match="cannot read it"):
        formats.load(tmp_path / "missing.json")
    pytest.importorskip("faultgate")
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"hello": 1}), encoding="utf-8")
    with pytest.raises(formats.FormatError, match="unrecognised trace format"):
        formats.load(other)


@pytest.mark.parametrize("fmt", ["openinference", "langfuse", "langsmith", "braintrust"])
def test_converted_traces_keep_prompt_and_answer(fmt):
    pytest.importorskip("faultgate")
    t = formats.load(DATA / f"{fmt}.json")[0]
    assert t.prompt == "Refund order o-7 for customer C-123"
    assert t.answer == "Refunded $120.00 to C-123."


def test_converted_answers_reach_the_checks(tmp_path):
    pytest.importorskip("faultgate")
    from faultgate_zero.checks import Options, abstention

    text = (DATA / "langsmith.json").read_text(encoding="utf-8")
    assert "Refunded $120.00 to C-123." in text
    f = tmp_path / "abstains.json"
    f.write_text(text.replace("Refunded $120.00 to C-123.", "I could not find that information."), encoding="utf-8")
    assert abstention(formats.load(f)[0], Options()).fired


def test_chat_message_text_becomes_parts():
    msgs = formats.normalize_messages(json.dumps([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "function": {"name": "lookup"}}]},
        {"role": "assistant", "content": [{"type": "text", "text": "done"}, {"type": "image_url"}]},
        {"role": "assistant", "parts": [{"type": "text", "content": "kept"}]},
    ]))
    assert msgs[0] == {"role": "user", "parts": [{"type": "text", "content": "hi"}]}
    assert msgs[1]["parts"] == []  # tool calls are read from parts and tool spans, as faultgate does
    assert msgs[2]["parts"] == [{"type": "text", "content": "done"}]
    assert msgs[3]["parts"] == [{"type": "text", "content": "kept"}]


@pytest.mark.parametrize("doc", [{"resourceSpans": [None]}, {"resourceSpans": [{"scopeSpans": [None]}]},
                                 {"resourceSpans": [{"scopeSpans": [{"spans": [None]}]}]},
                                 {"resourceSpans": [{"scopeSpans": [{"spans": [{"attributes": [None]}]}]}]}])
def test_malformed_otlp_is_a_format_error_in_auto_mode(tmp_path, doc):
    f = tmp_path / "bad.json"
    f.write_text(json.dumps(doc), encoding="utf-8")
    for fmt in ("auto", "otlp"):
        with pytest.raises(formats.FormatError):
            formats.load(f, fmt)
