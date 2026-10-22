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
