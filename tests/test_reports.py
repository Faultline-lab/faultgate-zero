import json
import os
from pathlib import Path

import pytest

from faultgate_zero import otlp
from faultgate_zero.runner import check

TESTS = Path(__file__).parent
SRC = TESTS.parent / "src"

SUITE = '''
import faultgate_zero as fz
from fakes.agents import openai_agent
from fakes.openai_fakes import Script, client, completion

def test_loop():
    s = Script(*[completion(tool_calls=[(f"c{i}", "search", {"q": f"q{i}"})]) for i in range(6)])
    openai_agent(client(s), "find", {"search": lambda q: []}, max_steps=6)

def test_refund():
    s = Script(completion(tool_calls=[("c1", "issue_refund", {"amount": 900, "email": "jane@example.com"})]),
               completion("done"))
    with fz.run("refund", user_id="u1"):
        openai_agent(client(s), "refund", {"issue_refund": lambda **kw: {"ok": True}})

def test_clean():
    s = Script(completion("hello"))
    openai_agent(client(s), "hi", {})
'''

POLICY = {"tools": {"issue_refund": {"limits": {"amount": 500}, "allowed_values": {"email": ["ops@corp.com"]}}}}


@pytest.fixture
def tp(pytester, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(TESTS), str(SRC), os.environ.get("PYTHONPATH", "")]))
    pytester.syspathinsert(TESTS)
    pytester.makepyfile(test_suite=SUITE)
    (pytester.path / ".faultline").mkdir()
    (pytester.path / ".faultline" / "policy.json").write_text(json.dumps(POLICY))
    return pytester


def test_json_report_schema_and_redaction(tp):
    r = tp.runpytest("--fgz-json=fgz.json")
    r.assert_outcomes(passed=1, failed=2)
    doc = json.loads((tp.path / "fgz.json").read_text())
    assert set(doc) == {"tool", "version", "runs", "summary"}
    assert doc["tool"] == "faultgate-zero"
    assert set(doc["runs"][0]) == {"test", "scenario", "steps", "model_calls", "tokens", "latency_ms",
                                   "capture_complete", "skipped", "checked_by_test", "verdicts"}
    assert doc["summary"] == {"runs": 3, "failed_runs": 2, "skipped_runs": 0, "fired": {"policy": 1, "search_loop": 1}}
    refund = next(x for x in doc["runs"] if x["scenario"] == "refund")
    [pol] = [v for v in refund["verdicts"] if v["check"] == "policy"]
    assert pol["fired"] and pol["step"] == 1
    assert "jane@example.com" not in json.dumps(doc) and "[REDACTED:email]" in pol["reason"]
    r.stdout.fnmatch_lines(["faultgate-zero: JSON report written to *fgz.json"])


def test_markdown_summary(tp):
    tp.runpytest("--fgz-markdown=summary.md")
    md = (tp.path / "summary.md").read_text()
    assert md.startswith("## faultgate-zero\n\n**3 runs checked, 2 with failures**")
    assert "| test_suite.py::test_loop | test_suite.py::test_loop | search_loop | 6 |" in md
    assert "| test_suite.py::test_refund | refund | policy | 1 | step 1: issue_refund.amount 900 > max 500" in md


def test_otlp_export_round_trips_to_the_same_verdicts(tp):
    tp.runpytest("--fgz-export-otlp=runs.json", "--fgz-json=fgz.json")
    traces = otlp.load_otlp(tp.path / "runs.json")
    assert sorted(t.scenario for t in traces) == ["refund", "test_suite.py::test_clean", "test_suite.py::test_loop"]
    report = json.loads((tp.path / "fgz.json").read_text())
    from faultgate_zero.policy import Policy
    from faultgate_zero.runner import Settings

    s = Settings(policy=Policy.from_dict(POLICY))
    for t in traces:
        fired_now = sorted(v.check for v in check(t, s).fired)
        fired_then = sorted(v["check"] for x in report["runs"] if x["scenario"] == t.scenario
                            for v in x["verdicts"] if v["fired"])
        assert fired_now == fired_then, t.scenario
    raw = (tp.path / "runs.json").read_text()
    assert "jane@example.com" in raw  # the export is raw by design (opt-in, documented)


def test_xdist_workers_merge_into_one_report(tp):
    r = tp.runpytest_subprocess("-n", "2", "--fgz-json=fgz.json", "-p", "xdist")
    r.assert_outcomes(passed=1, failed=2)
    doc = json.loads((tp.path / "fgz.json").read_text())
    assert doc["summary"]["runs"] == 3 and doc["summary"]["failed_runs"] == 2
    r.stdout.fnmatch_lines(["faultgate-zero: 3 runs checked, 2 with failures"])


def test_pyproject_report_paths_are_relative_to_the_root(tp):
    (tp.path / "pyproject.toml").write_text('[tool.faultgate_zero]\njson = "reports/fgz.json"\n')
    (tp.path / "reports").mkdir()
    tp.runpytest()
    assert (tp.path / "reports" / "fgz.json").is_file()
