"""Regression tests for the Codex review (REJECT) of the full branch: one test per finding."""
import json
import sys
import time
import warnings

import openai
import pytest

import faultgate_zero as fz
from faultgate_zero import patch
from faultgate_zero.adapters import openai_chat
from faultgate_zero.policy import Policy, PolicyError
from faultgate_zero.redact import builtin

from fakes.agents import openai_agent
from fakes.openai_fakes import SSE, Script, chat_stream, client, completion
from fakes.traces import step, trace


# 1. a capture warning must never stop the SDK call, even with warnings as errors

def test_capture_problem_with_warnings_as_errors_still_calls_the_sdk(monkeypatch):
    script = Script(completion("hello"))
    c = client(script)
    monkeypatch.setattr(openai_chat, "start", lambda args, kwargs: 1 / 0)
    with fz.run() as run:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            r = c.chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}])
    assert r.choices[0].message.content == "hello"
    assert len(script.requests) == 1
    assert not run.capture_complete


# 2. an error raised while a stream is read is the model call's error

def test_stream_error_is_recorded_and_a_swallowed_one_fires():
    events = [(None, {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
                      "choices": [{"index": 0, "delta": {"content": "Hel"}, "finish_reason": None}]}),
              (None, {"error": {"message": "upstream overloaded", "type": "server_error"}})]

    def agent():
        try:
            for _ in client(Script(SSE(events))).chat.completions.create(
                    model="m", messages=[{"role": "user", "content": "x"}], stream=True):
                pass
        except openai.APIError:
            return "Sorry, try later."

    with fz.run() as run:
        assert agent() == "Sorry, try later."
    assert isinstance(run.model_calls[0].error, openai.APIError)
    assert "unrecovered_model_error" in [v.check for v in run.check().fired]


# 3. policy reads the arguments the decorated tool ran with, not the ones the model proposed

def test_policy_checks_executed_arguments():
    @fz.tool
    def issue_refund(order_id, amount):
        return {"ok": True}

    script = Script(completion(tool_calls=[("c1", "issue_refund", {"order_id": "1", "amount": 100})]), completion("ok"))
    with fz.run() as run:
        c = client(script)
        r = c.chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}])
        issue_refund(order_id="1", amount=900)  # the agent ran something else than the model asked for
    pol = Policy.from_dict({"tools": {"issue_refund": {"limits": {"amount": 500}}}})
    [st] = run.steps
    assert json.loads(st.arguments) == {"order_id": "1", "amount": 900}
    assert run.check(checks="policy", policy=pol).fired[0].reason == "step 1: issue_refund.amount 900 > max 500"


# 4. run.check() inside a test honours the test's marker

def test_explicit_check_uses_the_marker_policy(pytester):
    pytester.syspathinsert(str(__import__("pathlib").Path(__file__).parent))
    (pytester.path / "strict.json").write_text('{"tools": {"issue_refund": {"limits": {"amount": 500}}}}')
    pytester.makepyfile(test_m='''
import pytest, faultgate_zero as fz
from fakes.agents import openai_agent
from fakes.openai_fakes import Script, client, completion

@pytest.mark.faultgate(policy="strict.json")
def test_refund():
    s = Script(completion(tool_calls=[("c1", "issue_refund", {"amount": 900})]), completion("done"))
    with fz.run("refund") as run:
        openai_agent(client(s), "refund", {"issue_refund": lambda amount: {"ok": True}})
    assert [v.check for v in run.check().fired] == ["policy"]
''')
    r = pytester.runpytest("--fgz-json=out.json")
    r.assert_outcomes(passed=1)
    doc = json.loads((pytester.path / "out.json").read_text())
    [run] = doc["runs"]
    assert run["checked_by_test"] is True and doc["summary"]["failed_runs"] == 0


# 5. a loop the agent broke out of and resumed keeps everything it read

def test_resumed_stream_keeps_all_output():
    with fz.run() as run:
        s = client(Script(chat_stream("abcdefghijkl"))).chat.completions.create(
            model="m", messages=[{"role": "user", "content": "x"}], stream=True)
        for _ in s:
            break
        for _ in s:
            pass
    assert run.trace.answer == "abcdefghijkl"
    assert run.trace.model_calls[0].attrs["gen_ai.response.finish_reasons"] == ["stop"]


def test_stream_dropped_after_next_keeps_what_was_read():
    with fz.run() as run:
        s = client(Script(chat_stream("abcdef"))).chat.completions.create(
            model="m", messages=[{"role": "user", "content": "x"}], stream=True)
        next(s), next(s)  # role chunk and "abc", then the agent just stops reading
    assert run.trace.answer == "abc"


# 6. covered in test_checks.py::test_repeated_call_needs_three_seen_identical_results


# 7. secrets in run names and test ids are masked in summaries and reports

def test_run_name_and_capture_warning_are_masked(monkeypatch):
    script = Script(*[completion(tool_calls=[(f"c{i}", "search", {"q": f"q{i}"})]) for i in range(6)])
    with fz.run("user jane@example.com key sk-proj-abcdefghijklmnop1234") as run:
        openai_agent(client(script), "find", {"search": lambda q: []}, max_steps=6)
    summary = run.check().summary()
    assert "jane@example.com" not in summary and "sk-proj-abcdefghijklmnop1234" not in summary
    assert "[REDACTED:email]" in summary

    def broken(args, kwargs):
        raise RuntimeError("Authorization: Bearer abcdefgh12345678")

    monkeypatch.setattr(openai_chat, "start", broken)
    with fz.run() as run2:
        with pytest.warns(fz.FaultgateZeroWarning) as w:
            client(Script(completion("x"))).chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}])
    assert "abcdefgh12345678" not in str(w[0].message) and "abcdefgh12345678" not in run2.capture_errors[0]


# 8. install() never imports an SDK itself; it patches it when the agent's code imports it

def test_install_patches_on_import_without_importing(tmp_path, monkeypatch):
    pkg = tmp_path / "fgz_fake_sdk"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("IMPORTED = True\nclass Client:\n    def create(self, **kw):\n        return 'sdk'\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    t = patch.Target("fgz_fake_sdk", "Client", "create", False, "openai_chat")
    try:
        assert patch.install([t]) == []                 # not imported yet: nothing patched, nothing imported
        assert "fgz_fake_sdk" not in sys.modules
        import fgz_fake_sdk                              # the agent's code imports it
        assert fgz_fake_sdk.Client.__dict__["create"] in patch._wrappers
        with fz.run() as run:
            assert fgz_fake_sdk.Client().create(model="m", messages=[]) == "sdk"
        assert len(run.model_calls) == 1
    finally:
        for entry in list(patch._installed):
            if entry[0].__module__ == "fgz_fake_sdk":
                patch._installed.remove(entry)
        patch._pending.pop("fgz_fake_sdk", None)
        sys.modules.pop("fgz_fake_sdk", None)


# 9. no regex in the hot path can be made slow by model-written arguments

def test_nested_quantifier_deny_pattern_is_rejected():
    with pytest.raises(PolicyError, match="repeats a repeated group"):
        Policy.from_dict({"deny_patterns": ["(a+)+$"]})
    Policy.from_dict({"deny_patterns": [r"\.\./", "(ab)+", r"~/\.ssh"]})


def test_deny_patterns_skip_arguments_over_max_arg_chars():
    pol = Policy.from_dict({"max_arg_chars": 10, "deny_patterns": ["secret"]})
    t = trace([step("f", {"note": "secret " * 5})])
    assert [m for _, m in pol.violations(t)] == ["step 1: f arguments are 46 chars > max_arg_chars 10"]


def test_redaction_and_reasons_stay_fast_on_long_adversarial_text():
    t0 = time.perf_counter()
    builtin("a." * 50_000 + "!")
    builtin("x@" + "b." * 50_000 + "!")
    assert time.perf_counter() - t0 < 0.5
    pol = Policy.from_dict({"tools": {"f": {"allowed_values": {"v": ["ok"]}}}})
    [(_, msg)] = pol.violations(trace([step("f", {"v": "z" * 10_000})]))
    assert len(msg) < 300
    assert builtin("contact x@corp.example.com.") == "contact [REDACTED:email]."


def test_pytest_raises_hook_is_restored_when_the_session_ends(pytester):
    from _pytest._code.code import ExceptionInfo

    # this repo's own session runs with mode = "off", so no faultgate-zero hook may remain afterwards
    pytester.makepyfile("def test_x():\n    pass\n")
    pytester.runpytest().assert_outcomes(passed=1)
    assert not getattr(ExceptionInfo.__dict__["fill_unfilled"], "_fgz", False)
