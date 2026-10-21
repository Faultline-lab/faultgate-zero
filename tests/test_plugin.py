"""The plugin end to end, through pytester: each test is one agent run, checked when it ends."""
from pathlib import Path

import pytest

TESTS = Path(__file__).parent

AGENTS = '''
import json, pytest, openai
import faultgate_zero as fz
from fakes.agents import openai_agent
from fakes.openai_fakes import Script, client, completion, error_body

def loop_script(n=6):
    return Script(*[completion(tool_calls=[(f"c{i}", "search", {"q": f"q{i}"})]) for i in range(n)])

def run_loop():
    return openai_agent(client(loop_script()), "find", {"search": lambda q: []}, max_steps=6)

def run_clean():
    s = Script(completion(tool_calls=[("c1", "search", {"q": "a"})]), completion("found a"))
    return openai_agent(client(s), "find a", {"search": lambda q: [q]})
'''


@pytest.fixture
def tp(pytester):
    pytester.syspathinsert(TESTS)
    pytester.makepyfile(agents=AGENTS)
    return pytester


def test_clean_test_passes_and_looping_test_fails_with_step_reason(tp):
    tp.makepyfile(test_a='''
from agents import run_clean, run_loop
def test_clean():
    assert run_clean() == "found a"
def test_loop():
    assert run_loop() is None   # answer-only asserts pass today
''')
    r = tp.runpytest()
    r.assert_outcomes(passed=1, failed=1)
    r.stdout.fnmatch_lines([
        "*faultgate-zero: 1 check fired on run 'test_a.py::test_loop' (6 steps, 6 model calls)",
        "*search_loop  no answer after 6 tool calls (search); 5 of 5 results were empty",
        "*hint: mark the test with @pytest.mark.faultgate(skip=[[]...[]]) if a check is expected to fire",
        "faultgate-zero: 2 runs checked, 1 with failures",
        "  search_loop: 1",
    ])


def test_marker_skip_and_off(tp):
    tp.makepyfile(test_b='''
import pytest
from agents import run_loop
@pytest.mark.faultgate(skip=["search_loop"])
def test_skip():
    run_loop()
@pytest.mark.faultgate(off=True)
def test_off():
    run_loop()
''')
    tp.runpytest().assert_outcomes(passed=2)


def test_warn_mode_passes_and_lists_warnings(tp):
    tp.makepyfile(test_c="from agents import run_loop\ndef test_loop():\n    run_loop()\n")
    r = tp.runpytest("--fgz=warn")
    r.assert_outcomes(passed=1)
    r.stdout.fnmatch_lines(["*faultgate-zero warnings*", "test_c.py::test_loop", "*search_loop*"])


def test_off_mode_and_env(tp, monkeypatch):
    tp.makepyfile(test_d="from agents import run_loop\ndef test_loop():\n    run_loop()\n")
    tp.runpytest("--fgz=off").assert_outcomes(passed=1)
    monkeypatch.setenv("FAULTGATE_ZERO", "off")
    tp.runpytest().assert_outcomes(passed=1)


def test_already_failing_test_gets_an_extra_section(tp):
    tp.makepyfile(test_e="from agents import run_loop\ndef test_loop():\n    run_loop()\n    assert False, 'own assert'\n")
    r = tp.runpytest()
    r.assert_outcomes(failed=1)
    r.stdout.fnmatch_lines(["*own assert*", "*faultgate-zero*", "*search_loop*"])


def test_explicit_runs_are_checked_separately_and_named(tp):
    tp.makepyfile(test_f='''
import faultgate_zero as fz
from agents import run_clean, run_loop
def test_two_runs():
    with fz.run("clean-one"):
        run_clean()
    with fz.run("loop-one"):
        run_loop()
''')
    r = tp.runpytest()
    r.assert_outcomes(failed=1)
    r.stdout.fnmatch_lines(["*1 check fired on run 'loop-one'*"])
    r.stdout.fnmatch_lines(["faultgate-zero: 2 runs checked, 1 with failures"])


def test_run_the_test_checked_itself_is_left_alone(tp):
    tp.makepyfile(test_g='''
import faultgate_zero as fz
from agents import run_loop
def test_expects_the_loop_to_be_caught():
    with fz.run("loop") as run:
        run_loop()
    assert [v.check for v in run.report.fired] == ["search_loop"]
''')
    tp.runpytest().assert_outcomes(passed=1)


def test_pytest_raises_around_a_provider_error_passes(tp):
    tp.makepyfile(test_h='''
import openai, pytest
from fakes.agents import openai_agent
from fakes.openai_fakes import Script, client, error_body

class AgentError(Exception):
    pass

def agent_that_wraps():
    try:
        openai_agent(client(Script((500, error_body("down", "server_error")))), "q", {})
    except openai.InternalServerError as e:
        raise AgentError("provider down") from e

def agent_that_swallows():
    try:
        openai_agent(client(Script((500, error_body("down", "server_error")))), "q", {})
    except openai.InternalServerError:
        return "Sorry, something went wrong."

def test_expected_sdk_error():
    with pytest.raises(openai.InternalServerError):
        openai_agent(client(Script((500, error_body("down", "server_error")))), "q", {})

def test_expected_wrapped_error():
    with pytest.raises(AgentError):
        agent_that_wraps()

def test_swallowed_error_is_caught():
    assert agent_that_swallows() == "Sorry, something went wrong."
''')
    r = tp.runpytest()
    r.assert_outcomes(passed=2, failed=1)
    r.stdout.fnmatch_lines(["*test_swallowed_error_is_caught*", "*unrecovered_model_error*model call 1 of 1 raised InternalServerError*"])


def test_policy_from_pyproject_and_bad_config_is_a_usage_error(tp):
    tp.makepyfile(test_i='''
import faultgate_zero as fz
from fakes.agents import openai_agent
from fakes.openai_fakes import Script, client, completion
def test_refund():
    s = Script(completion(tool_calls=[("c1", "issue_refund", {"amount": 900})]), completion("done"))
    with fz.run("refund", user_id="u1"):
        openai_agent(client(s), "refund", {"issue_refund": lambda amount: {"ok": True}})
''')
    (tp.path / ".faultline").mkdir()
    (tp.path / ".faultline" / "policy.json").write_text('{"tools": {"issue_refund": {"limits": {"amount": 500}}}}')
    r = tp.runpytest()
    r.assert_outcomes(failed=1)
    r.stdout.fnmatch_lines(["*policy  step 1: issue_refund.amount 900 > max 500*"])
    (tp.path / "pyproject.toml").write_text('[tool.faultgate_zero]\nmode = "loud"\n')
    r2 = tp.runpytest()
    assert r2.ret == pytest.ExitCode.USAGE_ERROR
    r2.stderr.fnmatch_lines(["*faultgate-zero: faultgate-zero: mode 'loud': use one of fail, warn, off*"])


def test_async_test_records_into_the_implicit_run(tp):
    tp.makepyfile(test_j='''
import asyncio
from fakes.agents import async_openai_agent
from fakes.openai_fakes import Script, async_client, completion
def test_async_loop():
    s = Script(*[completion(tool_calls=[(f"c{i}", "search", {"q": f"q{i}"})]) for i in range(6)])
    asyncio.run(async_openai_agent(async_client(s), "find", {"search": lambda q: []}, max_steps=6))
''')
    r = tp.runpytest()
    r.assert_outcomes(failed=1)
    r.stdout.fnmatch_lines(["*search_loop*"])
