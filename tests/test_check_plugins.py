"""Check plugins: the ``faultgate_zero.checks`` entry point group (0.2)."""
import os
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

from faultgate_zero import checks
from faultgate_zero.checks import Options, Verdict, discover, run_plugin
from faultgate_zero.recorder import FaultgateZeroWarning
from faultgate_zero.trace import Trace

TESTS = Path(__file__).parent


class EP:
    """A stand-in for importlib.metadata.EntryPoint."""

    def __init__(self, name, obj=None, error=None):
        self.name, self.value, self._obj, self._error = name, f"fake:{name}", obj, error

    def load(self):
        if self._error:
            raise self._error
        return self._obj


def plugin(name, default=True, run=None):
    return SimpleNamespace(name=name, default=default, run=run or (lambda t, o: Verdict(name, False, "ok")))


def test_a_plugin_is_discovered():
    found, errors = discover([EP("shape", plugin("shape"))])
    assert list(found) == ["shape"] and errors == []


@pytest.mark.parametrize("name", ["policy", "search_loop", "abstention"])
def test_a_name_clash_with_a_builtin_check_is_an_error(name):
    found, errors = discover([EP("x", plugin(name))])
    assert found == {} and "clashes with a built-in check" in errors[0] and repr(name) in errors[0]


def test_two_plugins_with_one_name_and_malformed_plugins_are_errors():
    found, errors = discover([EP("a", plugin("dup")), EP("b", plugin("dup")),
                              EP("c", SimpleNamespace(name="nodefault", run=lambda t, o: None)),
                              EP("d", error=ImportError("no module"))])
    assert list(found) == ["dup"]
    assert len(errors) == 3
    assert "used by two plugins" in errors[0]
    assert "needs name (str), default (bool) and run(trace, options)" in errors[1]
    assert "cannot load it: ImportError" in errors[2]


def _trace():
    return Trace("t1", "s", "agent", None, [])


def test_a_crashing_plugin_gives_a_verdict_that_did_not_fire_and_a_warning():
    def boom(t, o):
        raise KeyError("secret@example.com")

    with pytest.warns(FaultgateZeroWarning, match="check 'shape' crashed"):
        v = run_plugin(plugin("shape", run=boom), _trace(), Options())
    assert v == Verdict("shape", False, "check crashed: KeyError")
    assert checks.crashed(v)


def test_a_plugin_returning_the_wrong_type_counts_as_a_crash():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = run_plugin(plugin("shape", run=lambda t, o: "fired"), _trace(), Options())
    assert v.reason == "check crashed: TypeError" and not v.fired


def test_the_verdict_carries_the_plugin_name():
    v = run_plugin(plugin("shape", run=lambda t, o: Verdict("other", True, "why", 2)), _trace(), Options())
    assert v == Verdict("shape", True, "why", 2)


def test_builtin_defaults_come_first_and_are_unchanged():
    assert checks.BUILTIN_DEFAULTS == ("search_loop", "repeated_call", "unrecovered_tool_error",
                                       "unrecovered_model_error", "context_overflow", "policy")
    assert checks.DEFAULT_CHECKS[:len(checks.BUILTIN_DEFAULTS)] == checks.BUILTIN_DEFAULTS


# ---- end to end: a real entry point in a dist-info folder, in a pytest subprocess

AGENTS = '''
from fakes.agents import openai_agent
from fakes.openai_fakes import Script, client, completion

def run_clean():
    s = Script(completion(tool_calls=[("c1", "search", {"q": "a"})]), completion("found a"))
    return openai_agent(client(s), "find a", {"search": lambda q: [q]})
'''

PLUGIN = '''
from faultgate_zero.checks import Verdict

class Shape:
    name = "shape"
    default = True
    def run(self, trace, options):
        bad = [s for s in trace.steps if s.tool == "search"]
        if bad:
            return Verdict("shape", True, f"step {bad[0].index} search: expected number, got string", bad[0].index)
        return Verdict("shape", False, "ok")

class Crash:
    name = "crash"
    default = True
    def run(self, trace, options):
        raise RuntimeError("boom")

class Clash:
    name = "policy"
    default = True
    def run(self, trace, options):
        return Verdict("policy", False, "")

SHAPE, CRASH, CLASH = Shape(), Crash(), Clash()
'''


def _install(pytester, monkeypatch, entries):
    pytester.makepyfile(agents=AGENTS, fakeplug=PLUGIN)
    info = pytester.path / "fakeplug-0.1.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: fakeplug\nVersion: 0.1\n", encoding="utf-8")
    lines = "\n".join(f"{n} = fakeplug:{obj}" for n, obj in entries)
    (info / "entry_points.txt").write_text(f"[faultgate_zero.checks]\n{lines}\n", encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(pytester.path), str(TESTS)]))


def test_an_installed_plugin_runs_and_can_be_skipped(pytester, monkeypatch):
    _install(pytester, monkeypatch, [("shape", "SHAPE")])
    pytester.makepyfile(test_a='''
import pytest
from agents import run_clean
def test_fires():
    assert run_clean() == "found a"
@pytest.mark.faultgate(skip=["shape"])
def test_skipped():
    assert run_clean() == "found a"
''')
    r = pytester.runpytest_subprocess()
    r.assert_outcomes(passed=1, failed=1)
    r.stdout.fnmatch_lines(["*shape  step 1 search: expected number, got string", "  shape: 1"])


def test_a_crashing_plugin_never_fails_the_users_test(pytester, monkeypatch):
    _install(pytester, monkeypatch, [("crash", "CRASH")])
    pytester.makepyfile(test_a='''
from agents import run_clean
def test_ok():
    assert run_clean() == "found a"
''')
    r = pytester.runpytest_subprocess("-W", "default")
    r.assert_outcomes(passed=1)
    r.stdout.fnmatch_lines(["faultgate-zero: 1 check crashed and did not fire*"])


def test_a_plugin_named_like_a_builtin_check_stops_the_session(pytester, monkeypatch):
    _install(pytester, monkeypatch, [("clash", "CLASH")])
    pytester.makepyfile(test_a="def test_ok():\n    assert True\n")
    r = pytester.runpytest_subprocess()
    assert r.ret == pytest.ExitCode.USAGE_ERROR
    r.stderr.fnmatch_lines(["*check plugin 'clash' (fakeplug:CLASH): name 'policy' clashes with a built-in check*"])
