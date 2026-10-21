"""Denied text is literal and case-insensitive (CTO decision after the regex reviews): exact, linear in the
argument, found wherever it sits, never fooled by regex syntax."""
import json
import time

import pytest

from faultgate_zero.policy import DEFAULT_DENY, Policy

from fakes.traces import step, trace


def msgs(t, policy=None):
    return [m for _, m in Policy.from_dict(policy or {}).violations(t)]


@pytest.mark.parametrize("args,hit", [
    ({"path": "../../secrets"}, "../"),
    ({"path": "..\\..\\boot.ini"}, "..\\"),
    ({"url": "FILE:///etc/hosts"}, "file://"),
    ({"path": "~/.SSH/config"}, "~/.ssh"),
    ({"path": "/ETC/PASSWD"}, "/etc/passwd"),
    ({"key": "home/u/ID_RSA"}, "id_rsa"),
])
def test_default_denied_text_is_found_case_insensitively(args, hit):
    assert msgs(trace([step("f", args)])) == [f"step 1: f arguments contain denied text {hit!r}"]


def test_clean_arguments_pass():
    t = trace([step("read_file", {"path": "reports/q3.txt"}), step("fetch", {"url": "https://example.com/a..b"})])
    assert msgs(t) == []


def test_found_anywhere_in_a_long_argument_and_fast():
    for at in (0, 1023, 1024, 50_000, 199_990):
        args = {"pad": "x" * at, "path": "/etc/passwd", "tail": "y" * 1000}
        assert msgs(trace([step("f", args)])) == ["step 1: f arguments contain denied text '/etc/passwd'"], at
    t = trace([step("f", {"pad": "i" * 500_000 + "!"})])
    t0 = time.perf_counter()
    Policy.from_dict({"deny_patterns": ["(a|aa)+$", "a*a*a*a*b", "\\w+$"]}).violations(t)
    assert time.perf_counter() - t0 < 0.5


def test_regex_syntax_is_matched_literally():
    pol = {"deny_patterns": ["a.b", "(x|y)+"]}
    assert msgs(trace([step("f", {"v": "axb"})]), pol) == []
    assert msgs(trace([step("f", {"v": "see a.b here"})]), pol) == ["step 1: f arguments contain denied text 'a.b'"]
    assert msgs(trace([step("f", {"v": "(x|y)+"})]), pol) == ["step 1: f arguments contain denied text '(x|y)+'"]


def test_custom_list_replaces_the_defaults():
    assert msgs(trace([step("f", {"path": "../x"})]), {"deny_patterns": ["DROP TABLE"]}) == []
    assert msgs(trace([step("q", {"sql": "drop table users"})]), {"deny_patterns": ["DROP TABLE"]}) == [
        "step 1: q arguments contain denied text 'DROP TABLE'"]
    assert DEFAULT_DENY == ("../", "..\\", "file://", "~/.ssh", "/etc/passwd", "id_rsa")


def test_json_escaped_backslashes_still_match():
    # canonical JSON escapes a backslash as two; "..\\" (dot, dot, backslash) is still a substring
    raw = json.dumps({"path": "..\\windows"})
    assert msgs(trace([step("f", raw)])) == ["step 1: f arguments contain denied text '..\\\\'"]
