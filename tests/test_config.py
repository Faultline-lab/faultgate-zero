import json

import pytest

from faultgate_zero.config import ConfigError, for_marker, resolve_session
from faultgate_zero.policy import PolicyError

NONE = {"mode": None, "checks": None, "policy": None, "redact": None}


def write_pyproject(root, body):
    (root / "pyproject.toml").write_text("[tool.faultgate_zero]\n" + body, encoding="utf-8")


def test_defaults(tmp_path):
    c = resolve_session(NONE, env={}, root=tmp_path)
    assert c.mode == "fail" and c.redact == "builtin" and c.policy_path is None
    assert "abstention" not in c.checks and "policy" in c.checks


def test_precedence_cli_over_env_over_pyproject(tmp_path):
    write_pyproject(tmp_path, 'mode = "off"\nchecks = ["repeated_call"]\n')
    assert resolve_session(NONE, env={}, root=tmp_path).mode == "off"
    assert resolve_session(NONE, env={"FAULTGATE_ZERO": "warn"}, root=tmp_path).mode == "warn"
    assert resolve_session({**NONE, "mode": "fail"}, env={"FAULTGATE_ZERO": "warn"}, root=tmp_path).mode == "fail"
    assert resolve_session(NONE, env={}, root=tmp_path).checks == ("repeated_call",)
    assert resolve_session({**NONE, "checks": "policy,abstention"}, env={}, root=tmp_path).checks == ("policy", "abstention")


def test_policy_discovery_and_relative_paths(tmp_path):
    (tmp_path / ".faultline").mkdir()
    (tmp_path / ".faultline" / "policy.json").write_text(json.dumps({"max_calls": 3}), encoding="utf-8")
    c = resolve_session(NONE, env={}, root=tmp_path)
    assert c.policy.max_calls == 3 and c.policy_path.endswith("policy.json")
    (tmp_path / "strict.json").write_text(json.dumps({"max_calls": 1}), encoding="utf-8")
    assert resolve_session({**NONE, "policy": "strict.json"}, env={}, root=tmp_path).policy.max_calls == 1
    assert for_marker(c, {"policy": "strict.json"}, "t").policy.max_calls == 1


@pytest.mark.parametrize("body,match", [
    ('mod = "fail"\n', "mod: unknown key"),
    ('mode = "loud"\n', "mode 'loud'"),
    ('checks = ["loops"]\n', "unknown check 'loops'"),
    ('policy = "missing.json"\n', "not found"),
    ('redact = "presidio"\n', "unknown redaction"),
    ('loop_min_steps = 0\n', "must be a positive integer"),
])
def test_usage_errors(tmp_path, body, match):
    write_pyproject(tmp_path, body)
    with pytest.raises((ConfigError, PolicyError, ValueError), match=match):
        resolve_session(NONE, env={}, root=tmp_path)


def test_marker(tmp_path):
    c = resolve_session(NONE, env={}, root=tmp_path)
    assert for_marker(c, {"off": True}, "t").mode == "off"
    m = for_marker(c, {"skip": ["repeated_call"]}, "t")
    assert "repeated_call" not in m.settings.checks and "repeated_call" in c.settings.checks
    assert for_marker(c, {"skip": "policy"}, "t").skip == ("policy",)
    with pytest.raises(ConfigError, match="unknown argument"):
        for_marker(c, {"skips": ["x"]}, "t")
    with pytest.raises(ValueError, match="unknown check"):
        for_marker(c, {"skip": ["repeat"]}, "t")
