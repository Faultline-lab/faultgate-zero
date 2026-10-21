import json

import pytest

import faultgate_zero as fz

from fakes.agents import openai_agent
from fakes.openai_fakes import Script, client, completion


def refund_agent_run(amount, policy=None, user="cust_42"):
    script = Script(
        completion(tool_calls=[("c1", "lookup_order", {"order_id": "1042"})]),
        completion(tool_calls=[("c2", "lookup_order", {"order_id": "1042"})]),
        completion(tool_calls=[("c3", "lookup_order", {"order_id": "1042"})]),
        completion(tool_calls=[("c4", "issue_refund", {"order_id": "1042", "amount": amount,
                                                        "email": "jane@example.com"})]),
        completion("Refund issued."),
    )
    tools = {"lookup_order": lambda order_id: {"order_id": order_id, "charges": 2},
             "issue_refund": lambda **kw: {"ok": True}}
    with fz.run("refund-double-charge", user_id=user) as run:
        openai_agent(client(script), "Refund one charge on order 1042", tools)
    return run


POLICY = {"tools": {"issue_refund": {"limits": {"amount": 500}}}}


def test_assert_clean_lists_each_fired_check_with_step_reasons(tmp_path):
    p = tmp_path / "policy.json"
    p.write_text(json.dumps(POLICY), encoding="utf-8")
    run = refund_agent_run(900)
    with pytest.raises(fz.AgentCheckFailed) as info:
        run.assert_clean(policy=p)
    assert str(info.value) == (
        "faultgate-zero: 2 checks fired on run 'refund-double-charge' (4 steps, 5 model calls)\n"
        "  repeated_call  step 3: lookup_order called 3x with identical arguments {\"order_id\":\"1042\"} and identical results\n"
        "  policy         step 4: issue_refund.amount 900 > max 500")
    assert isinstance(info.value, AssertionError)


def test_skip_and_checks_select():
    run = refund_agent_run(20)
    run.assert_clean(skip=["repeated_call"])
    assert [v.check for v in run.check(checks="repeated_call").fired] == ["repeated_call"]
    with pytest.raises(ValueError, match="unknown check"):
        run.check(skip=["repeat"])


def test_reasons_are_redacted():
    run = refund_agent_run(20)
    r = run.check(checks="policy", policy=fz.Policy.from_dict({"deny_patterns": ["@EXAMPLE.com"]}))
    assert r.fired[0].reason == "step 4: issue_refund arguments contain denied text '@EXAMPLE.com'"
    run2 = refund_agent_run(20)
    r2 = run2.check(checks="policy", policy=fz.Policy.from_dict(
        {"tools": {"issue_refund": {"allowed_values": {"email": ["ops@corp.com"]}}}}))
    assert "jane@example.com" not in r2.fired[0].reason and "[REDACTED:email]" in r2.fired[0].reason


def test_report_and_clean_run():
    script = Script(completion(tool_calls=[("c1", "lookup_order", {"order_id": "1"})]), completion("ok"))
    with fz.run("clean") as run:
        openai_agent(client(script), "q", {"lookup_order": lambda order_id: {"id": order_id}})
    assert run.report.passed
    assert str(run.assert_clean().summary()) == "faultgate-zero: run 'clean' is clean (1 steps, 2 model calls)"


def test_capture_incomplete_skips_checks(monkeypatch):
    from faultgate_zero.adapters import openai_chat

    def broken(call, result, run):
        raise RuntimeError("bug")

    monkeypatch.setattr(openai_chat, "finish", broken)
    with pytest.warns(fz.FaultgateZeroWarning):
        run = refund_agent_run(900)
    r = run.check(policy=fz.Policy.from_dict(POLICY))
    assert r.skipped == "capture incomplete" and r.passed
