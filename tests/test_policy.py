import json

import pytest

from faultgate_zero.policy import Policy, PolicyError

from fakes.traces import step, trace

RETAIL = {
    "allow": ["lookup_order", "verify_identity", "issue_refund"],
    "max_arg_chars": 200,
    "max_calls": 6,
    "tools": {"issue_refund": {"max_calls": 1, "limits": {"amount": 500, "meta.fee": 10},
                               "allowed_values": {"currency": ["USD", "EUR"]},
                               "required_args": ["order_id", "amount"], "identity_arg": "customer_id",
                               "requires": ["verify_identity"]}},
}
OK_REFUND = {"order_id": "1042", "amount": 20, "currency": "USD", "customer_id": "cust_42"}


def msgs(t, policy=RETAIL):
    return [m for _, m in Policy.from_dict(policy).violations(t)]


def test_a_clean_run_passes():
    t = trace([step("verify_identity", {"customer_id": "cust_42"}, {"ok": True}), step("issue_refund", OK_REFUND)],
              user_id="cust_42")
    assert msgs(t) == []


@pytest.mark.parametrize("args,expected", [
    ({**OK_REFUND, "amount": 900}, "step 2: issue_refund.amount 900 > max 500"),
    ({**OK_REFUND, "amount": "$900.00"}, "step 2: issue_refund.amount '$900.00' > max 500".replace("'$900.00'", "$900.00")),
    ({**OK_REFUND, "meta": {"fee": 25}}, "step 2: issue_refund.meta.fee 25 > max 10"),
    ({**OK_REFUND, "currency": "BTC"}, "step 2: issue_refund.currency 'BTC' is not one of ['USD', 'EUR']"),
    ({k: v for k, v in OK_REFUND.items() if k != "amount"}, "step 2: issue_refund is missing required argument 'amount'"),
    ({**OK_REFUND, "customer_id": "cust_99"}, "step 2: issue_refund.customer_id 'cust_99' does not match the run's user id 'cust_42'"),
])
def test_tool_rules(args, expected):
    t = trace([step("verify_identity", {}, {"ok": True}), step("issue_refund", args)], user_id="cust_42")
    assert msgs(t) == [expected]


def test_requires_a_successful_earlier_prerequisite():
    failed_first = trace([step("verify_identity", {}, "Error: locked"), step("issue_refund", OK_REFUND)], user_id="cust_42")
    assert msgs(failed_first) == ["step 2: issue_refund called before a successful verify_identity"]
    never = trace([step("issue_refund", OK_REFUND)], user_id="cust_42")
    assert msgs(never) == ["step 1: issue_refund called before a successful verify_identity"]


def test_identity_without_a_user_id_is_a_violation():
    t = trace([step("verify_identity", {}, "ok"), step("issue_refund", OK_REFUND)])
    assert msgs(t) == ["step 2: issue_refund.customer_id: no user id for identity check (set fz.run(user_id=...))"]


def test_allow_list_arg_size_and_call_budgets():
    t = trace([step("verify_identity", {}, "ok"), step("issue_refund", OK_REFUND), step("issue_refund", OK_REFUND),
               step("delete_account", {"id": 1}), step("lookup_order", {"note": "x" * 300})]
              + [step("lookup_order", {"id": i}) for i in range(2)], user_id="cust_42")
    assert msgs(t) == [
        "step 3: issue_refund called 2x > max_calls 1",
        "step 4: tool 'delete_account' is not in the allow list",
        'step 5: lookup_order arguments are 311 chars > max_arg_chars 200',
        "7 tool calls > max_calls 6",
    ]


def test_default_deny_patterns_apply_with_no_file():
    t = trace([step("read_file", {"path": "../../etc/passwd"}), step("fetch", {"url": "file:///home/u/.ssh/id_rsa"}),
               step("read_file", {"path": "reports/q3.txt"})])
    assert msgs(t, {}) == [
        "step 1: read_file arguments contain denied text '../'",
        "step 2: fetch arguments contain denied text 'file://'",
    ]


def test_require_complete():
    pol = {"require_complete": True}
    assert msgs(trace([], answer=None), pol) == ["run incomplete: no final answer"]
    assert msgs(trace([], answer="x", termination="error: KeyError: 'a'"), pol) == ["run incomplete: error: KeyError: 'a'"]
    assert msgs(trace([], answer="x"), pol) == []


@pytest.mark.parametrize("bad,problem", [
    ({"alow": []}, "policy <dict>: alow: unknown key"),
    ({"deny_patterns": [""]}, "policy <dict>: deny_patterns: an empty string would deny every call"),
    ({"deny_patterns": "../"}, "policy <dict>: deny_patterns: must be a list of strings"),
    ({"max_calls": "5"}, "policy <dict>: max_calls: must be a non-negative integer"),
    ({"tools": {"x": {"limits": {"amount": "500"}}}}, "policy <dict>: tools.x.limits: must map argument names to numbers"),
    ({"tools": {"x": {"maxcalls": 1}}}, "policy <dict>: tools.x.maxcalls: unknown key"),
    ([], "policy <dict>: (top level): must be a JSON object"),
])
def test_bad_policies_are_usage_errors_naming_key_and_problem(bad, problem):
    with pytest.raises(PolicyError) as info:
        Policy.from_dict(bad)
    assert str(info.value).startswith(problem)


def test_load_from_file(tmp_path):
    p = tmp_path / "policy.json"
    p.write_text(json.dumps(RETAIL), encoding="utf-8")
    assert Policy.load(p).tools["issue_refund"].limits == {"amount": 500.0, "meta.fee": 10.0}
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(PolicyError, match="is not valid JSON"):
        Policy.load(p)
    with pytest.raises(PolicyError, match="not found"):
        Policy.load(tmp_path / "missing.json")
