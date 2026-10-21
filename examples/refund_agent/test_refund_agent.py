"""Answer-only tests of the refund agent. Every one of them passes without Faultgate-Zero.

With Faultgate-Zero installed, the first passes and the other six fail, each naming the step and the
reason: the agent looped, repeated itself, ignored a tool error, hid a provider outage, ran out of
tokens, or broke the refund policy in .faultline/policy.json.
"""
import faultgate_zero as fz
from fake_llm import call, provider_down, say, scripted_client
from refund_agent import run_agent


def test_refunds_one_double_charge():
    llm = scripted_client(
        call(("verify_identity", {"customer_id": "cust_42"})),
        call(("lookup_order", {"order_id": "1042"})),
        call(("issue_refund", {"order_id": "1042", "amount": 20, "customer_id": "cust_42"})),
        say("I refunded one $20 charge on order 1042."),
    )
    with fz.run("refund-double-charge", user_id="cust_42"):
        answer = run_agent(llm, "I was charged twice for order 1042, please refund one")
    assert "refunded" in answer


def test_gives_up_when_the_order_cannot_be_found():
    # search_loop: six empty searches, then the agent runs out of steps without an answer
    llm = scripted_client(*[call(("search_orders", {"query": q})) for q in
                            ("10422", "order 10422", "#10422", "1O422", "10-422", "ten four two two", "10422 refund", "x")])
    answer = run_agent(llm, "Refund my double charge on order 10422")
    assert answer is None or "refund" not in answer


def test_answers_about_an_order():
    # repeated_call: the same lookup three times with the same result
    llm = scripted_client(*[call(("lookup_order", {"order_id": "1042"}))] * 3, say("Order 1042 was delivered."))
    assert "delivered" in run_agent(llm, "Where is order 1042?")


def test_explains_a_missing_order():
    # unrecovered_tool_error: the lookup fails and the agent answers anyway
    llm = scripted_client(call(("lookup_order", {"order_id": "9999"})),
                          say("Your order 9999 is on its way and will arrive tomorrow."))
    assert "order" in run_agent(llm, "Where is order 9999?")


def test_handles_a_provider_outage():
    # unrecovered_model_error: the provider is down and the agent returns a fallback message
    llm = scripted_client(provider_down())
    assert run_agent(llm, "Refund order 1042") is not None


def test_summarises_the_refund_policy():
    # context_overflow: the answer was cut at the token limit
    llm = scripted_client(say("Our refund policy allows one refund per double charge, provided that", "length"))
    assert "refund" in run_agent(llm, "What is your refund policy?")


def test_refunds_a_large_order():
    # policy: $900 is over the $500 limit, and the agent never verified the customer
    llm = scripted_client(call(("lookup_order", {"order_id": "1043"})),
                          call(("issue_refund", {"order_id": "1043", "amount": 900, "customer_id": "cust_43"})),
                          say("Done, I refunded $900."))
    with fz.run("refund-large-order", user_id="cust_43"):
        answer = run_agent(llm, "Refund order 1043")
    assert "refunded" in answer
