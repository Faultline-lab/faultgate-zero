"""A small support agent that refunds double charges, written the way teams usually write one.

It has no tracing and knows nothing about Faultgate-Zero. Its tests (``test_refund_agent.py``) only
assert on the answer, as most agent tests do; Faultgate-Zero checks what the agent did on the way.
"""
from __future__ import annotations

import json
from typing import Any

import openai

ORDERS = {
    "1042": {"order_id": "1042", "customer_id": "cust_42", "charges": [20.0, 20.0], "status": "delivered"},
    "1043": {"order_id": "1043", "customer_id": "cust_43", "charges": [900.0], "status": "delivered"},
}

TOOLS = [
    {"type": "function", "function": {"name": "verify_identity", "description": "Check who the customer is.",
                                      "parameters": {"type": "object", "properties": {"customer_id": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "lookup_order", "description": "Fetch an order and its charges.",
                                      "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "search_orders", "description": "Search orders by text.",
                                      "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "issue_refund", "description": "Refund an amount on an order.",
                                      "parameters": {"type": "object", "properties": {
                                          "order_id": {"type": "string"}, "amount": {"type": "number"},
                                          "customer_id": {"type": "string"}}}}},
]


def verify_identity(customer_id: str) -> dict[str, Any]:
    return {"verified": customer_id.startswith("cust_")}


def lookup_order(order_id: str) -> dict[str, Any]:
    if order_id not in ORDERS:
        raise KeyError(f"order {order_id} not found")
    return ORDERS[order_id]


def search_orders(query: str) -> list[dict[str, Any]]:
    return [o for o in ORDERS.values() if query and query in json.dumps(o)]


def issue_refund(order_id: str, amount: float, customer_id: str = "") -> dict[str, Any]:
    return {"refunded": amount, "order_id": order_id}


TOOL_FNS = {"verify_identity": verify_identity, "lookup_order": lookup_order, "search_orders": search_orders,
            "issue_refund": issue_refund}


def run_agent(client: openai.OpenAI, prompt: str, *, max_steps: int = 8) -> str | None:
    messages: list[Any] = [
        {"role": "system", "content": "You are a support agent. Verify the customer before any refund. "
                                      "Refunds above $500 need a human."},
        {"role": "user", "content": prompt},
    ]
    for _ in range(max_steps):
        try:
            r = client.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=TOOLS)
        except openai.APIError:
            return "Sorry, something went wrong. Please try again later."  # a fallback that hides the failure
        msg = r.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls:
            return msg.content
        for tc in msg.tool_calls:
            try:
                content = json.dumps(TOOL_FNS[tc.function.name](**json.loads(tc.function.arguments)))
            except Exception as exc:
                content = f"Error: {type(exc).__name__}: {exc}"
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
    return None
