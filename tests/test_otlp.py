import json
import re

import faultgate_zero as fz
from faultgate_zero import otlp

from fakes.agents import openai_agent
from fakes.openai_fakes import Script, client, completion, error_body


def _snapshot(t):
    return {"scenario": t.scenario, "agent": t.agent, "user": t.user_id, "answer": t.answer, "prompt": t.prompt,
            "termination": t.termination, "usage": t.usage, "capture": t.capture_complete,
            "steps": [(s.index, s.tool, s.call_id, s.arguments, s.result, s.is_error, s.result_seen) for s in t.steps],
            "errors": [(s.name, s.error, s.status_message) for s in t.spans]}


def test_export_and_load_round_trip(tmp_path):
    script = Script(completion(tool_calls=[("c1", "search", {"q": "a"})]),
                    (429, error_body("rate limited", "rate_limit_error")),
                    completion("found it"))
    c = client(script)
    with fz.run("rt", user_id="u7", agent="bot") as run:
        try:
            openai_agent(c, "find a", {"search": lambda q: {"hits": []}})
        except Exception:
            openai_agent(c, "find a", {"search": lambda q: {"hits": []}})
    path = tmp_path / "runs.json"
    otlp.write([run.trace], path)
    [loaded] = otlp.load_otlp(path)
    assert _snapshot(loaded) == _snapshot(run.trace)


def test_ids_and_shape_are_valid_otlp():
    with fz.run() as run:
        client(Script(completion("x"))).chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}])
    doc = otlp.export([run.trace])
    spans = doc["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert len(spans) == 2
    for s in spans:
        assert re.fullmatch(r"[0-9a-f]{32}", s["traceId"]) and re.fullmatch(r"[0-9a-f]{16}", s["spanId"])
    attrs = {a["key"]: a["value"] for a in spans[1]["attributes"]}
    assert "stringValue" in attrs["gen_ai.input.messages"]
    json.loads(attrs["gen_ai.input.messages"]["stringValue"])
    assert attrs["gen_ai.response.finish_reasons"] == {"arrayValue": {"values": [{"stringValue": "stop"}]}}
    assert attrs["gen_ai.usage.input_tokens"] == {"intValue": "10"}
