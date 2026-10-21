"""One firing fixture and the clean look-alikes from spec v1.1 section C, per check, with exact reasons."""
import pytest

from faultgate_zero.checks import (DEFAULT_CHECKS, Options, abstention, context_overflow, repeated_call, resolve,
                                   search_loop, unrecovered_model_error, unrecovered_tool_error)

from fakes.traces import chat, step, trace

O = Options()


# search_loop

def test_search_loop_fires_on_no_answer_and_mostly_empty_results():
    t = trace([step("search", {"q": f"q{i}"}, []) for i in range(6)], answer=None)
    v = search_loop(t, O)
    assert v.fired and v.step == 6
    assert v.reason == "no answer after 6 tool calls (search); 6 of 6 results were empty"


@pytest.mark.parametrize("t", [
    trace([step("search", {"q": f"q{i}"}, []) for i in range(6)], answer="Nothing matched your order number."),
    trace([step("search", {"q": f"q{i}"}, {"hits": [1]}) for i in range(6)], answer=None),
    # 2 of 5 seen results empty (0.4); counting the unseen last result as empty would make it 3 of 6 (0.5)
    trace([step("search", {"q": f"h{i}"}, [i]) for i in range(3)] + [step("search", {"q": f"e{i}"}, []) for i in range(2)]
          + [step("search", {"q": "last"}, None)], answer=None),
    trace([step("search", {"q": f"q{i}"}, []) for i in range(4)], answer=None),
], ids=["answered", "results-not-empty", "unseen-results-not-counted-as-empty", "below-min-steps"])
def test_search_loop_clean_look_alikes(t):
    assert not search_loop(t, O).fired


def test_empty_result_shapes():
    for r in ["", "[]", "{}", "null", {"hits": [], "query": "x"}, {"count": 0, "items": [1]}]:
        t = trace([step("search", {"q": i}, r) for i in range(5)], answer=None)
        assert search_loop(t, O).fired, r
    t = trace([step("search", {"q": i}, {"hits": [], "note": "index rebuilding"}) for i in range(5)], answer=None)
    assert not search_loop(t, O).fired


# repeated_call

def test_repeated_call_fires_on_three_identical_calls_with_identical_results():
    t = trace([step("lookup_order", {"order_id": "1042"}, {"status": "shipped"})] * 3)
    v = repeated_call(t, O)
    assert v.fired and v.step == 3
    assert v.reason == ('step 3: lookup_order called 3x with identical arguments {"order_id":"1042"} '
                        "and identical results")


def test_repeated_call_counts_argument_order_and_spacing_as_identical():
    t = trace([step("f", '{"a": 1, "b": 2}', "x"), step("f", '{"b":2,"a":1}', "x"), step("f", '{"a":1,"b":2}', "x")])
    assert repeated_call(t, O).fired


def test_repeated_call_fires_when_the_third_result_was_never_seen():
    t = trace([step("f", {"a": 1}, "x")] * 2 + [step("f", {"a": 1}, None)], answer=None)
    assert repeated_call(t, O).fired


@pytest.mark.parametrize("t", [
    trace([step("job_status", {"id": 7}, {"state": s}) for s in ("queued", "running", "done")]),
    trace([step("lookup_order", {"order_id": n}, "x") for n in ("1", "2", "3")]),
    trace([step("lookup_order", {"order_id": "1"}, "x")] * 2),
], ids=["polling-result-changes", "different-arguments", "only-two-calls"])
def test_repeated_call_clean_look_alikes(t):
    assert not repeated_call(t, O).fired


# unrecovered_tool_error

def test_unrecovered_tool_error_fires_and_names_the_step():
    t = trace([step("lookup_order", {"id": 1}, "Error: database timeout"), step("search", {}, [1])])
    v = unrecovered_tool_error(t, O)
    assert v.fired and v.step == 1
    assert v.reason == ("step 1: lookup_order failed (result starts with 'Error:': 'Error: database timeout') "
                        "and was never called successfully again")


@pytest.mark.parametrize("result,err", [
    ("Traceback (most recent call last):\n  File x", False), ({"error": "not found"}, False),
    ({"status": "failed"}, False), ("permission denied", True), ("exception: boom", False),
])
def test_unrecovered_tool_error_signals(result, err):
    assert unrecovered_tool_error(trace([step("t", {}, result, error=err)]), O).fired


@pytest.mark.parametrize("t", [
    trace([step("lookup_order", {"id": 1}, "Error: timeout"), step("lookup_order", {"id": 1}, {"ok": 1})]),
    trace([step("lookup_order", {}, {"data": {"error": None}, "items": [{"error": "old note"}]})]),
    trace([step("lookup_order", {}, {"error": None, "order": 1})]),
    trace([step("f", {}, "ok")], chats=[chat(90, 50, error="RateLimitError: slow down")]),
], ids=["retried-successfully", "error-inside-data", "null-error-field", "model-error-is-not-a-tool-error"])
def test_unrecovered_tool_error_clean_look_alikes(t):
    assert not unrecovered_tool_error(t, O).fired


def test_retry_of_an_error_that_also_fails_is_still_unrecovered():
    t = trace([step("f", {}, "Error: a"), step("f", {}, "Error: b")])
    assert unrecovered_tool_error(t, O).fired


# unrecovered_model_error

def test_unrecovered_model_error_fires_when_no_later_call_succeeds():
    t = trace([step("f", {}, "ok")], answer="Sorry, something went wrong.",
              chats=[chat(90, 99_000, error="APIConnectionError: Connection error.")])
    v = unrecovered_model_error(t, O)
    assert v.fired
    assert v.reason == ("model call 3 of 3 raised APIConnectionError (APIConnectionError: Connection error.) "
                        "and no later model call succeeded")


def test_model_error_retried_successfully_is_clean():
    t = trace([step("f", {}, "ok")], chats=[chat(90, 50, error="RateLimitError: slow down")])
    assert not unrecovered_model_error(t, O).fired


# context_overflow

def test_context_overflow_fires_on_length_finish_reason():
    v = context_overflow(trace([], answer="The refund policy says that", finish="length"), O)
    assert v.fired and v.reason == "model call 1 stopped on finish reason 'length': output cut at the token limit"


def test_context_overflow_fires_on_provider_context_error():
    t = trace([], chats=[chat(5, 5, error="BadRequestError: This model's maximum context length is 8192 tokens")])
    v = context_overflow(t, O)
    assert v.fired and v.reason == "model call 1 hit the context limit: 'maximum context length'"


def test_context_overflow_fires_on_termination():
    assert context_overflow(trace([], answer=None, termination="error: BadRequestError: prompt is too long"), O).fired


@pytest.mark.parametrize("answer", [
    "The context window of this model is 128k tokens.", "Your plan allows max tokens of 4096 per request."])
def test_context_overflow_clean_look_alikes(answer):
    assert not context_overflow(trace([], answer=answer), O).fired


# abstention (off by default)

def test_abstention_fires_only_when_asked():
    t = trace([step("search", {}, [])], answer="I couldn't find any order with that number.")
    assert abstention(t, O).fired
    assert "abstention" not in DEFAULT_CHECKS
    assert not abstention(trace([], answer="Found order 1042; it shipped."), O).fired


def test_resolve_rejects_unknown_names():
    assert resolve("search_loop, policy") == ("search_loop", "policy")
    with pytest.raises(ValueError, match="unknown check 'loop'"):
        resolve("loop")
