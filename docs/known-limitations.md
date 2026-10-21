# Known limitations

What faultgate-zero does not do, or does imperfectly, as of 0.1.0 (unreleased).

## Capture

- **Raw HTTP calls are not captured.** Only the SDK methods in the supported table are patched. A
  framework or client that calls a provider's HTTP API directly is invisible; use
  `fz.record_model_call()` for those calls.
- **Plain threads do not inherit the run.** Calls made in a `threading.Thread` started without a copied
  context (`contextvars.copy_context().run`) are not recorded, and neither is work sent to
  `loop.run_in_executor` or a bare `ThreadPoolExecutor`. `asyncio` tasks and `asyncio.to_thread` inherit it.
- **Only the first choice is read** when a chat completion asks for `n > 1` choices.
- **Imported-early functions.** `from litellm import completion` executed before the plugin loads keeps
  the unpatched function. Under pytest the plugin loads before test modules import, so this only
  affects code imported in `conftest.py` at plugin-load time or outside pytest.
- **Tool results without `@fz.tool`** are read from the next model request, so the last result of a
  run is unknown (`result_seen = False`). Checks never treat an unknown result as empty or as an error.
- **The decorator's name must match** the tool name the model calls; otherwise the decorated call
  cannot take the model's call id and counts as a separate step.

## Checks

- **Model-call errors that reach your test count as expected.** If the agent lets a provider error
  propagate and your test catches it, by `pytest.raises` or by a `try` in the test function,
  `unrecovered_model_error` does not fire. If a helper function in your test module (not the test
  function itself) catches it without `pytest.raises`, it does fire; mark that test with
  `@pytest.mark.faultgate(skip=["unrecovered_model_error"])`.
- **A tool error recovered with a different tool** (search A failed, search B worked) still fires
  `unrecovered_tool_error`, because the failing tool never succeeded. Skip the check on that test if
  the fallback is the intended design.
- **A later successful call to the same tool counts as recovery, whatever its arguments.** If
  `lookup_order("9999")` fails and `lookup_order("1042")` then succeeds, the first error is treated as
  recovered. This is what lets a retry with corrected arguments pass; the price is that a failure for
  one input can be masked by a success for another.
- **One run per test.** Concurrent sub-tasks (`asyncio.gather`) inside one test share the test's run, so
  one task's success can mask another's tool error. Give each task its own `fz.run(...)` to check them
  separately.
- **Polling that repeats identical results and then succeeds does not fire `repeated_call`**, because
  not all the calls returned the same result. A poll that never changes state does.
- **A missing `user_id` fails the identity rule on purpose.** A policy with `identity_arg` cannot be
  satisfied without knowing who the user is; pass `fz.run(user_id=...)`.
- **Long arguments are scanned in windows.** Arguments over 1,024 characters are searched in
  overlapping 1,024-character windows (256 characters of overlap), so the time a deny pattern takes
  grows linearly with the argument, and a denied path is found wherever it sits. A single match longer
  than 256 characters that straddles two windows can be missed; keep deny patterns short.
- **Deny patterns are vetted conservatively.** A pattern that repeats a group which itself repeats or
  branches is refused, including some safe ones such as `(?:ab?)+`. Split it into simpler patterns.
- **Plain-text errors are not recognised** unless they start with `Error:`, `Exception:` or a
  traceback, or the tool result sets an error flag or an `error` field.
- **The answer is the model's text.** An agent that finishes by calling a tool (a `submit_answer` or
  `final_output` tool) has no text answer, so the policy's `require_complete` reports "no final
  answer" for it. Leave `require_complete` off for such agents.
- **No golden answers and no LLM judge.** Answer correctness is left to your own asserts.

## Reports

- `--fgz-export-otlp` writes raw prompts, answers and tool payloads. It is opt-in and its help text says
  so; do not upload it where those payloads must not go.
- The JSON report's reasons are redacted with the built-in patterns (emails, Luhn-valid card numbers,
  common API key formats, bearer tokens, JWTs). Names, addresses and other personal data are not.

## Tested versions

Only the versions in the README's table are tested. SDK releases that move or rename the patched
methods produce a warning at start-up ("not found in this version; its calls are not captured").
