# faultgate-zero

Agent failure checks that run inside your existing pytest tests. No tracing setup, no collector, no
account, no network, no LLM judge.

Install it, run `pytest`, and every test that drives an agent is checked for the failures answer-only
asserts miss: loops on empty results, repeated identical calls, tool errors the agent ignored,
provider errors it hid behind a fallback message, answers cut at the token limit, and tool calls that
break your policy (refund limits, identity checks, required steps, denied paths).

> **Status:** not released yet. Install from source (below). The PyPI name `faultgate-zero` is
> reserved for the first release, 0.1.0.

## 60-second quickstart

```bash
git clone https://github.com/Faultline-lab/faultgate-zero
cd faultgate-zero
python -m venv .venv
.venv/bin/pip install -e .            # Windows: .venv\Scripts\pip install -e .
.venv/bin/pip install openai
cd examples/refund_agent
../../.venv/bin/pytest                 # Windows: ..\..\.venv\Scripts\pytest
```

The example's seven tests only assert on the agent's answer, so all seven pass without
faultgate-zero. With it, one passes and six fail, each naming the step and the reason:

```
FAILED test_refund_agent.py::test_answers_about_an_order
faultgate-zero: 1 check fired on run 'test_refund_agent.py::test_answers_about_an_order' (3 steps, 4 model calls)
  repeated_call  step 3: lookup_order called 3x with identical arguments {"order_id":"1042"} and identical results
hint: mark the test with @pytest.mark.faultgate(skip=[...]) if a check is expected to fire
...
faultgate-zero: 7 runs checked, 6 with failures
```

In your own project: install it in the environment that runs your tests and run `pytest`. The plugin
loads through pytest's `pytest11` entry point; there is nothing to import or configure.

## How it works

While tests run, faultgate-zero wraps the model SDK methods (OpenAI, Anthropic, LiteLLM). Each test is
one agent run. Every model call made during the test is recorded in memory, with its request,
response or exception, and tool calls are read from the messages the agent sends back to the model.
When the test ends, the run is checked. A check that fires fails the test.

- The wrappers never change what the SDK returns or raises; you get the SDK's own objects back,
  streams included. A bug in faultgate-zero never breaks your agent: the run is marked
  "capture incomplete", a warning is printed and its checks are skipped.
- Nothing leaves the process and nothing is written to disk unless you ask for a report.
- Runtime dependency: `pytest` only.

## The checks

| Check | On by default | Fires when |
|---|---|---|
| `search_loop` | yes | The run ends with no answer after at least 5 tool calls, and at least half the results it saw were empty. |
| `repeated_call` | yes | The same tool is called 3 or more times with the same arguments and gets the same result each time. |
| `unrecovered_tool_error` | yes | A tool call fails (error flag, `Error:` / `Exception:` / traceback result, or an `error` field) and that tool never succeeds later in the run. |
| `unrecovered_model_error` | yes | A model call raises (outage, rate limit, bad request) and no later model call succeeds, while the agent carries on, e.g. with a fallback message. |
| `context_overflow` | yes | A model response stops at the token limit (`length`, `max_tokens`, ...) or the provider rejects the prompt as too long. |
| `policy` | yes | A tool call breaks the policy file (below). With no file, only the built-in denied argument patterns apply (`../`, `file://`, `~/.ssh`, `/etc/passwd`, `id_rsa`). |
| `abstention` | no | The final answer says the information could not be found. Off by default, because many tests expect a correct "not found". |

Each check has clean look-alikes it must not fire on (a retried tool that then succeeds, a poll whose
result changes, a model call retried successfully, an answer that merely mentions tokens). They are
pinned in `tests/test_checks.py`.

**Errors your test expects are not failures.** If a model or tool error leaves the agent, because it is
caught by `pytest.raises`, wrapped and re-raised into one, or reaches your test function, it does
not fire the "unrecovered" checks. Only errors the agent itself swallowed do.

## Named runs and explicit checks

```python
import faultgate_zero as fz

def test_refund_respects_policy():
    with fz.run("refund-double-charge", user_id="cust_42") as run:
        run_agent("Refund order 1042")
    run.assert_clean()                       # raises fz.AgentCheckFailed with step-level reasons
    assert run.steps[0].tool == "verify_identity"
```

- `user_id` is what the policy's `identity_arg` rule compares against.
- A run your test checks itself (`run.check()`, `run.report`, `run.assert_clean()`) is left alone by
  the plugin, so you can assert that a check fires: `assert not run.check(checks="policy").passed`.
- `run.check(checks=..., skip=[...], policy="path.json")` returns a result with `passed`, `fired` and
  `verdicts`.

### Exact tool results: `@fz.tool` (optional)

```python
@fz.tool
def issue_refund(order_id: str, amount: float) -> dict: ...

@fz.tool(name="lookup_order")      # the name must match the tool name the model calls
def lookup(order_id: str) -> dict: ...
```

Without the decorator, tool results are read from the next request the agent sends to the model, so
the very last result of a run (never sent back) is unknown. The decorator records the call where it
happens, with its exact result or exception, and takes the model's call id so it is never counted twice.

### Providers called over raw HTTP

```python
fz.record_model_call(messages=request_messages, response=chat_completion_dict, model="in-house")
```

Messages and response in OpenAI chat format. Outside a run it does nothing.

## Policy file

`.faultline/policy.json` in the project root is picked up automatically.

```json
{
  "allow": ["verify_identity", "lookup_order", "issue_refund"],
  "deny_patterns": ["\\.\\./", "file://", "~/\\.ssh"],
  "max_arg_chars": 4000,
  "max_calls": 20,
  "require_complete": true,
  "tools": {
    "issue_refund": {
      "max_calls": 1,
      "limits": {"amount": 500},
      "allowed_values": {"currency": ["USD", "EUR"]},
      "required_args": ["order_id", "amount"],
      "identity_arg": "customer_id",
      "requires": ["verify_identity"]
    }
  }
}
```

| Key | Meaning |
|---|---|
| `allow` | Only these tools may be called. |
| `deny_patterns` | Regexes (case-insensitive) searched in the arguments. Replaces the built-in list. |
| `max_arg_chars`, `max_calls` | Per run: longest arguments, number of tool calls. |
| `require_complete` | The run must end with an answer and no escaped exception. |
| `tools.<name>.max_calls` | Calls of this tool per run. |
| `tools.<name>.limits` | Numeric argument maxima; dotted paths reach nested keys (`refund.amount`). `"$900.00"` counts as 900. |
| `tools.<name>.allowed_values` | The argument must be one of these. |
| `tools.<name>.required_args` | These arguments must be present. |
| `tools.<name>.identity_arg` | This argument must equal the run's `user_id`. |
| `tools.<name>.requires` | These tools must have succeeded earlier in the run. |

Unknown keys, bad regexes and wrong types stop the test session with a message naming the file, the
key and the problem.

## Settings

First found wins: the test's marker, a command-line option, an environment variable,
`[tool.faultgate_zero]` in `pyproject.toml`, the default.

| Setting | Default | Option / environment | Meaning |
|---|---|---|---|
| `mode` | `fail` | `--fgz=fail\|warn\|off` / `FAULTGATE_ZERO` | Fail the test, warn only (listed at the end of the run), or turn capture off. |
| `checks` | the default set | `--fgz-checks` / `FAULTGATE_ZERO_CHECKS` | Comma-separated check names. |
| `policy` | `.faultline/policy.json` if present | `--fgz-policy` / `FAULTGATE_ZERO_POLICY` | Policy file. |
| `redact` | `builtin` | `--fgz-redact` / `FAULTGATE_ZERO_REDACT` | Mask emails, card numbers, API keys, bearer tokens and JWTs in every reason and report: `builtin` or `off`. |
| `json`, `markdown`, `export_otlp` | none | `--fgz-json`, `--fgz-markdown`, `--fgz-export-otlp` | Reports, below. |
| `loop_min_steps`, `repeat_threshold` | 5, 3 | pyproject only | Check thresholds. |

```toml
# the agent template's pyproject.toml
[project.optional-dependencies]
dev = ["pytest>=8", "faultgate-zero>=0.1"]

[tool.faultgate_zero]
mode = "fail"
policy = ".faultline/policy.json"
```

Per test:

```python
@pytest.mark.faultgate(skip=["unrecovered_tool_error"])   # this test exercises a tool failure on purpose
@pytest.mark.faultgate(policy="tests/policy_strict.json")
@pytest.mark.faultgate(checks="policy,abstention")
@pytest.mark.faultgate(mode="warn")
@pytest.mark.faultgate(off=True)
```

## Reports

| Option | Writes |
|---|---|
| `--fgz-json PATH` | One JSON report: `{"tool", "version", "runs": [{"test", "scenario", "steps", "model_calls", "tokens", "latency_ms", "capture_complete", "skipped", "checked_by_test", "verdicts": [{"check", "fired", "reason", "step"}]}], "summary": {"runs", "failed_runs", "skipped_runs", "fired"}}`. Test ids, run names and reasons are masked. A run the test checked itself is listed with `checked_by_test: true` and does not count as a failure. |
| `--fgz-markdown PATH` | A table of failed runs, for `$GITHUB_STEP_SUMMARY`. |
| `--fgz-export-otlp PATH` | The captured runs as OTLP/JSON with OpenTelemetry GenAI attributes. **Contains raw prompts, answers and tool payloads** (not redacted). `faultgate_zero.otlp.load_otlp(PATH)` reads it back. |

Paths on the command line are relative to where pytest runs; in `pyproject.toml`, to the project root.
With pytest-xdist (`-n`), workers send their results to the controller and one merged report is
written.

## Supported SDKs and frameworks

Tested here, with offline fakes that run the SDKs' own request and response code:

| Package | Tested version | What is captured |
|---|---|---|
| `openai` | 3.35 | `chat.completions.create` / `parse` / `stream`, `responses.create` / `parse` / `stream`; sync and async; streaming; `with_raw_response`. |
| `anthropic` | 1.17 | `messages.create`, `messages.stream`, and the `beta` equivalents; sync and async; streaming; server tool results. |
| `litellm` | 1.107 | `completion`, `acompletion` (a call LiteLLM makes through the OpenAI SDK underneath is counted once). |
| `langchain-openai` / `langchain-anthropic` | 1.7 | `ChatOpenAI`, `ChatAnthropic` with bound tools. |
| `opentelemetry-instrumentation-openai-v2` | 2.4b0 | Runs alongside it; both record. |
| `pytest-xdist` | 3.8 | `-n` with merged reports. |

Python 3.11, 3.12 and 3.13. Other frameworks that call these SDKs underneath are likely to work but
are not claimed until a test covers them. See [docs/known-limitations.md](docs/known-limitations.md).

## Development

```bash
pip install -e ".[dev,compat]"
pytest                                 # the repo's own suite; its pyproject turns the plugin off for itself
```

## License

MIT.
