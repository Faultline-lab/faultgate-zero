# Changelog

## 0.2.0 (unreleased)

Adds features only; everything in 0.1.0 works as before.

- Check plugins: the `faultgate_zero.checks` entry point group. A plugin object has `name`, `default`
  and `run(trace, options) -> Verdict`; default plugins join `DEFAULT_CHECKS`. A name clash with a
  built-in check is a usage error at startup; a plugin that raises gives a verdict that did not fire
  (`check crashed: <ExceptionType>`), a `FaultgateZeroWarning` and a summary line.
- `faultgate_zero.formats.load(path, fmt="auto")`: OTLP through faultgate-zero's own loader, and
  OpenInference, Langfuse, LangSmith and Braintrust through faultgate's loaders, behind the new
  `faultgate-zero[formats]` extra. `formats.to_zero()` converts a faultgate trace.

## 0.1.0

First usable version.

- Release review: detect Responses API failures returned without SDK exceptions; preserve received
  text and function-call arguments when a Responses stream ends early.
- CI and tagged releases run the test suite on Python 3.11, 3.12 and 3.13 before publishing.

- pytest plugin (`pytest11` entry point): each test is one agent run, checked when it ends; `fail`,
  `warn` and `off` modes; per-test `@pytest.mark.faultgate(...)`; settings from marker, `--fgz-*`
  options, `FAULTGATE_ZERO*` environment variables and `[tool.faultgate_zero]`.
- Capture for OpenAI (chat completions and Responses API), Anthropic (messages, beta messages, stream
  helper) and LiteLLM; sync and async; streaming recorded in place; LangChain chat models and the
  OpenTelemetry OpenAI instrumentor tested alongside.
- Seven deterministic checks: `search_loop`, `repeated_call`, `unrecovered_tool_error`,
  `unrecovered_model_error`, `context_overflow`, `policy`, `abstention` (off by default).
- Policy file (`.faultline/policy.json`): allow list, denied patterns, call and argument budgets, numeric
  limits, allowed values, required arguments, identity, prerequisites, complete runs.
- `fz.run()`, `run.check()`, `run.assert_clean()`, `@fz.tool`, `fz.record_model_call()`.
- Reports: `--fgz-json`, `--fgz-markdown`, `--fgz-export-otlp` (OTLP/JSON, GenAI semantic conventions);
  merged under pytest-xdist. Built-in redaction of personal data and secrets in reasons.

## 0.0.0

Placeholder to reserve the name. Not published.
