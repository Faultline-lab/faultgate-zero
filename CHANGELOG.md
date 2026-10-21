# Changelog

## 0.1.0 (unreleased)

First usable version.

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
