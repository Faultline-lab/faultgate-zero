"""pytest plugin, loaded through the ``pytest11`` entry point: each test is one agent run, checked when it ends."""
from __future__ import annotations

import inspect
import json
import os
from dataclasses import dataclass, field

import pytest

from . import otlp, patch, recorder, runner
from . import report as rpt
from .config import Config, ConfigError, for_marker, resolve_session
from .policy import PolicyError

_CFG = pytest.StashKey[Config]()
_STATE = pytest.StashKey["_State"]()
_ITEM_RUNS = pytest.StashKey[list]()
_ITEM_CFG = pytest.StashKey[Config]()

HINT = "hint: mark the test with @pytest.mark.faultgate(skip=[...]) if a check is expected to fire"


@dataclass
class _State:
    collecting: list | None = None
    entries: list[dict] = field(default_factory=list)  # one per checked run, from this process or xdist workers
    spans: list[dict] = field(default_factory=list)    # OTLP spans, only with --fgz-export-otlp

    def on_close(self, run: recorder.Run) -> None:
        if self.collecting is not None:
            self.collecting.append(run)


def pytest_addoption(parser: pytest.Parser) -> None:
    g = parser.getgroup("faultgate-zero", "agent failure checks")
    g.addoption("--fgz", dest="fgz_mode", choices=("fail", "warn", "off"), default=None,
                help="fail the test, warn only, or turn capture off (default: fail)")
    g.addoption("--fgz-checks", dest="fgz_checks", default=None, help="comma-separated check names")
    g.addoption("--fgz-policy", dest="fgz_policy", default=None,
                help="policy JSON file (default: .faultline/policy.json if present)")
    g.addoption("--fgz-redact", dest="fgz_redact", choices=("builtin", "off"), default=None,
                help="mask personal data and secrets in reasons (default: builtin)")
    g.addoption("--fgz-json", dest="fgz_json", default=None, help="write a JSON report of all checked runs")
    g.addoption("--fgz-markdown", dest="fgz_markdown", default=None, help="write a Markdown summary (for CI job summaries)")
    g.addoption("--fgz-export-otlp", dest="fgz_otlp", default=None,
                help="write captured runs as OTLP/JSON. Contains raw prompts, answers and tool payloads.")


def _installed_raises_hook() -> None:
    """Note every exception ``pytest.raises`` catches, so an error the test expected is not reported as
    one the agent swallowed. ``ExceptionInfo.fill_unfilled`` is where both raises forms store it."""
    from _pytest._code.code import ExceptionInfo

    original = ExceptionInfo.__dict__.get("fill_unfilled")
    if original is None or getattr(original, "_fgz", False):
        return

    def fill_unfilled(self, exc_info):  # type: ignore[no-untyped-def]
        try:
            recorder.note_expected(exc_info[1])
        except Exception:
            pass
        return original(self, exc_info)

    fill_unfilled._fgz = True  # type: ignore[attr-defined]
    ExceptionInfo.fill_unfilled = fill_unfilled  # type: ignore[method-assign]


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "faultgate(skip=[...], checks=..., policy=..., mode=..., off=False): "
                                       "Faultgate-Zero settings for this test")
    here = config.invocation_params.dir  # command-line paths are relative to where pytest was run
    opts = {"mode": config.getoption("fgz_mode"), "checks": config.getoption("fgz_checks"),
            "policy": _abs(config.getoption("fgz_policy"), here), "redact": config.getoption("fgz_redact"),
            "json": _abs(config.getoption("fgz_json"), here), "markdown": _abs(config.getoption("fgz_markdown"), here),
            "export_otlp": _abs(config.getoption("fgz_otlp"), here)}
    try:
        cfg = resolve_session(opts, root=config.rootpath)
    except (ConfigError, PolicyError, ValueError) as exc:
        raise pytest.UsageError(f"faultgate-zero: {exc}") from None
    config.stash[_CFG] = cfg
    state = _State()
    config.stash[_STATE] = state
    if cfg.mode == "off":
        return
    patch.install()
    _installed_raises_hook()
    runner.set_default_settings(cfg.settings)
    recorder.add_listener(state.on_close)


def _abs(path: str | None, base) -> str | None:
    return None if path is None else os.path.join(str(base), path)


def pytest_unconfigure(config: pytest.Config) -> None:
    state = config.stash.get(_STATE, None)
    if state is not None:
        recorder.remove_listener(state.on_close)
    runner.set_default_settings(runner.Settings())


def _item_config(item: pytest.Item) -> Config:
    cfg = item.config.stash[_CFG]
    for mark in reversed(list(item.iter_markers("faultgate"))):  # closest marker last, so it wins
        try:
            cfg = for_marker(cfg, dict(mark.kwargs), item.nodeid)
        except (ConfigError, PolicyError, ValueError) as exc:
            raise pytest.UsageError(f"faultgate-zero: {exc}") from None
    return cfg


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item: pytest.Item):
    state = item.config.stash.get(_STATE, None)
    cfg = item.config.stash.get(_CFG, None)
    if state is None or cfg is None or cfg.mode == "off":
        return (yield)
    icfg = _item_config(item)
    item.stash[_ITEM_CFG] = icfg
    if icfg.mode == "off":
        return (yield)
    runs: list = []
    item.stash[_ITEM_RUNS] = runs
    run = recorder.Run(test=item.nodeid, agent=getattr(getattr(item, "module", None), "__name__", None),
                       implicit=True)
    fn = getattr(item, "obj", None)
    if fn is not None:
        run.owner_codes.add(getattr(inspect.unwrap(fn), "__code__", None))
    state.collecting = runs
    run.open()
    exc: BaseException | None = None
    try:
        return (yield)
    except BaseException as e:
        exc = e
        raise
    finally:
        run.close(exc)
        state.collecting = None


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    report = yield
    if call.when != "call":
        return report
    runs = item.stash.get(_ITEM_RUNS, None)
    icfg = item.stash.get(_ITEM_CFG, None)
    state = item.config.stash.get(_STATE, None)
    if not runs or icfg is None or state is None:
        return report
    settings = icfg.settings
    export = item.config.stash[_CFG].export_otlp is not None
    checked: list[runner.RunResult] = []
    for r in runs:
        if not r.model_calls:
            continue
        if export:
            state.spans += otlp.spans_of(r.trace)
        if not r.checked:
            checked.append(runner.check(r.trace, settings))
    fired = [res for res in checked if res.fired]
    if fired:
        text = "\n\n".join(res.summary() for res in fired) + "\n" + HINT
        if icfg.mode == "fail" and report.passed and not hasattr(report, "wasxfail"):
            report.outcome = "failed"
            report.longrepr = text
        else:
            report.sections.append(("faultgate-zero", text))
    state.entries += [rpt.entry(item.nodeid, res, report.passed) for res in checked]
    return report


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    config = session.config
    state, cfg = config.stash.get(_STATE, None), config.stash.get(_CFG, None)
    if state is None or cfg is None or cfg.mode == "off":
        return
    if hasattr(config, "workeroutput"):  # an xdist worker: hand everything to the controller
        config.workeroutput["faultgate_zero"] = json.dumps({"entries": state.entries, "spans": state.spans})
        return
    if cfg.json:
        rpt.write_json(state.entries, cfg.json)
    if cfg.markdown:
        rpt.write_markdown(state.entries, cfg.markdown)
    if cfg.export_otlp:
        otlp.write_spans(state.spans, cfg.export_otlp)


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node, error) -> None:
    """xdist controller: collect a finished worker's checked runs."""
    state = node.config.stash.get(_STATE, None)
    raw = getattr(node, "workeroutput", {}).get("faultgate_zero")
    if state is None or not raw:
        return
    data = json.loads(raw)
    state.entries += data.get("entries", [])
    state.spans += data.get("spans", [])


def pytest_terminal_summary(terminalreporter, exitstatus: int, config: pytest.Config) -> None:
    state = config.stash.get(_STATE, None)
    cfg = config.stash.get(_CFG, None)
    if state is None or cfg is None:
        return
    tr = terminalreporter
    if cfg.mode == "off":
        tr.write_line("faultgate-zero: off")
        return
    warned = [e for e in state.entries if e["_fired"] and e["_test_passed"]]
    if warned:
        tr.write_sep("-", "faultgate-zero warnings")
        for e in warned:
            tr.write_line(e["test"])
            for line in e["_summary"].splitlines():
                tr.write_line(f"  {line}")
    s = rpt.summary(state.entries)
    line = f"faultgate-zero: {s['runs']} runs checked, {s['failed_runs']} with failures"
    if s["skipped_runs"]:
        line += f", {s['skipped_runs']} skipped (capture incomplete)"
    tr.write_line(line)
    for name, n in s["fired"].items():
        tr.write_line(f"  {name}: {n}")
    for label, path in (("JSON report", cfg.json), ("Markdown summary", cfg.markdown), ("OTLP export", cfg.export_otlp)):
        if path:
            tr.write_line(f"faultgate-zero: {label} written to {path}")
