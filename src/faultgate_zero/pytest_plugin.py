"""pytest plugin, loaded through the ``pytest11`` entry point: each test is one agent run, checked when it ends."""
from __future__ import annotations

import inspect
from collections import Counter
from dataclasses import dataclass, field

import pytest

from . import patch, recorder, runner
from .config import Config, ConfigError, for_marker, resolve_session
from .policy import PolicyError

_CFG = pytest.StashKey[Config]()
_STATE = pytest.StashKey["_State"]()
_ITEM_RUNS = pytest.StashKey[list]()
_ITEM_CFG = pytest.StashKey[Config]()

HINT = "hint: mark the test with @pytest.mark.faultgate(skip=[...]) if a check is expected to fire"


@dataclass
class Checked:
    nodeid: str
    run: recorder.Run
    result: runner.RunResult


@dataclass
class _State:
    collecting: list | None = None
    checked: list[Checked] = field(default_factory=list)
    warned: list[tuple[str, str]] = field(default_factory=list)

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
    opts = {"mode": config.getoption("fgz_mode"), "checks": config.getoption("fgz_checks"),
            "policy": config.getoption("fgz_policy"), "redact": config.getoption("fgz_redact"),
            "json": config.getoption("fgz_json"), "markdown": config.getoption("fgz_markdown"),
            "export_otlp": config.getoption("fgz_otlp")}
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
    fired = []
    settings = icfg.settings
    for r in runs:
        if r.checked or not r.model_calls:
            continue
        result = runner.check(r.trace, settings)
        state.checked.append(Checked(item.nodeid, r, result))
        if result.fired:
            fired.append(result)
    if not fired:
        return report
    text = "\n\n".join(res.summary() for res in fired) + "\n" + HINT
    if icfg.mode == "fail" and report.passed and not hasattr(report, "wasxfail"):
        report.outcome = "failed"
        report.longrepr = text
    else:
        report.sections.append(("faultgate-zero", text))
        if report.passed:
            state.warned.append((item.nodeid, text))
    return report


def pytest_terminal_summary(terminalreporter, exitstatus: int, config: pytest.Config) -> None:
    state = config.stash.get(_STATE, None)
    cfg = config.stash.get(_CFG, None)
    if state is None or cfg is None:
        return
    tr = terminalreporter
    if cfg.mode == "off":
        tr.write_line("faultgate-zero: off")
        return
    if state.warned:
        tr.write_sep("-", "faultgate-zero warnings")
        for nodeid, text in state.warned:
            tr.write_line(nodeid)
            for line in text.splitlines():
                tr.write_line(f"  {line}")
    failed = [c for c in state.checked if c.result.fired]
    counts = Counter(v.check for c in failed for v in c.result.fired)
    skipped = sum(1 for c in state.checked if c.result.skipped)
    line = f"faultgate-zero: {len(state.checked)} runs checked, {len(failed)} with failures"
    if skipped:
        line += f", {skipped} skipped (capture incomplete)"
    tr.write_line(line)
    for name, n in sorted(counts.items()):
        tr.write_line(f"  {name}: {n}")
