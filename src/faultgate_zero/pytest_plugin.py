"""pytest plugin, loaded through the ``pytest11`` entry point."""
from __future__ import annotations

import pytest

from . import patch, recorder

_RUNS = pytest.StashKey[list]()
_LISTENER = pytest.StashKey[object]()


def pytest_configure(config: pytest.Config) -> None:
    patch.install()
    runs: list = []
    config.stash[_RUNS] = runs

    def collect(run: recorder.Run) -> None:
        if run.model_calls:
            runs.append(run)

    config.stash[_LISTENER] = collect
    recorder.add_listener(collect)


def pytest_unconfigure(config: pytest.Config) -> None:
    fn = config.stash.get(_LISTENER, None)
    if fn is not None:
        recorder.remove_listener(fn)


def pytest_terminal_summary(terminalreporter, exitstatus: int, config: pytest.Config) -> None:
    runs = config.stash.get(_RUNS, [])
    terminalreporter.write_line(f"faultgate-zero: {len(runs)} runs checked")
