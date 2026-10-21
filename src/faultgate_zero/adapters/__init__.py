"""Provider adapters. Each module has ``start(args, kwargs) -> ModelCall``, ``finish(call, result, run)``
(returns the result the caller gets, unchanged) and ``normalise(call) -> dict`` (semconv span attributes).
Adding a provider means adding one module here plus its fixtures."""
from __future__ import annotations

import importlib
from types import ModuleType


def get(name: str) -> ModuleType:
    return importlib.import_module(f"{__name__}.{name}")
