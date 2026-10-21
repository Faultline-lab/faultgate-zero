"""Settings resolution. First found wins: the test's marker, a pytest option, an environment variable,
``[tool.faultgate_zero]`` in pyproject.toml, then the default."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from . import redact
from .checks import DEFAULT_CHECKS, Options, resolve
from .policy import Policy
from .runner import Settings

MODES = ("fail", "warn", "off")
PYPROJECT_KEYS = {"mode", "checks", "policy", "redact", "json", "markdown", "export_otlp",
                  "loop_min_steps", "repeat_threshold"}
MARKER_KEYS = {"skip", "checks", "policy", "mode", "off"}
DEFAULT_POLICY = Path(".faultline") / "policy.json"


class ConfigError(ValueError):
    pass


@dataclass
class Config:
    mode: str = "fail"
    checks: tuple[str, ...] = DEFAULT_CHECKS
    policy_path: str | None = None
    policy: Policy = field(default_factory=Policy)
    redact: str = "builtin"
    json: str | None = None
    markdown: str | None = None
    export_otlp: str | None = None
    options: Options = field(default_factory=Options)
    root: Path = field(default_factory=Path.cwd)
    skip: tuple[str, ...] = ()

    @property
    def settings(self) -> Settings:
        names = tuple(n for n in self.checks if n not in self.skip)
        return Settings(checks=names, policy=self.policy, options=self.options, redact=redact.get(self.redact))


def _pyproject(root: Path) -> dict[str, Any]:
    p = root / "pyproject.toml"
    if not p.is_file():
        return {}
    try:
        data = tomllib.loads(p.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{p}: {exc}") from None
    section = data.get("tool", {}).get("faultgate_zero", {})
    unknown = set(section) - PYPROJECT_KEYS
    if unknown:
        raise ConfigError(f"{p}: [tool.faultgate_zero] {sorted(unknown)[0]}: unknown key; "
                          f"allowed keys: {', '.join(sorted(PYPROJECT_KEYS))}")
    return section


def _first(*vals: Any) -> Any:
    return next((v for v in vals if v is not None), None)


def _mode(v: Any, where: str) -> str:
    if v not in MODES:
        raise ConfigError(f"{where}: mode {v!r}: use one of {', '.join(MODES)}")
    return v


def _names(v: Any) -> tuple[str, ...]:
    return resolve(v if isinstance(v, (str, list, tuple)) else str(v))


def _load_policy(path: str | None, root: Path) -> tuple[str | None, Policy]:
    if path is None:
        found = root / DEFAULT_POLICY
        return (str(found), Policy.load(found)) if found.is_file() else (None, Policy())
    p = Path(path)
    if not p.is_absolute():
        p = root / p
    return str(p), Policy.load(p)


def resolve_session(opts: dict[str, Any], env: dict[str, str] | None = None, root: Path | None = None) -> Config:
    """``opts``: pytest option values (None when not given). ``env``: os.environ by default."""
    env = os.environ if env is None else env
    root = root or Path.cwd()
    pp = _pyproject(root)
    mode = _mode(_first(opts.get("mode"), env.get("FAULTGATE_ZERO") or None, pp.get("mode"), "fail"), "faultgate-zero")
    checks = _names(_first(opts.get("checks"), env.get("FAULTGATE_ZERO_CHECKS") or None, pp.get("checks"),
                           ",".join(DEFAULT_CHECKS)))
    red = _first(opts.get("redact"), env.get("FAULTGATE_ZERO_REDACT") or None, pp.get("redact"), "builtin")
    redact.get(red)
    policy_path, policy = _load_policy(_first(opts.get("policy"), env.get("FAULTGATE_ZERO_POLICY") or None,
                                              pp.get("policy")), root)
    o = Options()
    for k in ("loop_min_steps", "repeat_threshold"):
        if k in pp:
            if not isinstance(pp[k], int) or isinstance(pp[k], bool) or pp[k] < 1:
                raise ConfigError(f"[tool.faultgate_zero] {k}: must be a positive integer")
            o = replace(o, **{k: pp[k]})
    def out(key: str) -> str | None:  # pyproject paths are relative to the project root
        return _first(opts.get(key), str(root / pp[key]) if pp.get(key) else None)

    return Config(mode=mode, checks=checks, policy_path=policy_path, policy=policy, redact=red,
                  json=out("json"), markdown=out("markdown"), export_otlp=out("export_otlp"), options=o, root=root)


def for_marker(cfg: Config, kwargs: dict[str, Any], where: str) -> Config:
    """Apply a ``@pytest.mark.faultgate(...)`` marker's keyword arguments."""
    unknown = set(kwargs) - MARKER_KEYS
    if unknown:
        raise ConfigError(f"{where}: @pytest.mark.faultgate({sorted(unknown)[0]}=...): unknown argument; "
                          f"use {', '.join(sorted(MARKER_KEYS))}")
    out = replace(cfg)
    if kwargs.get("off"):
        out.mode = "off"
    if "mode" in kwargs:
        out.mode = _mode(kwargs["mode"], where)
    if "checks" in kwargs:
        out.checks = _names(kwargs["checks"])
    if "skip" in kwargs:
        skip = kwargs["skip"]
        out.skip = _names([skip] if isinstance(skip, str) else list(skip))
    if "policy" in kwargs:
        out.policy_path, out.policy = _load_policy(str(kwargs["policy"]), cfg.root)
    return out
