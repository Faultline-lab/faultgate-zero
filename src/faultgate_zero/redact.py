"""Mask personal data and secrets in everything Faultgate-Zero prints or writes (OTLP export excepted)."""
from __future__ import annotations

import re
from typing import Callable

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}")),
    ("key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{10,}|\bsk-(?:proj-)?[A-Za-z0-9_-]{16,}|\bAKIA[0-9A-Z]{16}\b"
                       r"|\bgh[pousr]_[A-Za-z0-9]{20,}|\bxox[abposr]-[A-Za-z0-9-]{10,}")),
    ("token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}")),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b")),
]
_CARD = re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])")


def _luhn(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def _card(m: re.Match[str]) -> str:
    digits = re.sub(r"\D", "", m.group(0))
    return "[REDACTED:card]" if 13 <= len(digits) <= 19 and _luhn(digits) else m.group(0)


def builtin(text: str) -> str:
    for label, rx in _PATTERNS:
        text = rx.sub(f"[REDACTED:{label}]", text)
    return _CARD.sub(_card, text)


def get(spec: str) -> Callable[[str], str]:
    if spec == "builtin":
        return builtin
    if spec == "off":
        return lambda s: s
    raise ValueError(f"unknown redaction {spec!r}; use builtin or off")
