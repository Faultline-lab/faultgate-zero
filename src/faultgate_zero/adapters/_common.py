from __future__ import annotations

from typing import Any


def as_dict(obj: Any) -> dict[str, Any]:
    """A plain dict view of an SDK object (pydantic model or dict); {} for anything else."""
    if isinstance(obj, dict):
        return obj
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            d = dump()
        except Exception:
            return {}
        return d if isinstance(d, dict) else {}
    return {}


def text_part(text: Any) -> dict[str, Any]:
    return {"type": "text", "content": str(text)}


def drop_none(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}
