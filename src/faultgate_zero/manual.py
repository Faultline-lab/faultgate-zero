"""Manual capture for model calls Faultgate-Zero cannot patch (a provider called over raw HTTP)."""
from __future__ import annotations

from typing import Any

from .recorder import ModelCall, Run, current_run


def record_model_call(*, messages: list[Any], response: Any = None, error: BaseException | None = None,
                      model: str = "", provider: str = "custom", run: Run | None = None) -> None:
    """Record one model call in the current run, in OpenAI chat format.

    ``messages`` is the request's message list; ``response`` a chat completion (dict or object) with
    ``choices[0].message`` (content, tool_calls) and ``finish_reason``; ``error`` the exception if the call
    failed. Outside a run this does nothing.
    """
    r = run or current_run()
    if r is None:
        return
    call = ModelCall(adapter="openai_chat", provider=provider, api="chat", model=model,
                     request={"messages": list(messages)}, response=response, error=error)
    call.start_ns = r.now()
    call.end_ns = r.now()
    r.model_calls.append(call)
