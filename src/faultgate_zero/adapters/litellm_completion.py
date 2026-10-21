"""LiteLLM (``litellm.completion`` / ``litellm.acompletion``). Requests and responses use the OpenAI chat
format, so the OpenAI chat normaliser does the work. Calls LiteLLM makes through a patched SDK
underneath are not recorded twice: the outermost recorded call wins."""
from __future__ import annotations

from typing import Any

from ..recorder import ModelCall, Run
from . import openai_chat


def start(args: tuple, kwargs: dict[str, Any]) -> ModelCall:
    model = kwargs.get("model", args[0] if args else "")
    msgs = kwargs.get("messages", args[1] if len(args) > 1 else None)
    provider = str(model).split("/", 1)[0] if "/" in str(model) else "litellm"
    return ModelCall(adapter="litellm_completion", provider=provider, api="chat", model=str(model or ""),
                     request={"messages": list(msgs) if isinstance(msgs, (list, tuple)) else []},
                     streamed=bool(kwargs.get("stream")))


def finish(call: ModelCall, result: Any, run: Run) -> Any:
    return openai_chat.finish(call, result, run)


tool_calls = openai_chat.tool_calls
normalise = openai_chat.normalise
