"""Endpoint probes behind `nell provider check`.

Four live checks against the configured endpoint. The tool-call probe matters
most: many local models can chat but can't tool-call, and every memory /
emotion / soul tool the companion uses depends on it.
"""

from __future__ import annotations

from dataclasses import dataclass

from brain.bridge.chat import ChatMessage, StreamError, TextDelta
from brain.bridge.openai_compat import OpenAICompatProvider
from brain.bridge.provider import ProviderError

_PROBE_TOOL = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Return the current time. Always call this when asked what time it is.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def run_endpoint_check(provider: OpenAICompatProvider) -> list[CheckResult]:
    """Run the four probes: reachable, chat, tool call, streaming."""
    results = [
        CheckResult(
            "endpoint reachable (GET /models)",
            provider.healthy(),
            provider.base_url,
        )
    ]

    try:
        reply = provider.chat([ChatMessage(role="user", content="Reply with the single word: ready")])
        results.append(CheckResult("chat returns text", bool(reply.content.strip()), reply.content.strip()[:80]))
    except ProviderError as exc:
        results.append(CheckResult("chat returns text", False, str(exc)))

    try:
        response = provider.chat(
            [ChatMessage(role="user", content="What time is it? Use the get_time tool.")],
            tools=[_PROBE_TOOL],
        )
        called = any(call.name == "get_time" for call in response.tool_calls)
        detail = "" if called else (
            "no tool call — this model may not support tool calling, "
            "so her memory and emotion tools won't work"
        )
        results.append(CheckResult("model calls a tool", called, detail))
    except ProviderError as exc:
        results.append(CheckResult("model calls a tool", False, str(exc)))

    events = list(provider.chat_stream([ChatMessage(role="user", content="Count from 1 to 5.")]))
    errors = [event for event in events if isinstance(event, StreamError)]
    deltas = [event for event in events if isinstance(event, TextDelta)]
    if errors:
        detail = f"[{errors[0].stage}] {errors[0].detail}"
    else:
        detail = f"{len(deltas)} chunks"
    results.append(CheckResult("streaming yields text", not errors and bool(deltas), detail))
    return results
