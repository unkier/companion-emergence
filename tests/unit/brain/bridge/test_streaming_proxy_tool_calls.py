"""_StreamingProxy turns streamed ToolCallEvents into ChatResponse.tool_calls.

Providers that hand tools back to run_tool_loop (openai-compat) surface them as
ToolCallEvents; without collection the live WS path would drop every tool call.
"""
from __future__ import annotations

import asyncio
from collections.abc import Iterable

from brain.bridge.chat import (
    ChatMessage,
    ChatResponse,
    StreamDone,
    TextDelta,
    ToolCall,
    ToolCallEvent,
)
from brain.bridge.server import _StreamingProxy


class _StreamProvider:
    def __init__(self, events: list) -> None:
        self._events = events

    def name(self) -> str:
        return "fake"

    def healthy(self) -> bool:
        return True

    def generate(self, prompt, *, system=None) -> str:
        return ""

    def complete(self, prompt) -> str:
        return ""

    def chat(self, messages, *, tools=None, options=None) -> ChatResponse:
        raise AssertionError("the proxy must use chat_stream")

    def chat_stream(self, messages, *, tools=None, options=None) -> Iterable:
        yield from self._events


def _run(events: list) -> tuple[ChatResponse, list]:
    loop = asyncio.new_event_loop()

    async def _go():
        q: asyncio.Queue = asyncio.Queue()
        proxy = _StreamingProxy(_StreamProvider(events), q, asyncio.get_event_loop())
        resp = await asyncio.to_thread(
            proxy.chat, [ChatMessage(role="user", content="hi")], tools=None, options=None
        )
        pushed = []
        while not q.empty():
            pushed.append(q.get_nowait())
        return resp, pushed

    try:
        return loop.run_until_complete(_go())
    finally:
        loop.close()


def test_proxy_collects_tool_call_events_into_the_response():
    resp, pushed = _run(
        [
            TextDelta(text="Let me look. "),
            ToolCallEvent(id="c1", name="search_memories", arguments={"query": "tea"}),
            ToolCallEvent(name="get_soul", arguments={}),
            StreamDone(content="Let me look."),
        ]
    )
    assert resp.tool_calls == (
        ToolCall(id="c1", name="search_memories", arguments={"query": "tea"}),
        ToolCall(id="call_1", name="get_soul", arguments={}),
    )
    assert resp.content == "Let me look. "
    assert pushed == ["Let me look. "]


def test_proxy_text_only_stream_still_returns_no_tool_calls():
    resp, _ = _run([TextDelta(text="hi"), StreamDone(content="hi")])
    assert resp.tool_calls == ()
