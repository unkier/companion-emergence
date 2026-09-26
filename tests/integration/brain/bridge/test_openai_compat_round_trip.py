"""End to end: the model calls a brain tool, the brain runs it, the model answers.

Exercises the real run_tool_loop + dispatch against a fake OpenAI-compatible
server — once on the plain chat() path (background / non-WS) and once through
_StreamingProxy (the live NellFace WS path).
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from brain.bridge.chat import ChatMessage
from brain.bridge.openai_compat import OpenAICompatProvider, OpenAICompatSettings
from brain.bridge.server import _StreamingProxy
from brain.chat.tool_loop import build_tools_list, run_tool_loop
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore

_SETTINGS = OpenAICompatSettings.from_config({"base_url": "http://fake.local/v1", "model": "m"})
_TOOL_CALL = {"id": "c1", "type": "function", "function": {"name": "get_emotional_state", "arguments": ""}}


@pytest.fixture
def persona_dir(tmp_path: Path) -> Path:
    d = tmp_path / "personas" / "nell"
    d.mkdir(parents=True)
    return d


def _messages() -> list[ChatMessage]:
    return [ChatMessage(role="system", content="You are nell."), ChatMessage(role="user", content="How are you?")]


def _run_loop(provider, persona_dir: Path):
    store = MemoryStore(db_path=":memory:")
    hebbian = HebbianMatrix(db_path=":memory:")
    try:
        return run_tool_loop(
            _messages(),
            provider=provider,
            tools=build_tools_list(),
            store=store,
            hebbian=hebbian,
            persona_dir=persona_dir,
        )
    finally:
        store.close()
        hebbian.close()


def _assert_tool_result_was_sent_back(second_request: dict) -> None:
    messages = second_request["messages"]
    assert messages[-2]["role"] == "assistant"
    assert messages[-2]["tool_calls"][0]["id"] == "c1"
    assert messages[-1]["role"] == "tool"
    assert messages[-1]["tool_call_id"] == "c1"
    json.loads(messages[-1]["content"])  # the dispatched result, as JSON


def test_round_trip_on_the_chat_path(persona_dir: Path) -> None:
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            message = {"role": "assistant", "content": None, "tool_calls": [_TOOL_CALL]}
        else:
            message = {"role": "assistant", "content": "I feel steady."}
        return httpx.Response(200, json={"choices": [{"message": message}]})

    provider = OpenAICompatProvider(_SETTINGS, "m", None, transport=httpx.MockTransport(handler))
    response, invocations = _run_loop(provider, persona_dir)

    assert response.content == "I feel steady."
    assert [inv["name"] for inv in invocations] == ["get_emotional_state"]
    assert len(requests) == 2
    _assert_tool_result_was_sent_back(requests[1])


def test_round_trip_through_the_streaming_proxy(persona_dir: Path) -> None:
    requests: list[dict] = []

    def sse(*chunks: dict) -> bytes:
        return ("".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n").encode()

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            fragment = {"index": 0, **_TOOL_CALL}
            body = sse({"choices": [{"delta": {"tool_calls": [fragment]}}]})
        else:
            body = sse({"choices": [{"delta": {"content": "I feel "}}]}, {"choices": [{"delta": {"content": "steady."}}]})
        return httpx.Response(200, content=body)

    provider = OpenAICompatProvider(_SETTINGS, "m", None, transport=httpx.MockTransport(handler))

    async def _go():
        q: asyncio.Queue = asyncio.Queue()
        proxy = _StreamingProxy(provider, q, asyncio.get_running_loop())
        result = await asyncio.to_thread(_run_loop, proxy, persona_dir)
        await asyncio.sleep(0)
        pushed = []
        while not q.empty():
            pushed.append(q.get_nowait())
        return result, pushed

    loop = asyncio.new_event_loop()
    try:
        (response, invocations), pushed = loop.run_until_complete(_go())
    finally:
        loop.close()

    assert all(r["stream"] is True for r in requests)
    assert response.content == "I feel steady."
    assert "".join(pushed) == "I feel steady."
    assert [inv["name"] for inv in invocations] == ["get_emotional_state"]
    _assert_tool_result_was_sent_back(requests[1])
