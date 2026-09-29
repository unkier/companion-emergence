"""run_endpoint_check — the four probes behind `nell provider check`."""

from __future__ import annotations

import json

import httpx

from brain.bridge.openai_compat import OpenAICompatProvider, OpenAICompatSettings
from brain.bridge.openai_compat_check import run_endpoint_check

_SETTINGS = OpenAICompatSettings.from_config({"base_url": "http://localhost:1234/v1", "model": "m"})


def _provider(handler) -> OpenAICompatProvider:
    return OpenAICompatProvider(_SETTINGS, "m", None, transport=httpx.MockTransport(handler))


def _handler(*, calls_tools: bool):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": []})
        body = json.loads(request.content)
        if body.get("stream"):
            sse = b'data: {"choices":[{"delta":{"content":"1 2 3"}}]}\n\ndata: [DONE]\n\n'
            return httpx.Response(200, content=sse)
        if body.get("tools") and calls_tools:
            call = {"id": "c1", "type": "function", "function": {"name": "get_time", "arguments": "{}"}}
            return httpx.Response(200, json={"choices": [{"message": {"content": None, "tool_calls": [call]}}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ready"}}]})

    return handler


def test_every_check_passes_against_a_capable_endpoint() -> None:
    results = run_endpoint_check(_provider(_handler(calls_tools=True)))
    assert [r.ok for r in results] == [True, True, True, True]


def test_tool_check_fails_when_the_model_answers_in_text() -> None:
    results = run_endpoint_check(_provider(_handler(calls_tools=False)))
    assert [r.ok for r in results] == [True, True, False, True]
    assert "tool calling" in results[2].detail


def test_an_unreachable_endpoint_fails_every_check() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    results = run_endpoint_check(_provider(handler))
    assert [r.ok for r in results] == [False, False, False, False]
    assert "openai_compat_request" in results[1].detail
