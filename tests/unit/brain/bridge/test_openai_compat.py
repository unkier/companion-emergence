"""Tests for brain/bridge/openai_compat.py — the OpenAI-compatible provider."""

from __future__ import annotations

import base64
import json
import logging
import os
import stat
import sys
from pathlib import Path

import httpx
import pytest

from brain.bridge import provider_auth
from brain.bridge.chat import (
    ChatMessage,
    ImageBlock,
    StreamDone,
    StreamError,
    TextBlock,
    TextDelta,
    ToolCall,
    ToolCallEvent,
)
from brain.bridge.model_tier import TIER_BACKGROUND_CLASSIFIER, TIER_COMPACTION
from brain.bridge.openai_compat import (
    API_KEY_ENV,
    SECRETS_FILENAME,
    OpenAICompatProvider,
    OpenAICompatSettings,
    clean_reply,
    generation_params,
    resolve_api_key,
    to_openai_messages,
    write_api_key,
)
from brain.bridge.provider import ProviderError, get_provider

_MIN_BLOCK = {"base_url": "http://localhost:1234/v1/", "model": "main-model"}


# ── settings ────────────────────────────────────────────────────────────────


def test_settings_minimal_block_uses_defaults_and_strips_trailing_slash() -> None:
    s = OpenAICompatSettings.from_config(_MIN_BLOCK)
    assert s.base_url == "http://localhost:1234/v1"
    assert s.default_model == "main-model"
    assert s.tier_models == {}
    assert s.vision is False
    assert s.timeout_s == 300.0


@pytest.mark.parametrize(
    ("block", "fragment"),
    [
        (None, "missing"),
        ({"model": "m"}, "base_url"),
        ({"base_url": "  ", "model": "m"}, "base_url"),
        ({"base_url": "http://x/v1"}, "model"),
        ({**_MIN_BLOCK, "tier_models": ["compaction"]}, "tier_models"),
        ({**_MIN_BLOCK, "tier_models": {"compaction": ""}}, "compaction"),
        ({**_MIN_BLOCK, "vision": "yes"}, "vision"),
        ({**_MIN_BLOCK, "timeout_s": 0}, "timeout_s"),
        ({**_MIN_BLOCK, "timeout_s": True}, "timeout_s"),
    ],
)
def test_settings_invalid_block_raises_config_error(block: object, fragment: str) -> None:
    with pytest.raises(ProviderError) as info:
        OpenAICompatSettings.from_config(block)
    assert info.value.stage == "openai_compat_config"
    assert fragment in info.value.detail


def test_settings_tier_override_and_default_fallback() -> None:
    s = OpenAICompatSettings.from_config({**_MIN_BLOCK, "tier_models": {"compaction": "small"}})
    assert s.model_for_tier("compaction") == "small"
    assert s.model_for_tier("background-generative") == "main-model"
    assert s.model_for_tier(None) == "main-model"


def test_settings_drops_unknown_and_local_model_tiers(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="brain.bridge.openai_compat")
    s = OpenAICompatSettings.from_config(
        {**_MIN_BLOCK, "tier_models": {"embedding": "x", "no-such-tier": "y", "compaction": "small"}}
    )
    assert s.tier_models == {"compaction": "small"}
    assert "embedding" in caplog.text
    assert "no-such-tier" in caplog.text


# ── API key ─────────────────────────────────────────────────────────────────


def test_api_key_absent_returns_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    assert resolve_api_key(tmp_path) is None


def test_api_key_round_trips_through_secrets_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    write_api_key(tmp_path, "sk-file")
    assert resolve_api_key(tmp_path) == "sk-file"
    stored = json.loads((tmp_path / SECRETS_FILENAME).read_text(encoding="utf-8"))
    assert stored == {"openai_compat_api_key": "sk-file"}


def test_api_key_env_overrides_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_api_key(tmp_path, "sk-file")
    monkeypatch.setenv(API_KEY_ENV, "  sk-env\n")
    assert resolve_api_key(tmp_path) == "sk-env"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
def test_secrets_file_is_owner_only(tmp_path: Path) -> None:
    path = write_api_key(tmp_path, "sk-file")
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


def test_writing_an_empty_key_deletes_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    write_api_key(tmp_path, "sk-file")
    write_api_key(tmp_path, None)
    assert not (tmp_path / SECRETS_FILENAME).exists()
    assert resolve_api_key(tmp_path) is None


def test_corrupt_secrets_file_returns_none_without_logging_its_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="brain.bridge.openai_compat")
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    (tmp_path / SECRETS_FILENAME).write_text("sk-leaky-not-json", encoding="utf-8")
    assert resolve_api_key(tmp_path) is None
    assert "sk-leaky" not in caplog.text


# ── request translation ────────────────────────────────────────────────────


def _sys(text: str) -> ChatMessage:
    return ChatMessage(role="system", content=text)


def _user(text: str) -> ChatMessage:
    return ChatMessage(role="user", content=text)


def _translate(messages: list[ChatMessage], **kw: object) -> list[dict]:
    kw.setdefault("vision", False)
    kw.setdefault("persona_dir", None)
    return to_openai_messages(messages, **kw)  # type: ignore[arg-type]


def test_leading_system_messages_merge_into_one() -> None:
    out = _translate([_sys("a"), _sys("b"), _user("hi")])
    assert out == [{"role": "system", "content": "a\n\nb"}, {"role": "user", "content": "hi"}]


def test_late_system_message_becomes_a_user_note() -> None:
    out = _translate([_sys("a"), _user("hi"), _sys("late")])
    assert out[-1] == {"role": "user", "content": "[system note] late"}


def test_volatile_suffix_appends_to_the_last_user_message() -> None:
    msgs = [_sys("s"), _user("first"), ChatMessage(role="assistant", content="ok"), _user("second")]
    out = _translate(msgs, volatile_suffix="CTX")
    assert out[1]["content"] == "first"
    assert out[-1]["content"] == "second\n\nCTX"


def test_volatile_suffix_without_a_user_message_becomes_a_trailing_user_message() -> None:
    out = _translate([_sys("s")], volatile_suffix="CTX")
    assert out == [{"role": "system", "content": "s"}, {"role": "user", "content": "CTX"}]


def test_assistant_tool_calls_and_tool_results_translate() -> None:
    call = ToolCall(id="c1", name="search_memories", arguments={"query": "tea"})
    msgs = [
        _user("hi"),
        ChatMessage(role="assistant", content="", tool_calls=(call,)),
        ChatMessage(role="tool", content='{"results": []}', tool_call_id="c1"),
    ]
    out = _translate(msgs)
    assert out[1] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "search_memories", "arguments": '{"query": "tea"}'},
            }
        ],
    }
    assert out[2] == {"role": "tool", "tool_call_id": "c1", "content": '{"results": []}'}


def _image_tool_turn() -> list[ChatMessage]:
    result = {
        "path": "/tmp/cat.png",
        "image": {"media_type": "image/png", "data_b64": "QUJD", "size_bytes": 3},
    }
    call = ToolCall(id="c1", name="read_file", arguments={"path": "/tmp/cat.png"})
    return [
        _user("look at this"),
        ChatMessage(role="assistant", content="", tool_calls=(call,)),
        ChatMessage(role="tool", content=json.dumps(result), tool_call_id="c1"),
    ]


def test_tool_image_is_stripped_from_text_and_attached_when_vision_is_on() -> None:
    out = _translate(_image_tool_turn(), vision=True)
    assert "QUJD" not in out[2]["content"]
    assert json.loads(out[2]["content"])["image"] == "[attached below]"
    assert out[3]["role"] == "user"
    assert out[3]["content"][-1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,QUJD"},
    }


def test_tool_image_becomes_a_text_stub_when_vision_is_off() -> None:
    out = _translate(_image_tool_turn(), vision=False)
    assert "QUJD" not in json.dumps(out)
    assert json.loads(out[2]["content"])["image"] == "[image not viewable with this model]"
    assert len(out) == 3


def test_image_block_is_read_from_the_persona_images_dir(tmp_path: Path) -> None:
    from brain.images import compute_sha, image_path

    data = b"fake-png-bytes"
    sha = compute_sha(data)
    target = image_path(tmp_path, sha, "image/png")
    target.parent.mkdir(parents=True)
    target.write_bytes(data)
    msg = ChatMessage(
        role="user", content=(TextBlock(text="look"), ImageBlock(image_sha=sha, media_type="image/png"))
    )
    out = _translate([msg], vision=True, persona_dir=tmp_path)
    assert out[0]["content"][0] == {"type": "text", "text": "look"}
    expected = "data:image/png;base64," + base64.b64encode(data).decode("ascii")
    assert out[0]["content"][1] == {"type": "image_url", "image_url": {"url": expected}}


def test_missing_image_file_degrades_to_the_text_stub(tmp_path: Path) -> None:
    msg = ChatMessage(
        role="user",
        content=(TextBlock(text="look"), ImageBlock(image_sha="a" * 64, media_type="image/png")),
    )
    out = _translate([msg], vision=True, persona_dir=tmp_path)
    assert out[0]["content"] == "look\n[image not viewable with this model]"


def test_generation_params_forward_only_the_allow_list() -> None:
    options = {
        "persona_dir": "/x",
        "session_id": "s",
        "volatile_suffix": "v",
        "include_block_clock": False,
        "temperature": 0.7,
        "num_ctx": 8192,
    }
    assert generation_params(options) == {"temperature": 0.7}
    assert generation_params(None) == {}


# ── reply cleanup ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("plain reply", "plain reply"),
        ("<think>plan</think>Hello", "Hello"),
        ("Hi<think>unfinished", "Hi"),
        # Review Focus 2: the template injected <think>, so only </think> is in the text.
        ("reasoning here</think>\nHello", "Hello"),
        ("Hello\nUser: and then I said", "Hello"),
    ],
)
def test_clean_reply(raw: str, expected: str) -> None:
    assert clean_reply(raw) == expected


# ── provider: non-streaming ────────────────────────────────────────────────


def _provider(
    handler,
    *,
    api_key: str | None = None,
    persona_dir: Path | None = None,
    **block: object,
) -> OpenAICompatProvider:
    settings = OpenAICompatSettings.from_config({**_MIN_BLOCK, **block})
    return OpenAICompatProvider(
        settings,
        "main-model",
        api_key,
        persona_dir=persona_dir,
        transport=httpx.MockTransport(handler),
    )


def _completion(content: str | None = "hi", tool_calls: list | None = None, usage: dict | None = None) -> dict:
    message: dict = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    body: dict = {"choices": [{"index": 0, "message": message, "finish_reason": "stop"}]}
    if usage is not None:
        body["usage"] = usage
    return body


def test_chat_posts_to_chat_completions_and_returns_content() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=_completion("Hello there"))

    resp = _provider(handler).chat([_user("hi")])
    assert resp.content == "Hello there"
    assert resp.tool_calls == ()
    assert seen["url"] == "http://localhost:1234/v1/chat/completions"
    assert seen["body"]["model"] == "main-model"
    assert seen["body"]["stream"] is False
    assert "tools" not in seen["body"]
    assert seen["auth"] is None


def test_api_key_is_sent_as_a_bearer_token() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=_completion())

    _provider(handler, api_key="sk-test").chat([_user("hi")])
    assert seen["auth"] == "Bearer sk-test"


def test_tools_pass_through_and_tool_calls_are_parsed() -> None:
    tools = [{"type": "function", "function": {"name": "search_memories", "parameters": {"type": "object", "properties": {}}}}]
    raw_call = {"id": "c1", "type": "function", "function": {"name": "search_memories", "arguments": '{"query": "tea"}'}}

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["tools"] == tools
        return httpx.Response(200, json=_completion(None, tool_calls=[raw_call]))

    resp = _provider(handler).chat([_user("hi")], tools=tools)
    assert resp.content == ""
    assert resp.tool_calls == (ToolCall(id="c1", name="search_memories", arguments={"query": "tea"}),)


@pytest.mark.parametrize("function", [{"name": "get_emotional_state", "arguments": ""}, {"name": "get_emotional_state"}])
def test_no_argument_tool_call_normalizes_to_an_empty_dict(function: dict) -> None:
    """Review Focus 1: servers send "" (or nothing) for parameterless tools."""
    handler = lambda request: httpx.Response(200, json=_completion(None, tool_calls=[{"type": "function", "function": function}]))  # noqa: E731
    resp = _provider(handler).chat([_user("hi")])
    assert resp.tool_calls == (ToolCall(id="call_0", name="get_emotional_state", arguments={}),)


def test_malformed_tool_call_arguments_raise_a_parse_error() -> None:
    raw_call = {"id": "c1", "type": "function", "function": {"name": "x", "arguments": "{not json"}}
    handler = lambda request: httpx.Response(200, json=_completion(None, tool_calls=[raw_call]))  # noqa: E731
    with pytest.raises(ProviderError) as info:
        _provider(handler).chat([_user("hi")])
    assert info.value.stage == "openai_compat_parse"


def test_http_200_with_an_error_body_raises_a_parse_error_quoting_it() -> None:
    """Review Focus 5."""
    handler = lambda request: httpx.Response(200, json={"error": {"message": "model not loaded"}})  # noqa: E731
    with pytest.raises(ProviderError) as info:
        _provider(handler).chat([_user("hi")])
    assert info.value.stage == "openai_compat_parse"
    assert "model not loaded" in info.value.detail


def test_http_404_detail_names_the_full_url() -> None:
    """Review Focus 4: a base_url missing /v1 shows up as a 404 on a visible URL."""
    handler = lambda request: httpx.Response(404, text="Not Found")  # noqa: E731
    with pytest.raises(ProviderError) as info:
        _provider(handler).chat([_user("hi")])
    assert info.value.stage == "openai_compat_http"
    assert "http://localhost:1234/v1/chat/completions" in info.value.detail
    assert "404" in info.value.detail


def test_http_401_detail_points_at_the_key_and_never_trips_claude_auth_markers() -> None:
    body = {"error": {"message": "Failed to authenticate. Not logged in."}}
    handler = lambda request: httpx.Response(401, json=body)  # noqa: E731
    with pytest.raises(ProviderError) as info:
        _provider(handler, api_key="sk-secret").chat([_user("hi")])
    detail = info.value.detail
    assert info.value.stage == "openai_compat_http"
    assert SECRETS_FILENAME in detail
    assert API_KEY_ENV in detail
    assert not provider_auth.is_auth_failure(detail)
    assert "sk-secret" not in detail


def test_connection_failure_raises_a_request_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ProviderError) as info:
        _provider(handler).chat([_user("hi")])
    assert info.value.stage == "openai_compat_request"


def test_timeout_raises_a_timeout_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(ProviderError) as info:
        _provider(handler).chat([_user("hi")])
    assert info.value.stage == "openai_compat_timeout"


def test_generate_sends_system_and_user_and_logs_usage(tmp_path: Path) -> None:
    seen: dict = {}
    usage = {"prompt_tokens": 12, "completion_tokens": 3, "prompt_tokens_details": {"cached_tokens": 8}}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["messages"] = json.loads(request.content)["messages"]
        return httpx.Response(200, json=_completion("done", usage=usage))

    assert _provider(handler, persona_dir=tmp_path).generate("prompt", system="sys") == "done"
    assert seen["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "prompt"}]
    row = json.loads((tmp_path / "chat_usage.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert row["call_type"] == "generate"
    assert row["model"] == "main-model"
    assert (row["input_tokens"], row["output_tokens"], row["cache_read_input_tokens"]) == (12, 3, 8)


def test_chat_strips_context_options_and_applies_the_volatile_suffix() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_completion())

    options = {"persona_dir": "/x", "session_id": "s", "volatile_suffix": "CTX", "temperature": 0.2}
    _provider(handler).chat([_sys("s"), _user("hi")], options=options)
    body = seen["body"]
    assert "persona_dir" not in body
    assert "session_id" not in body
    assert body["temperature"] == 0.2
    assert body["messages"][-1]["content"] == "hi\n\nCTX"


def test_reply_reasoning_is_stripped() -> None:
    handler = lambda request: httpx.Response(200, json=_completion("<think>x</think>Hi"))  # noqa: E731
    assert _provider(handler).chat([_user("hi")]).content == "Hi"


def test_healthy_is_true_on_models_200_and_false_when_unreachable() -> None:
    def ok(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": []})

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    assert _provider(ok).healthy() is True
    assert _provider(down).healthy() is False


def test_name_includes_the_model() -> None:
    assert _provider(lambda request: httpx.Response(200)).name() == "openai-compat:main-model"


# ── provider: streaming ────────────────────────────────────────────────────


def _sse(*chunks: dict, done: bool = True) -> bytes:
    lines = [f"data: {json.dumps(chunk)}\n\n" for chunk in chunks]
    if done:
        lines.append("data: [DONE]\n\n")
    return "".join(lines).encode("utf-8")


def _delta(content: str | None = None, tool_calls: list | None = None) -> dict:
    delta: dict = {}
    if content is not None:
        delta["content"] = content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    return {"choices": [{"index": 0, "delta": delta}]}


def _stream(body: bytes, *, status: int = 200, persona_dir: Path | None = None) -> list:
    handler = lambda request: httpx.Response(  # noqa: E731
        status, content=body, headers={"content-type": "text/event-stream"}
    )
    return list(_provider(handler, persona_dir=persona_dir).chat_stream([_user("hi")]))


def _texts(events: list) -> str:
    return "".join(e.text for e in events if isinstance(e, TextDelta))


def test_tool_call_event_id_defaults_to_empty() -> None:
    assert ToolCallEvent(name="x").id == ""


def test_stream_yields_text_deltas_then_done() -> None:
    events = _stream(_sse(_delta("Hel"), _delta("lo")))
    assert [e.text for e in events if isinstance(e, TextDelta)] == ["Hel", "lo"]
    assert isinstance(events[-1], StreamDone)
    assert events[-1].content == "Hello"


def test_stream_request_sets_stream_and_asks_for_usage() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=_sse(_delta("ok")))

    list(_provider(handler).chat_stream([_user("hi")]))
    assert seen["body"]["stream"] is True
    assert seen["body"]["stream_options"] == {"include_usage": True}


def test_stream_assembles_a_tool_call_split_across_chunks() -> None:
    events = _stream(
        _sse(
            _delta(tool_calls=[{"index": 0, "id": "c1", "type": "function", "function": {"name": "search_memories", "arguments": ""}}]),
            _delta(tool_calls=[{"index": 0, "function": {"arguments": '{"query": '}}]),
            _delta(tool_calls=[{"index": 0, "function": {"arguments": '"tea"}'}}]),
        )
    )
    calls = [e for e in events if isinstance(e, ToolCallEvent)]
    assert calls == [ToolCallEvent(id="c1", name="search_memories", arguments={"query": "tea"})]
    assert isinstance(events[-1], StreamDone)


def test_stream_yields_parallel_tool_calls_in_index_order() -> None:
    events = _stream(
        _sse(
            _delta(tool_calls=[{"index": 1, "id": "c2", "function": {"name": "get_soul", "arguments": "{}"}}]),
            _delta(tool_calls=[{"index": 0, "id": "c1", "function": {"name": "get_body_state", "arguments": "{}"}}]),
        )
    )
    assert [e.id for e in events if isinstance(e, ToolCallEvent)] == ["c1", "c2"]


def test_stream_tool_fragments_without_index_are_assembled() -> None:
    """Review Focus 3."""
    events = _stream(
        _sse(
            _delta(tool_calls=[{"id": "c1", "function": {"name": "a", "arguments": "{}"}}]),
            _delta(tool_calls=[{"id": "c2", "function": {"name": "b", "arguments": '{"x"'}}]),
            _delta(tool_calls=[{"function": {"arguments": ": 1}"}}]),
        )
    )
    assert [e for e in events if isinstance(e, ToolCallEvent)] == [
        ToolCallEvent(id="c1", name="a", arguments={}),
        ToolCallEvent(id="c2", name="b", arguments={"x": 1}),
    ]


def test_stream_no_argument_tool_call_gets_an_empty_dict() -> None:
    """Review Focus 1 (streaming)."""
    events = _stream(_sse(_delta(tool_calls=[{"index": 0, "function": {"name": "get_emotional_state", "arguments": ""}}])))
    assert [e for e in events if isinstance(e, ToolCallEvent)] == [
        ToolCallEvent(id="call_0", name="get_emotional_state", arguments={})
    ]


def test_stream_drops_a_think_span_split_across_chunks() -> None:
    events = _stream(_sse(_delta("<thi"), _delta("nk>secret plan</th"), _delta("ink>Hi"), _delta(" there")))
    assert _texts(events) == "Hi there"
    assert events[-1].content == "Hi there"


def test_stream_keeps_text_that_only_looks_like_a_tag_start() -> None:
    assert _texts(_stream(_sse(_delta("a <th"), _delta("ree")))) == "a <three"


def test_stream_ignores_reasoning_content() -> None:
    reasoning = {"choices": [{"index": 0, "delta": {"reasoning_content": "hmm"}}]}
    assert _texts(_stream(_sse(reasoning, _delta("Hi")))) == "Hi"


def test_stream_skips_keepalive_comments() -> None:
    assert _texts(_stream(b": ping\n\n" + _sse(_delta("ok")))) == "ok"


def test_stream_http_error_yields_a_single_stream_error() -> None:
    events = _stream(b"boom", status=500)
    assert len(events) == 1
    assert isinstance(events[0], StreamError)
    assert events[0].stage == "openai_compat_http"
    assert "boom" in events[0].detail


def test_stream_mid_stream_error_chunk_yields_a_stream_error() -> None:
    """Review Focus 5 (streaming)."""
    events = _stream(_sse(_delta("par"), {"error": {"message": "overloaded"}}))
    assert isinstance(events[-1], StreamError)
    assert "overloaded" in events[-1].detail


def test_stream_invalid_chunk_yields_a_parse_error() -> None:
    events = _stream(b"data: {nope\n\n")
    assert isinstance(events[-1], StreamError)
    assert events[-1].stage == "openai_compat_parse"


def test_stream_malformed_tool_arguments_yield_a_parse_error() -> None:
    events = _stream(_sse(_delta(tool_calls=[{"index": 0, "function": {"name": "x", "arguments": "{bad"}}])))
    assert isinstance(events[-1], StreamError)
    assert events[-1].stage == "openai_compat_parse"


def test_stream_timeout_yields_a_timeout_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("idle", request=request)

    events = list(_provider(handler).chat_stream([_user("hi")]))
    assert len(events) == 1
    assert events[0].stage == "openai_compat_timeout"


def test_stream_logs_usage_from_the_final_chunk(tmp_path: Path) -> None:
    usage_chunk = {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 1}}
    _stream(_sse(_delta("hi"), usage_chunk), persona_dir=tmp_path)
    row = json.loads((tmp_path / "chat_usage.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert row["call_type"] == "chat_stream"
    assert (row["input_tokens"], row["output_tokens"]) == (5, 1)


def test_stream_no_index_fragments_repeating_the_same_id_are_one_call() -> None:
    """Review Focus 3: some servers repeat the call id on every fragment."""
    events = _stream(
        _sse(
            _delta(tool_calls=[{"id": "c1", "function": {"name": "search_memories", "arguments": ""}}]),
            _delta(tool_calls=[{"id": "c1", "function": {"arguments": '{"query": '}}]),
            _delta(tool_calls=[{"id": "c1", "function": {"arguments": '"tea"}'}}]),
        )
    )
    assert [e for e in events if isinstance(e, ToolCallEvent)] == [
        ToolCallEvent(id="c1", name="search_memories", arguments={"query": "tea"})
    ]


def test_stream_json_error_body_instead_of_sse_yields_a_stream_error() -> None:
    """Review Focus 5: HTTP 200, but the body is a JSON error, not an event stream."""
    events = _stream(json.dumps({"error": {"message": "model not loaded"}}).encode())
    assert len(events) == 1
    assert isinstance(events[0], StreamError)
    assert "model not loaded" in events[0].detail


def test_stream_recovers_a_plain_json_completion_when_the_server_ignores_stream() -> None:
    events = _stream(json.dumps(_completion("Hello there")).encode())
    assert _texts(events) == "Hello there"
    assert isinstance(events[-1], StreamDone)
    assert events[-1].content == "Hello there"


def test_stream_recovers_tool_calls_from_a_plain_json_completion() -> None:
    call = {"id": "c1", "type": "function", "function": {"name": "get_soul", "arguments": ""}}
    events = _stream(json.dumps(_completion(None, tool_calls=[call])).encode())
    assert [e for e in events if isinstance(e, ToolCallEvent)] == [
        ToolCallEvent(id="c1", name="get_soul", arguments={})
    ]
    assert isinstance(events[-1], StreamDone)


def test_stream_with_an_empty_body_yields_a_stream_error() -> None:
    events = _stream(b"")
    assert len(events) == 1
    assert isinstance(events[0], StreamError)
    assert events[0].stage == "openai_compat_parse"
    assert "no stream data" in events[0].detail


def test_stream_error_event_line_yields_a_stream_error() -> None:
    """Older llama.cpp builds report errors as ``error:`` lines."""
    events = _stream(b'error: {"code": 500, "message": "context size exceeded"}\n\n')
    assert len(events) == 1
    assert isinstance(events[0], StreamError)
    assert "context size exceeded" in events[0].detail


def test_stream_with_no_reply_and_no_tool_calls_yields_a_stream_error() -> None:
    """A reasoning model that hits max_tokens inside <think> leaves nothing to say;
    that must surface as an error naming the finish reason, not an empty bubble."""
    finish = {"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]}
    events = _stream(_sse(_delta("<think>still going"), finish))
    assert isinstance(events[-1], StreamError)
    assert events[-1].stage == "openai_compat_empty"
    assert "length" in events[-1].detail
    assert not any(isinstance(e, StreamDone) for e in events)


# ── factory ────────────────────────────────────────────────────────────────


def _openai_persona(tmp_path: Path, **block: object) -> Path:
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    config = {"provider": "openai-compat", "openai_compat": {**_MIN_BLOCK, **block}}
    (persona_dir / "persona_config.json").write_text(json.dumps(config), encoding="utf-8")
    return persona_dir


def test_get_provider_openai_compat_requires_persona_dir() -> None:
    with pytest.raises(ProviderError) as info:
        get_provider("openai-compat")
    assert info.value.stage == "openai_compat_config"


def test_get_provider_openai_compat_without_a_block_fails_loudly(tmp_path: Path) -> None:
    (tmp_path / "persona_config.json").write_text('{"provider": "openai-compat"}', encoding="utf-8")
    with pytest.raises(ProviderError) as info:
        get_provider("openai-compat", persona_dir=tmp_path)
    assert info.value.stage == "openai_compat_config"


def test_get_provider_openai_compat_picks_the_tier_model(tmp_path: Path) -> None:
    persona_dir = _openai_persona(tmp_path, tier_models={"compaction": "small-model"})
    provider = get_provider("openai-compat", persona_dir=persona_dir, tier=TIER_COMPACTION)
    assert isinstance(provider, OpenAICompatProvider)
    assert provider._model == "small-model"


def test_get_provider_openai_compat_ignores_a_claude_alias_override(tmp_path: Path) -> None:
    persona_dir = _openai_persona(tmp_path)
    provider = get_provider(
        "openai-compat",
        persona_dir=persona_dir,
        model_override="haiku",
        tier=TIER_BACKGROUND_CLASSIFIER,
    )
    assert provider._model == "main-model"


def test_get_provider_openai_compat_reads_the_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    persona_dir = _openai_persona(tmp_path)
    write_api_key(persona_dir, "sk-x")
    assert get_provider("openai-compat", persona_dir=persona_dir)._api_key == "sk-x"
