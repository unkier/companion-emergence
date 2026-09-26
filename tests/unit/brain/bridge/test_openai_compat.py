"""Tests for brain/bridge/openai_compat.py — the OpenAI-compatible provider."""

from __future__ import annotations

import base64
import json
import logging
import os
import stat
import sys
from pathlib import Path

import pytest

from brain.bridge.chat import ChatMessage, ImageBlock, TextBlock, ToolCall
from brain.bridge.openai_compat import (
    API_KEY_ENV,
    SECRETS_FILENAME,
    OpenAICompatSettings,
    clean_reply,
    generation_params,
    resolve_api_key,
    to_openai_messages,
    write_api_key,
)
from brain.bridge.provider import ProviderError

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
