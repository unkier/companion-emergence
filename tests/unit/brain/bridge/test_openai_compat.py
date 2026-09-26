"""Tests for brain/bridge/openai_compat.py — the OpenAI-compatible provider."""

from __future__ import annotations

import json
import logging
import os
import stat
import sys
from pathlib import Path

import pytest

from brain.bridge.openai_compat import (
    API_KEY_ENV,
    SECRETS_FILENAME,
    OpenAICompatSettings,
    resolve_api_key,
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
