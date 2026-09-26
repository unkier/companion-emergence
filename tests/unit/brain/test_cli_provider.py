"""`nell provider` commands + the dev-CLI provider helper."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from brain import cli
from brain.bridge.model_tier import TIER_DEV_CLI
from brain.bridge.openai_compat import API_KEY_ENV, SECRETS_FILENAME, resolve_api_key, write_api_key
from brain.bridge.openai_compat_check import CheckResult


@pytest.fixture
def persona_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    home = tmp_path / "home"
    persona_dir = home / "personas" / "nell"
    persona_dir.mkdir(parents=True)
    monkeypatch.setenv("KINDLED_HOME", str(home))
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    return persona_dir


def test_cli_provider_openai_compat_passes_persona_dir_and_the_dev_tier(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list = []
    monkeypatch.setattr(cli, "get_provider", lambda name, **kw: seen.append((name, kw)) or "P")
    assert cli._cli_provider("openai-compat", tmp_path) == "P"
    assert seen == [("openai-compat", {"persona_dir": tmp_path, "tier": TIER_DEV_CLI})]


def test_cli_provider_claude_keeps_the_bare_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list = []
    monkeypatch.setattr(cli, "get_provider", lambda name, **kw: seen.append((name, kw)) or "P")
    cli._cli_provider("claude-cli", tmp_path)
    assert seen == [("claude-cli", {})]


def test_set_key_stores_the_key_without_echoing_it(
    persona_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("getpass.getpass", lambda prompt="": "sk-typed")
    assert cli.main(["provider", "set-key", "--persona", "nell"]) == 0
    assert resolve_api_key(persona_dir) == "sk-typed"
    assert "sk-typed" not in capsys.readouterr().out


def test_set_key_empty_input_removes_the_key(
    persona_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_api_key(persona_dir, "sk-old")
    monkeypatch.setattr("getpass.getpass", lambda prompt="": "")
    assert cli.main(["provider", "set-key", "--persona", "nell"]) == 0
    assert not (persona_dir / SECRETS_FILENAME).exists()


def test_set_key_for_a_missing_persona_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path / "empty-home"))
    assert cli.main(["provider", "set-key", "--persona", "ghost"]) == 1


def test_check_reports_a_config_error(
    persona_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (persona_dir / "persona_config.json").write_text('{"provider": "openai-compat"}')
    assert cli.main(["provider", "check", "--persona", "nell"]) == 1
    assert "base_url" in capsys.readouterr().err


def test_check_prints_each_result_and_fails_when_one_fails(
    persona_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    block = {"base_url": "http://127.0.0.1:9/v1", "model": "m"}
    (persona_dir / "persona_config.json").write_text(
        json.dumps({"provider": "openai-compat", "openai_compat": block})
    )
    monkeypatch.setattr(
        "brain.bridge.openai_compat_check.run_endpoint_check",
        lambda provider: [
            CheckResult("endpoint reachable", True, ""),
            CheckResult("model calls a tool", False, "no tool call"),
        ],
    )
    assert cli.main(["provider", "check", "--persona", "nell"]) == 1
    out = capsys.readouterr().out
    assert "[PASS] endpoint reachable" in out
    assert "[FAIL] model calls a tool — no tool call" in out
