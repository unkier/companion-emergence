"""PersonaConfig carries the openai-compat provider + its raw endpoint block."""

from __future__ import annotations

import json
from pathlib import Path

from brain.persona_config import KNOWN_PROVIDERS, PersonaConfig

_BLOCK = {
    "base_url": "http://localhost:1234/v1",
    "model": "qwen3-32b",
    "tier_models": {"compaction": "qwen3-4b"},
    "vision": True,
    "timeout_s": 120,
}


def _write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def test_openai_compat_is_a_known_provider() -> None:
    assert "openai-compat" in KNOWN_PROVIDERS


def test_openai_compat_provider_loads_without_falling_back(tmp_path: Path) -> None:
    path = tmp_path / "persona_config.json"
    _write(path, {"provider": "openai-compat", "openai_compat": _BLOCK})
    cfg = PersonaConfig.load(path)
    assert cfg.provider == "openai-compat"
    assert cfg.openai_compat == _BLOCK


def test_openai_compat_block_round_trips_through_save(tmp_path: Path) -> None:
    path = tmp_path / "persona_config.json"
    PersonaConfig(provider="openai-compat", openai_compat=_BLOCK).save(path)
    assert json.loads(path.read_text(encoding="utf-8"))["openai_compat"] == _BLOCK
    assert PersonaConfig.load(path).openai_compat == _BLOCK


def test_openai_compat_block_survives_the_bridge_startup_save(tmp_path: Path) -> None:
    """The bridge does load → touch_last_opened → save on every startup."""
    path = tmp_path / "persona_config.json"
    _write(path, {"provider": "openai-compat", "openai_compat": _BLOCK})
    cfg = PersonaConfig.load(path)
    cfg.touch_last_opened()
    cfg.save(path)
    assert PersonaConfig.load(path).openai_compat == _BLOCK


def test_non_dict_openai_compat_block_is_dropped(tmp_path: Path) -> None:
    path = tmp_path / "persona_config.json"
    _write(path, {"provider": "openai-compat", "openai_compat": "http://x"})
    assert PersonaConfig.load(path).openai_compat is None


def test_default_config_has_no_openai_compat_block() -> None:
    assert PersonaConfig().openai_compat is None


def test_save_omits_the_key_when_no_block_is_set(tmp_path: Path) -> None:
    """Other personas' config files keep their exact key set."""
    path = tmp_path / "persona_config.json"
    PersonaConfig().save(path)
    assert "openai_compat" not in json.loads(path.read_text(encoding="utf-8"))
