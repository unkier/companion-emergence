"""OpenAI-compatible Chat Completions provider.

Talks to any server exposing ``POST {base_url}/chat/completions`` — a local
server (LM Studio, llama.cpp server, vLLM, Ollama's ``/v1``) or a hosted API
(OpenAI, OpenRouter, DeepSeek, Groq). The provider never runs tools: tool calls
come back on ``ChatResponse.tool_calls`` (or as ``ToolCallEvent`` when
streaming) and ``run_tool_loop`` dispatches them in-process, like Ollama.

Design: docs/source-spec/2026-09-26-openai-compat-provider-design.md
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from brain.bridge.model_tier import (
    TIER_ATTUNEMENT_DETECTOR,
    TIER_BACKGROUND_CLASSIFIER,
    TIER_BACKGROUND_GENERATIVE,
    TIER_BACKGROUND_HOUSEKEEPING,
    TIER_COMPACTION,
    TIER_DEV_CLI,
    TIER_INTERACTIVE_CHAT,
    TIER_SELF_MODEL_ARTICULATE,
)
from brain.bridge.provider import ProviderError

logger = logging.getLogger(__name__)

PROVIDER_NAME = "openai-compat"
SECRETS_FILENAME = "provider_secrets.json"
API_KEY_ENV = "OPENAI_COMPAT_API_KEY"
_SECRETS_FIELD = "openai_compat_api_key"
_CONFIG_WHERE = "persona_config.json → openai_compat"

# The tiers that make LLM calls. The local-model tiers (embedding, reranker,
# relevance-judge) never go through a provider, so they can't be remapped here.
LLM_TIERS: frozenset[str] = frozenset(
    {
        TIER_INTERACTIVE_CHAT,
        TIER_ATTUNEMENT_DETECTOR,
        TIER_COMPACTION,
        TIER_SELF_MODEL_ARTICULATE,
        TIER_BACKGROUND_CLASSIFIER,
        TIER_BACKGROUND_GENERATIVE,
        TIER_BACKGROUND_HOUSEKEEPING,
        TIER_DEV_CLI,
    }
)


def _config_error(detail: str) -> ProviderError:
    return ProviderError("openai_compat_config", detail)


@dataclass(frozen=True)
class OpenAICompatSettings:
    """The validated ``openai_compat`` block from persona_config.json."""

    base_url: str
    default_model: str
    tier_models: dict[str, str] = field(default_factory=dict)
    vision: bool = False
    timeout_s: float = 300.0

    def model_for_tier(self, tier: str | None) -> str:
        """The model for ``tier``: its ``tier_models`` override, else the default."""
        if tier is not None and tier in self.tier_models:
            return self.tier_models[tier]
        return self.default_model

    @classmethod
    def from_config(cls, block: object) -> OpenAICompatSettings:
        """Validate the raw block. Raises ``ProviderError("openai_compat_config")``.

        Fails loudly on purpose: a persona set to openai-compat must never fall
        back to Claude because its endpoint block is broken.
        """
        if not isinstance(block, dict):
            raise _config_error(f"missing {_CONFIG_WHERE} block (it needs base_url and model)")
        base_url = block.get("base_url")
        if not isinstance(base_url, str) or not base_url.strip():
            raise _config_error(f"{_CONFIG_WHERE}.base_url must be a non-empty string")
        model = block.get("model")
        if not isinstance(model, str) or not model.strip():
            raise _config_error(f"{_CONFIG_WHERE}.model must be a non-empty string")
        raw_tiers = block.get("tier_models", {})
        if not isinstance(raw_tiers, dict):
            raise _config_error(f"{_CONFIG_WHERE}.tier_models must map tier names to model names")
        tier_models: dict[str, str] = {}
        for tier, tier_model in raw_tiers.items():
            if tier not in LLM_TIERS:
                logger.warning("openai_compat: ignoring unknown tier %r in tier_models", tier)
                continue
            if not isinstance(tier_model, str) or not tier_model.strip():
                raise _config_error(
                    f"{_CONFIG_WHERE}.tier_models[{tier!r}] must be a non-empty string"
                )
            tier_models[tier] = tier_model.strip()
        vision = block.get("vision", False)
        if not isinstance(vision, bool):
            raise _config_error(f"{_CONFIG_WHERE}.vision must be true or false")
        timeout_s = block.get("timeout_s", 300)
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, int | float) or timeout_s <= 0:
            raise _config_error(f"{_CONFIG_WHERE}.timeout_s must be a positive number")
        return cls(
            base_url=base_url.strip().rstrip("/"),
            default_model=model.strip(),
            tier_models=tier_models,
            vision=vision,
            timeout_s=float(timeout_s),
        )


def resolve_api_key(persona_dir: Path) -> str | None:
    """The API key: ``$OPENAI_COMPAT_API_KEY``, else provider_secrets.json, else None."""
    env_key = os.environ.get(API_KEY_ENV, "").strip()
    if env_key:
        return env_key
    path = persona_dir / SECRETS_FILENAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        # Never log the file's content — it holds the key.
        logger.warning("openai_compat: cannot read %s (%s)", path, type(exc).__name__)
        return None
    key = data.get(_SECRETS_FIELD) if isinstance(data, dict) else None
    return key.strip() if isinstance(key, str) and key.strip() else None


def write_api_key(persona_dir: Path, key: str | None) -> Path:
    """Store ``key`` owner-only (0600), or delete the file when ``key`` is empty."""
    path = persona_dir / SECRETS_FILENAME
    if not key:
        path.unlink(missing_ok=True)
        return path
    persona_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    # Create 0600 from the start so the key is never readable by others, even briefly.
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump({_SECRETS_FIELD: key}, fh)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    return path
