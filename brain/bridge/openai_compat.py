"""OpenAI-compatible Chat Completions provider.

Talks to any server exposing ``POST {base_url}/chat/completions`` — a local
server (LM Studio, llama.cpp server, vLLM, Ollama's ``/v1``) or a hosted API
(OpenAI, OpenRouter, DeepSeek, Groq). The provider never runs tools: tool calls
come back on ``ChatResponse.tool_calls`` (or as ``ToolCallEvent`` when
streaming) and ``run_tool_loop`` dispatches them in-process, like Ollama.

Design: docs/source-spec/2026-09-26-openai-compat-provider-design.md
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from brain.bridge.chat import ChatMessage, ImageBlock, TextBlock
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
from brain.bridge.provider import (
    _PROVIDER_CONTEXT_OPTION_KEYS,
    ProviderError,
    _truncate_at_role_leak,
)
from brain.images import read_image_bytes

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


# ---------------------------------------------------------------------------
# Request translation: brain ChatMessage list → Chat Completions `messages`
# ---------------------------------------------------------------------------

# Only these sampler options are forwarded — anything else could 400 a strict server.
_FORWARDED_OPTIONS = frozenset({"temperature", "top_p", "max_tokens", "stop", "seed"})
_IMAGE_ATTACHED = "[attached below]"
_IMAGE_UNVIEWABLE = "[image not viewable with this model]"


def generation_params(options: dict[str, Any] | None) -> dict[str, Any]:
    """Top-level request fields from ``options``: allow-listed sampler params only."""
    params: dict[str, Any] = {}
    for key, value in (options or {}).items():
        if key in _FORWARDED_OPTIONS:
            params[key] = value
        elif key not in _PROVIDER_CONTEXT_OPTION_KEYS:
            logger.debug("openai_compat: dropping unsupported option %r", key)
    return params


def to_openai_messages(
    messages: list[ChatMessage],
    *,
    vision: bool,
    persona_dir: Path | None,
    volatile_suffix: str | None = None,
) -> list[dict[str, Any]]:
    """Translate brain ChatMessages into Chat Completions ``messages``.

    Leading system messages merge into one; later ones become ``[system note]``
    user messages, because several local chat templates reject a non-initial
    system message. ``volatile_suffix`` (per-turn ambient context) is appended
    to the last user message, where the Claude path also places it.
    """
    out: list[dict[str, Any]] = []
    leading_system: list[str] = []
    in_leading_system = True
    pending_images: list[dict[str, Any]] = []

    def flush_images() -> None:
        # Tool messages can't carry images, so a vision turn follows them with one.
        if pending_images:
            intro = {"type": "text", "text": "(image from the tool result above)"}
            out.append({"role": "user", "content": [intro, *pending_images]})
            pending_images.clear()

    for msg in messages:
        if in_leading_system:
            if msg.role == "system":
                leading_system.append(msg.content_text())
                continue
            in_leading_system = False
            if leading_system:
                out.append({"role": "system", "content": "\n\n".join(leading_system)})
        if msg.role != "tool":
            flush_images()
        if msg.role == "system":
            out.append({"role": "user", "content": f"[system note] {msg.content_text()}"})
        elif msg.role == "tool":
            text, images = _split_tool_image(msg.content_text(), vision=vision)
            out.append({"role": "tool", "tool_call_id": msg.tool_call_id, "content": text})
            pending_images.extend(images)
        elif msg.role == "assistant":
            out.append(_assistant_entry(msg))
        else:
            content = _user_content(msg, vision=vision, persona_dir=persona_dir)
            out.append({"role": "user", "content": content})
    if in_leading_system and leading_system:
        out.append({"role": "system", "content": "\n\n".join(leading_system)})
    flush_images()
    if volatile_suffix:
        _append_to_last_user(out, volatile_suffix)
    return out


def _assistant_entry(msg: ChatMessage) -> dict[str, Any]:
    text = msg.content_text()
    entry: dict[str, Any] = {"role": "assistant", "content": text}
    if msg.tool_calls:
        entry["content"] = text or None
        entry["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                },
            }
            for call in msg.tool_calls
        ]
    return entry


def _user_content(
    msg: ChatMessage, *, vision: bool, persona_dir: Path | None
) -> str | list[dict[str, Any]]:
    blocks = msg.content_blocks()
    if not any(isinstance(block, ImageBlock) for block in blocks):
        return msg.content_text()
    parts: list[dict[str, Any]] = []
    for block in blocks:
        if isinstance(block, TextBlock):
            parts.append({"type": "text", "text": block.text})
            continue
        url = _image_block_url(block, persona_dir) if vision else None
        if url is None:
            parts.append({"type": "text", "text": _IMAGE_UNVIEWABLE})
        else:
            parts.append({"type": "image_url", "image_url": {"url": url}})
    if not any(part["type"] == "image_url" for part in parts):
        return "\n".join(part["text"] for part in parts)
    return parts


def _image_block_url(block: ImageBlock, persona_dir: Path | None) -> str | None:
    if persona_dir is None:
        return None
    try:
        data = read_image_bytes(persona_dir, block.image_sha, block.media_type)
    except (OSError, ValueError):
        return None
    return f"data:{block.media_type};base64,{base64.b64encode(data).decode('ascii')}"


def _split_tool_image(content: str, *, vision: bool) -> tuple[str, list[dict[str, Any]]]:
    """Pull read_file's base64 image out of a tool result — it is never sent as text."""
    try:
        payload = json.loads(content)
    except ValueError:
        return content, []
    image = payload.get("image") if isinstance(payload, dict) else None
    if not isinstance(image, dict) or not isinstance(image.get("data_b64"), str):
        return content, []
    media_type = image.get("media_type") or "image/png"
    payload["image"] = _IMAGE_ATTACHED if vision else _IMAGE_UNVIEWABLE
    parts: list[dict[str, Any]] = []
    if vision:
        url = f"data:{media_type};base64,{image['data_b64']}"
        parts.append({"type": "image_url", "image_url": {"url": url}})
    return json.dumps(payload, ensure_ascii=False), parts


def _append_to_last_user(out: list[dict[str, Any]], suffix: str) -> None:
    for entry in reversed(out):
        if entry["role"] != "user":
            continue
        content = entry["content"]
        if isinstance(content, str):
            entry["content"] = f"{content}\n\n{suffix}" if content else suffix
        else:
            entry["content"] = [*content, {"type": "text", "text": suffix}]
        return
    out.append({"role": "user", "content": suffix})


# ---------------------------------------------------------------------------
# Reply cleanup
# ---------------------------------------------------------------------------

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_THINK_SPAN = re.compile(r"<think>.*?</think>", re.DOTALL)


def clean_reply(text: str) -> str:
    """Strip reasoning spans and role-label overruns from a finished reply."""
    if _THINK_CLOSE in text and _THINK_OPEN not in text:
        # The chat template injected <think> server-side: all before </think> is reasoning.
        text = text.split(_THINK_CLOSE, 1)[1]
    text = _THINK_SPAN.sub("", text)
    open_at = text.find(_THINK_OPEN)
    if open_at != -1:
        text = text[:open_at]
    return _truncate_at_role_leak(text.strip())
