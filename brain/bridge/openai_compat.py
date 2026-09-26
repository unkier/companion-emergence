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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from brain.bridge import provider_auth
from brain.bridge.chat import ChatMessage, ChatResponse, ImageBlock, TextBlock, ToolCall
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
    LLMProvider,
    ProviderError,
    _truncate_at_role_leak,
)
from brain.bridge.usage_log import log_usage
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


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class OpenAICompatProvider(LLMProvider):
    """LLMProvider for any OpenAI-compatible ``/chat/completions`` endpoint."""

    def __init__(
        self,
        settings: OpenAICompatSettings,
        model: str,
        api_key: str | None = None,
        *,
        persona_dir: Path | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._model = model
        self._api_key = api_key
        self._persona_dir = persona_dir
        self._transport = transport

    @property
    def base_url(self) -> str:
        return self._settings.base_url

    def name(self) -> str:
        return f"{PROVIDER_NAME}:{self._model}"

    def healthy(self) -> bool:
        """``GET {base_url}/models`` — True when the endpoint answers 2xx."""
        try:
            with self._client(timeout=5.0) as client:
                return client.get(self._url("/models")).is_success
        except httpx.HTTPError:
            return False

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        messages: list[ChatMessage] = []
        if system:
            messages.append(ChatMessage(role="system", content=system))
        messages.append(ChatMessage(role="user", content=prompt))
        return self._complete(messages, tools=None, options=None, call_type="generate").content

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
    ) -> ChatResponse:
        return self._complete(messages, tools=tools, options=options, call_type="chat")

    # -- internals ----------------------------------------------------------

    def _url(self, path: str) -> str:
        return f"{self._settings.base_url}{path}"

    def _client(self, *, timeout: float) -> httpx.Client:
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        # trust_env=False: a stray proxy env var must not reroute her LLM traffic.
        return httpx.Client(
            headers=headers, timeout=timeout, transport=self._transport, trust_env=False
        )

    def _payload(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None,
        options: dict[str, Any] | None,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": to_openai_messages(
                messages,
                vision=self._settings.vision,
                persona_dir=self._persona_dir,
                volatile_suffix=(options or {}).get("volatile_suffix"),
            ),
            "stream": stream,
            **generation_params(options),
        }
        if tools:
            payload["tools"] = tools
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    def _complete(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None,
        options: dict[str, Any] | None,
        call_type: str,
    ) -> ChatResponse:
        started = time.monotonic()
        data = self._post(self._payload(messages, tools, options, stream=False))
        message = _first_message(data)
        raw_calls = message.get("tool_calls")
        tool_calls = tuple(
            _parse_tool_call(raw, index)
            for index, raw in enumerate(raw_calls if isinstance(raw_calls, list) else [])
        )
        content = message.get("content")
        self._log_usage(call_type, data.get("usage"), started)
        return ChatResponse(
            content=clean_reply(content if isinstance(content, str) else ""),
            tool_calls=tool_calls,
            raw=data,
        )

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = self._url("/chat/completions")
        try:
            with self._client(timeout=self._settings.timeout_s) as client:
                resp = client.post(url, json=payload)
        except httpx.TimeoutException as exc:
            raise ProviderError(
                "openai_compat_timeout",
                f"no response from {url} within {self._settings.timeout_s:g}s",
            ) from exc
        except httpx.RequestError as exc:
            raise ProviderError(
                "openai_compat_request", f"cannot reach {url}: {type(exc).__name__}: {exc}"
            ) from exc
        if not resp.is_success:
            raise ProviderError(
                "openai_compat_http", http_error_detail(resp.status_code, url, resp.text)
            )
        try:
            data = resp.json()
        except ValueError as exc:
            raise ProviderError(
                "openai_compat_parse", f"invalid JSON from {url}: {resp.text[:200]}"
            ) from exc
        if not isinstance(data, dict):
            raise ProviderError("openai_compat_parse", f"response from {url} is not a JSON object")
        return data

    def _log_usage(self, call_type: str, usage: object, started: float) -> None:
        if self._persona_dir is None:
            return
        usage = usage if isinstance(usage, dict) else {}
        details = usage.get("prompt_tokens_details")
        log_usage(
            self._persona_dir,
            call_type=call_type,
            model=self._model,
            frame={
                "usage": {
                    "input_tokens": usage.get("prompt_tokens"),
                    "output_tokens": usage.get("completion_tokens"),
                    "cache_read_input_tokens": (
                        details.get("cached_tokens") if isinstance(details, dict) else None
                    ),
                },
                "duration_ms": int((time.monotonic() - started) * 1000),
            },
        )


def http_error_detail(status: int, url: str, body: str) -> str:
    """Human-readable detail for a non-2xx response.

    Never contains the provider_auth markers, so an endpoint auth failure can't
    set off the Claude re-login flow.
    """
    excerpt = body.strip()[:200]
    if provider_auth.is_auth_failure(excerpt):
        excerpt = "(response body omitted)"
    if status in (401, 403):
        return (
            f"rejected by endpoint (HTTP {status}) — check the API key in the persona's "
            f"{SECRETS_FILENAME} or ${API_KEY_ENV}: {excerpt}"
        )
    return f"HTTP {status} from {url}: {excerpt}"


def _first_message(data: dict[str, Any]) -> dict[str, Any]:
    choices = data.get("choices")
    first = choices[0] if isinstance(choices, list) and choices else None
    message = first.get("message") if isinstance(first, dict) else None
    if not isinstance(message, dict):
        error = data.get("error")
        hint = f" (endpoint error: {json.dumps(error)[:200]})" if error else ""
        raise ProviderError("openai_compat_parse", f"response has no choices[0].message{hint}")
    return message


def _parse_tool_call(raw: object, index: int) -> ToolCall:
    """Normalize one tool call (missing id, empty arguments) and parse it."""
    if not isinstance(raw, dict) or not isinstance(raw.get("function"), dict):
        raise ProviderError("openai_compat_parse", f"malformed tool_call in response: {raw!r}"[:300])
    function = dict(raw["function"])
    if function.get("arguments") in (None, ""):
        function["arguments"] = "{}"
    normalized = {"id": raw.get("id") or f"call_{index}", "function": function}
    try:
        return ToolCall.from_provider_dict(normalized)
    except ValueError as exc:
        raise ProviderError("openai_compat_parse", f"malformed tool_call in response: {exc}") from exc
