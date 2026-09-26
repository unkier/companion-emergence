# OpenAI-compatible provider — design

**Date:** 2026-09-26
**Branch:** `unkier/openai-compat-provider`
**Status:** approved in brainstorming, pending written-spec review

## Goal

Let a persona run on any OpenAI-compatible Chat Completions endpoint — a local
server (LM Studio, llama.cpp server, vLLM, Ollama's `/v1`) or a hosted API
(OpenAI, OpenRouter, DeepSeek, Groq) — **with no Claude CLI and no Claude login
required**. Every LLM call the brain makes for that persona goes through the new
provider.

### Decisions from brainstorming

| Question | Decision |
|---|---|
| Target backend | Generic: base URL + optional API key, nothing vendor-specific |
| Scope | Fully Claude-free — including the two sites that hard-code `ClaudeCliProvider` |
| Models | Full per-tier map, with a default model for unlisted tiers |
| API key | Separate owner-only secrets file; env var overrides |
| Live chat | Token streaming + image input (behind a `vision` flag) |
| Implementation | Hand-rolled `httpx` client in a new module — no new dependencies |
| NellFace wizard | Untouched; stays Claude-only per the 2026-05-07 project decision |

### Non-goals

- Changing any `claude-cli` or `ollama` behaviour.
- Any NellFace (TypeScript/Rust) change.
- Chat-time web search/fetch for non-Claude providers (the Claude CLI's built-in
  `WebSearch`/`WebFetch` have no equivalent here; background research keeps
  working through the `ddgs` searcher).
- Retries, rate-limit backoff, cost estimation.

## 1. Configuration

### `persona_config.json`

A new nested block. The top-level `model` field keeps its current meaning
(Claude alias, validated against `KNOWN_MODELS`) and is ignored by this
provider.

```json
{
  "provider": "openai-compat",
  "openai_compat": {
    "base_url": "http://localhost:1234/v1",
    "model": "qwen3-32b",
    "tier_models": {
      "compaction": "qwen3-4b",
      "background-classifier": "qwen3-4b",
      "attunement-detector": "qwen3-4b"
    },
    "vision": false,
    "timeout_s": 300
  }
}
```

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `base_url` | str | yes | — | Includes the `/v1` segment; trailing `/` stripped |
| `model` | str | yes | — | Used for every tier not in `tier_models` |
| `tier_models` | dict[str, str] | no | `{}` | Keys are tier names from `brain/bridge/model_tier.py` |
| `vision` | bool | no | `false` | Send images as `image_url` parts |
| `timeout_s` | number | no | `300` | Per-request timeout (non-streaming) and per-chunk idle timeout (streaming) |

Valid `tier_models` keys are the LLM tiers: `interactive-chat`,
`attunement-detector`, `compaction`, `self-model-articulate`,
`background-classifier`, `background-generative`, `background-housekeeping`,
`dev-cli`. Any other key — including the local-model tiers `embedding`,
`reranker`, `relevance-judge` — is dropped with a logged warning.

### `PersonaConfig` changes (`brain/persona_config.py`)

- `KNOWN_PROVIDERS` gains `"openai-compat"`.
- New field `openai_compat: dict | None = None`. `_parse_data` keeps it only
  when it is a dict; `save()` writes it back. This is required: the bridge
  calls `save()` on every startup (`touch_last_opened`), and `save()` only
  writes known fields, so an unmodelled block would be silently deleted.
- Semantic validation of the block does **not** happen in `PersonaConfig`
  (which degrades bad values to defaults). It happens in
  `OpenAICompatSettings.from_config()` (below), which fails loudly.

### `OpenAICompatSettings` (in the new module)

A frozen dataclass parsed from the raw block:
`base_url`, `default_model`, `tier_models`, `vision`, `timeout_s`, plus
`model_for_tier(tier: str | None) -> str` (tier override, else default).
`from_config(block)` raises `ProviderError("openai_compat_config", ...)` when the
block is missing, `base_url`/`model` are missing or empty, or a field has the
wrong type. The message names the offending field and the file to edit. There
is never a silent fallback to Claude.

### Secrets

- File: `<persona_dir>/provider_secrets.json`, content
  `{"openai_compat_api_key": "<key>"}`. Written with mode `0600` on POSIX (on
  Windows the file inherits the persona dir's user-profile ACL, like other
  persona state).
- Resolution order: env `OPENAI_COMPAT_API_KEY` → secrets file → none.
- No key → no `Authorization` header (local servers).
- Key present → `Authorization: Bearer <key>`.
- The key never appears in logs, `ProviderError` details, `name()`, or usage rows.

## 2. Provider module — `brain/bridge/openai_compat.py`

`OpenAICompatProvider(LLMProvider)`:

```python
OpenAICompatProvider(
    settings: OpenAICompatSettings,
    model: str,
    api_key: str | None,
    *,
    transport: httpx.BaseTransport | None = None,  # test seam
)
```

Each call opens a short-lived `httpx.Client(transport=transport, trust_env=False)`
— matching the relay client's `trust_env=False` so a stray proxy env var can't
reroute the brain's LLM traffic.

### Methods

| Method | Behaviour |
|---|---|
| `name()` | `openai-compat:<model>`; `self._model` is set so `model_label_for_provider` works unchanged |
| `healthy()` | `GET {base_url}/models`, 5 s timeout, `True` on 2xx |
| `generate(prompt, *, system=None)` | `chat([system?, user])`, returns `.content` |
| `chat(messages, *, tools=None, options=None)` | `POST {base_url}/chat/completions`, `stream: false` → `ChatResponse(content, tool_calls, raw)` |
| `chat_stream(messages, *, tools=None, options=None)` | Same request with `stream: true` and `stream_options: {"include_usage": true}`; yields `TextDelta`, `ToolCallEvent`, then `StreamDone` or `StreamError` |

The provider never dispatches tools. Tool calls are returned to
`run_tool_loop`, which dispatches them in-process (the existing Ollama path).

### Request translation (`ChatMessage` → OpenAI messages)

1. **System messages.** All *leading* `system` messages are joined (blank line
   between) into one first `system` message. Any later `system` message is sent
   as `role: "user"` with content prefixed `[system note] ` — several local
   chat templates reject a non-initial system message.
2. **`volatile_suffix`.** When `options["volatile_suffix"]` is set, it is
   appended (after a blank line) to the text of the **last** `user` message,
   mirroring where `_format_claude_print_prompt` places it. If there is no user
   message, it is sent as a trailing `user` message.
3. **Option stripping.** Keys in `_PROVIDER_CONTEXT_OPTION_KEYS` are removed.
   Of what remains, only `temperature`, `top_p`, `max_tokens`, `stop`, `seed`
   are forwarded as top-level request fields; anything else is dropped (debug
   log) rather than risking a 400 from a strict server.
4. **Assistant tool calls** →
   `{"id", "type": "function", "function": {"name", "arguments": json.dumps(args)}}`.
   Assistant content that is empty and carries tool calls is sent as `null`.
5. **Tool results** → `{"role": "tool", "tool_call_id", "content"}`.
6. **Images.**
   - A tool-result message whose JSON content has an `image` object (the
     `read_file` image payload: base64 `data` + `media_type`) is rewritten: the
     `image.data` field is removed from the tool text and replaced with
     `"image": "[attached below]"` (vision on) or
     `"image": "[image not viewable with this model]"` (vision off).
   - With `vision: true`, a `user` message follows the tool results of that
     turn, carrying one `image_url` part per image:
     `{"type": "image_url", "image_url": {"url": "data:<media_type>;base64,<data>"}}`.
   - Any `ImageBlock` in the history is handled the same way, with bytes read
     from `<persona_dir>/images/<sha>.<ext>` (`options["persona_dir"]`). If the
     file is missing, it degrades to the text stub.
7. **Tools.** The `tools` list is already OpenAI-shaped
   (`{"type": "function", "function": schema}` from `build_tools_list`) and
   is passed through unchanged. `tool_choice` is not sent (server default `auto`).

### Response handling

- **Content cleanup** (both paths): remove `<think>…</think>` spans (and an
  unterminated leading `<think>` span), ignore `reasoning_content` /
  `reasoning` fields, then apply `_truncate_at_role_leak` as Ollama does.
- **Non-streaming:** `choices[0].message` → content + `ToolCall.from_provider_dict`
  for each `tool_calls` entry. A malformed tool call →
  `ProviderError("openai_compat_parse", ...)`.
- **Streaming:** Server-Sent Events, one `data:` JSON per line, ending at
  `data: [DONE]`. Text deltas are yielded as they arrive, with the think-span
  filter applied incrementally so reasoning text never reaches the WS.
  Tool-call fragments are accumulated by `index` (`id` and `function.name` from
  the first fragment, `function.arguments` concatenated); each completed call
  is yielded as a `ToolCallEvent` when the stream ends. The final chunk's
  `usage` (if any) is captured. `StreamDone.content` is the cleaned full text.
  A missing `id` gets a generated `call_<n>`.
- **Idle timeout (streaming):** if no chunk arrives within `timeout_s`, yield
  `StreamError("openai_compat_timeout", ...)`.

### Errors

All failures raise (or, when streaming, yield) `ProviderError` / `StreamError` with one of:

| Stage | Cause |
|---|---|
| `openai_compat_config` | Missing/invalid `openai_compat` block |
| `openai_compat_request` | Connection refused, DNS, TLS |
| `openai_compat_timeout` | Request or stream-idle timeout |
| `openai_compat_http` | Non-2xx; detail is `"<status>: <first 200 chars of body>"` |
| `openai_compat_parse` | Invalid JSON / missing `choices` / malformed tool call |

401/403 detail reads `rejected by endpoint (HTTP 401) — check the API key in
<persona>/provider_secrets.json or OPENAI_COMPAT_API_KEY`. Detail strings must
not contain the `provider_auth` markers (`"oauth session expired"`,
`"not logged in"`, `"failed to authenticate"`), so an endpoint auth error never
triggers the Claude re-login UX. The response-body excerpt is included verbatim,
so the check is applied to the whole detail: if a marker appears in the body
excerpt, the excerpt is dropped.

### Usage logging

After each call (when `options["persona_dir"]` is present) the provider calls
the existing `log_usage(persona_dir, call_type="chat"|"chat_stream"|"generate",
model=self._model, frame=...)` with
`frame = {"usage": {"input_tokens": prompt_tokens, "output_tokens":
completion_tokens, "cache_read_input_tokens":
prompt_tokens_details.cached_tokens}, "duration_ms": <measured>}`.
`total_cost_usd` stays `None`.

## 3. Wiring

### Factory — `brain/bridge/provider.py:get_provider`

- New keyword `tier: str | None = None`.
- New branch `"openai-compat"`: requires `persona_dir` (else
  `ProviderError("openai_compat_config", "persona_dir required")`); loads
  `PersonaConfig`, builds `OpenAICompatSettings`, resolves the API key, and
  returns `OpenAICompatProvider(settings, settings.model_for_tier(tier), key)`.
  `model_override` (a Claude alias) is ignored on this branch.
- `claude-cli`, `ollama` and `fake` ignore `tier`, so their behaviour is unchanged.
- Docstring updated to list the new name.

### Tiers — `brain/bridge/model_tier.py`

- `build_tier_provider(persona_dir, tier)` passes `tier=tier`.
- `build_interactive_chat_provider(persona_dir)` passes
  `tier=TIER_INTERACTIVE_CHAT`.
- This covers all ~30 existing background/chat construction sites without
  editing them. Module docstring comments that describe the kinds as
  "claude-cli/ollama/fake" gain `openai-compat`.

### Hard-coded Claude sites

`model_tier.py` gains a public `provider_kind(persona_dir) -> str` that wraps
the existing private `_resolve_provider_kind`, so these sites don't reach into a
private helper.

- `brain/attunement/detector.py` (`_call_haiku`): if `provider_kind(persona_dir)`
  is `openai-compat`, use
  `build_tier_provider(persona_dir, TIER_ATTUNEMENT_DETECTOR)` and call
  `generate(user_message, system=system_prompt)`. Otherwise the existing
  `ClaudeCliProvider` code runs unchanged. The failure logging path is shared.
  With no `persona_dir` the existing Claude path runs.
- `brain/ingest/emotion_backfill.py` (`_make_default_tagger`, when
  `provider is None`): the same kind check, using `TIER_BACKGROUND_CLASSIFIER`.
  `_make_default_tagger` gains a `persona_dir: Path | None = None` parameter,
  passed by its only caller `run_emotion_backfill(persona_dir, ...)`. When it is
  `None`, the existing Claude default applies.

### CLI — `brain/cli.py`

- New helper `_cli_provider(provider_name, persona_dir)` that returns
  `get_provider(provider_name, persona_dir=persona_dir, tier=TIER_DEV_CLI)` for
  `openai-compat`, and `get_provider(provider_name)` otherwise. It replaces the
  six bare `get_provider(provider_name)` calls. Passing `persona_dir` to
  `claude-cli` would change the dev CLI's model resolution, so that stays as-is.
- `--provider` help strings list `openai-compat`.
- New subcommand group `nell provider`:
  - `set-key --persona NAME`: prompts with `getpass` (never argv, so the key
    stays out of shell history), writes `provider_secrets.json` with `0600`.
    An empty input deletes the key.
  - `check --persona NAME`: loads settings (config errors are printed), then
    reports each of: `GET /models` reachable; a one-message non-streaming chat
    returns text; a chat offering one trivial tool (`get_time`, schema-only, not
    dispatched) with a prompt that requires it — reports whether the model
    emitted a tool call; a streaming chat yields at least one text delta.
    Exit code 0 only if all pass. The tool probe is the key check, because weak
    local models often can't tool-call.

### Bridge / server

- `brain/bridge/chat.py`: `ToolCallEvent` gains `id: str = ""` (additive,
  defaulted; nothing constructs it today).
- `brain/bridge/server.py` `_StreamingProxy.chat()`: collect `ToolCallEvent`s
  into `ChatResponse.tool_calls` (`ToolCall(id=ev.id or f"call_{n}", name,
  arguments)`). Additive: `ClaudeCliProvider.chat_stream` never yields them.
  The class docstring is updated.
- `POST /persona/config/model`: when the persona's provider is `openai-compat`,
  return `409 {"ok": false, "error": "model_managed_in_config"}` before
  `KNOWN_MODELS` validation.
- `brain/bridge/persona_state.py`: for `openai-compat`, `out["model"]` is
  `openai_compat.model` (or `None` if the block is invalid);
  `_default_model_for("openai-compat")` returns `None`.

### Untouched

The NellFace app, the setup wizard, `OllamaProvider`, `ClaudeCliProvider`,
`provider_auth`, and all `claude-cli` code paths.

## 4. Testing

All HTTP is faked with `httpx.MockTransport` through the constructor seam.
Streaming fixtures are hand-written SSE transcripts in the OpenAI Chat
Completions wire format, kept inline in the test module. The work
is test-first (pytest). CI runs `uv run pytest` and `uv run ruff check .` on
macOS, Linux and Windows.

- `tests/unit/brain/bridge/test_openai_compat.py`
  - Settings: required fields, type errors, tier-key filtering, `model_for_tier`.
  - Key resolution: env over file, file mode `0600` (POSIX-only test), no key
    means no header.
  - Translation: system merge, `[system note]` for late system messages,
    `volatile_suffix` placement (with and without a user message), option
    stripping/allow-list, assistant tool calls, tool results.
  - Images: `read_file` payload stripped with vision on and off; `image_url`
    message placement; `ImageBlock` from the persona images dir; missing file
    falls back to the stub.
  - Response parsing: string vs dict arguments, malformed tool call, think-span
    and `reasoning_content` removal, role-leak truncation.
  - Streaming: text deltas; tool-call arguments split across chunks; two
    parallel tool calls; the usage chunk; `[DONE]`; think span split across
    chunks; idle timeout gives `StreamError`.
  - Errors: each stage; 401 wording contains no `provider_auth` marker.
  - `healthy()` true and false; usage row written with mapped fields.
- `tests/unit/brain/test_persona_config.py` (extend): the `openai_compat` block
  round-trips through `save()` and `touch_last_opened` + `save()`; the new
  provider name is accepted.
- `tests/unit/brain/bridge/test_model_tier.py` (extend): `openai-compat` tier
  mapping for interactive and background tiers; claude-cli assertions unchanged.
- Factory, detector, backfill and CLI helper routing by provider kind;
  `nell provider set-key` / `check` (check against a MockTransport server).
- `_StreamingProxy` collects tool calls; the existing streaming-proxy tests
  stay green.
- Model route returns 409; `persona_state` model reporting.
- Integration: `run_tool_loop` with the provider over a MockTransport server.
  The model calls `get_emotional_state`, the brain dispatches it, and the
  second response answers using it.
- Manual: `nell provider check` and a live NellFace chat against a real
  OpenAI-compatible server.

## 5. Documentation

- New `docs/openai-compat-provider.md`: setup (`persona_config.json` block,
  `nell provider set-key`, `nell provider check`), a field reference, the tier
  list, and known limitations (no chat-time web search; tool-calling quality
  depends on the model; the NellFace model picker is Claude-only).
- One-line pointers from `README.md` (the providers paragraph) and `INSTALL.md`.

## 6. Follow-up issues (file separately, not in this PR)

1. `OllamaProvider.chat_stream` signature doesn't match `_StreamingProxy`'s call
   (`tools=` kwarg, yields `str` not `TextDelta`), so live Ollama chat likely
   fails. It also drops `volatile_suffix` and sends `read_file` image base64 as
   tool text.
2. NellFace `ModelPicker` should hide or adapt for non-Claude providers.
3. Chat-time web search/fetch as a brain-side tool (on the existing `ddgs`
   searcher) so non-Claude providers regain it.
4. `read_file` has no read deny-list: it can read `~/.ssh`,
   `provider_secrets.json`, `~/.claude` credentials.
5. Offering `openai-compat` in the NellFace setup wizard needs a revisit of the
   2026-05-07 "wizard is claude-cli only" decision (Hana).
