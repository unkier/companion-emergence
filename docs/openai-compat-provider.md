# Running a Kindled on an OpenAI-compatible endpoint

The `openai-compat` provider runs every model call a persona makes — chat,
dreams, reflexes, research, compaction, classifiers, the attunement detector —
against any server that speaks the OpenAI Chat Completions API. No Claude CLI
or Claude login is needed.

Works with local servers (LM Studio, llama.cpp `llama-server`, vLLM, Ollama's
`/v1` endpoint) and hosted APIs (OpenAI, OpenRouter, DeepSeek, Groq, ...).

## Setup

1. Stop the bridge (`nell supervisor stop --persona <name>`).
2. Edit `<persona>/persona_config.json` — set the provider and add an
   `openai_compat` block:

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

   Keep the file's other fields as they are.
3. If the endpoint needs a key: `nell provider set-key --persona <name>`
   (prompted, not echoed; stored in `<persona>/provider_secrets.json`, mode
   0600). `OPENAI_COMPAT_API_KEY` in the environment overrides the file.
4. Test it: `nell provider check --persona <name>`. All four checks should
   pass — especially **model calls a tool**.
5. Start the bridge again.

## Fields

| Field | Required | Default | Meaning |
|---|---|---|---|
| `base_url` | yes | — | Endpoint root including `/v1`, e.g. `https://openrouter.ai/api/v1` |
| `model` | yes | — | Model used for every tier not listed in `tier_models` |
| `tier_models` | no | `{}` | Per-tier model overrides (see below) |
| `vision` | no | `false` | Send images she opens as image parts; only enable for vision models |
| `timeout_s` | no | `300` | Request timeout; when streaming, the longest silence between chunks |

### Tiers

| Tier | What runs on it |
|---|---|
| `interactive-chat` | Live conversation in NellFace |
| `background-generative` | Dreams, reflexes, research, making |
| `background-classifier` | Extraction, emotion tagging, gates |
| `background-housekeeping` | Cheap maintenance passes |
| `compaction` | Summarising old conversation |
| `self-model-articulate` | Self-model write-ups |
| `attunement-detector` | Reading how you communicate |
| `dev-cli` | `nell dream` / `nell reflex` / … run by hand |

A small, fast model is usually fine for the classifier, housekeeping,
compaction and attunement tiers.

## Known limitations

- **Tool calling is essential.** Her memory, emotion, soul and file tools all
  depend on it. If `nell provider check` fails the tool check, pick another
  model (or a server/chat template with tool support enabled).
- **No chat-time web search/fetch.** Those are built into the Claude CLI.
  Background research still works.
- **The NellFace model picker is Claude-only** and will refuse to switch
  models for this persona — change `openai_compat.model` instead.
- **Reasoning models while streaming:** if your server leaves the model's
  reasoning in the reply text *without* an opening `<think>` tag, it can flash
  up in live chat while she replies. It is stripped from the saved reply, so it
  is gone when you reopen the conversation. Enable your server's reasoning
  parser (e.g. llama.cpp `--reasoning-format`, vLLM `--reasoning-parser`) so
  reasoning is sent separately; it is then ignored.
- The NellFace setup wizard only creates Claude personas; switch an existing
  persona as above.
