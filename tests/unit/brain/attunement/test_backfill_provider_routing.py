"""The backfill's default detector must carry persona_dir down to the model call,
so an openai-compat persona's history never goes to the Claude CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _openai_compat_persona(persona_dir: Path, n_turns: int = 15) -> None:
    block = {"base_url": "http://127.0.0.1:9/v1", "model": "m"}
    (persona_dir / "persona_config.json").write_text(
        json.dumps({"provider": "openai-compat", "openai_compat": block})
    )
    convs = persona_dir / "active_conversations"
    convs.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "id": f"m-{i}",
            "role": "user",
            "ts": f"2026-05-{(i % 28) + 1:02d}T12:00:00Z",
            "content": f"this is turn number {i} about various topics",
        }
        for i in range(n_turns)
    ]
    (convs / "main.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


@pytest.mark.parametrize("only_categories", [None, frozenset({"topic_affinity"})])
def test_backfill_default_detector_passes_persona_dir_to_the_model_call(
    tmp_path, monkeypatch, only_categories
):
    import brain.attunement.detector as det
    from brain.attunement.backfill import run_backfill
    from brain.bridge.provider import ClaudeCliProvider

    _openai_compat_persona(tmp_path)
    seen: list[object] = []

    def recording_call(system_prompt, user_message, *, persona_dir=None):
        seen.append(persona_dir)
        return ""

    def claude_must_not_run(*_a, **_k):
        raise AssertionError("an openai-compat persona must not use the Claude CLI")

    monkeypatch.setattr(det, "_call_haiku", recording_call)
    monkeypatch.setattr(ClaudeCliProvider, "generate", claude_must_not_run)

    run_backfill(tmp_path, only_categories=only_categories, delay_s=0)

    assert seen, "the backfill made no detector call"
    assert all(pd == tmp_path for pd in seen), seen
