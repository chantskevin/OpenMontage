"""Regression tests for fork issue #7 — proper fix at the proposal layer.

The original #7 fix forced "narration is the default" in the cinematic
script and asset directors. That broke the design's actual intent:
cinematic embraces three audio treatments (voice_led, dialogue_led,
music_only), and the USER decides at the proposal checkpoint.
Research surfaces → proposal locks → script and asset honor the lock.

These tests verify the rebuilt contract:

  1. `proposal_packet.schema.json` declares `audio_treatment` with the
     canonical enum and requires `mode` + `rationale` when the object
     is present. Schema still validates proposals that omit it (for
     non-cinematic pipelines), but cinematic proposals must lock it
     per the director skill.
  2. Cinematic proposal-director has a HARD RULE block naming the
     three modes, mode-specific commitments, and the reviewer finding
     for a missing lock.
  3. Cinematic script-director dispatches on `audio_treatment.mode`
     rather than defaulting. The #7 "narration is the default" phrasing
     must NOT come back.
  4. Cinematic asset-director dispatches on the same lock and names
     reviewer checks that catch every cross-mode mismatch
     (voice_led with no narration, music_only with narration, etc).
  5. Schema validation still accepts a schema-complete proposal that
     has audio_treatment populated.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent.parent


def _read(p: str) -> str:
    return (ROOT / p).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Schema
# ---------------------------------------------------------------------------


def test_proposal_packet_schema_declares_audio_treatment() -> None:
    schema = json.loads(_read("schemas/artifacts/proposal_packet.schema.json"))
    at = schema["properties"]["production_plan"]["properties"].get("audio_treatment")
    assert at is not None, (
        "proposal_packet.production_plan must declare audio_treatment so "
        "the cinematic proposal director can lock it at the schema layer."
    )
    assert at["required"] == ["mode", "rationale"], (
        "audio_treatment.mode and audio_treatment.rationale must BOTH be "
        "required when the object is present — rationale is load-bearing "
        "for the reviewer check that distinguishes deliberate music-only "
        "from accidental narration drop."
    )
    modes = at["properties"]["mode"]["enum"]
    assert set(modes) == {"voice_led", "dialogue_led", "music_only"}


def test_schema_accepts_proposal_with_audio_treatment_locked() -> None:
    """Round-trip: a minimal schema-valid cinematic proposal packet with
    audio_treatment locked validates."""
    from jsonschema import Draft202012Validator

    schema = json.loads(_read("schemas/artifacts/proposal_packet.schema.json"))
    example = {
        "version": "1.0",
        "concept_options": [
            {
                "id": f"c{i}", "title": f"t{i}", "hook": "h",
                "narrative_structure": "story", "visual_approach": "v",
                "target_duration_seconds": 10, "why_this_works": "w",
            }
            for i in range(1, 4)
        ],
        "selected_concept": {"concept_id": "c1", "rationale": "r"},
        "production_plan": {
            "pipeline": "cinematic",
            "stages": [],
            "render_runtime": "remotion",
            "audio_treatment": {
                "mode": "voice_led",
                "rationale": "brand commercial, founder voice carries the message",
            },
            "voice_selection": {
                "provider": "elevenlabs",
                "voice_id": "Rachel",
                "rationale": "warm health-register",
            },
        },
        "cost_estimate": {
            "total_estimated_usd": 1.0,
            "line_items": [],
            "budget_verdict": "within_budget",
        },
        "approval": {"status": "approved"},
    }
    Draft202012Validator(schema).validate(example)


def test_schema_rejects_audio_treatment_with_missing_rationale() -> None:
    """`rationale` must not be droppable — otherwise a director could
    satisfy the contract with just {mode: music_only} and no reason,
    defeating the reviewer check."""
    from jsonschema import Draft202012Validator, ValidationError

    schema = json.loads(_read("schemas/artifacts/proposal_packet.schema.json"))
    bad = {
        "version": "1.0",
        "concept_options": [
            {"id": f"c{i}", "title": "t", "hook": "h",
             "narrative_structure": "story", "visual_approach": "v",
             "target_duration_seconds": 10, "why_this_works": "w"}
            for i in range(1, 4)
        ],
        "selected_concept": {"concept_id": "c1", "rationale": "r"},
        "production_plan": {
            "pipeline": "cinematic",
            "stages": [],
            "render_runtime": "remotion",
            "audio_treatment": {"mode": "music_only"},  # rationale missing
        },
        "cost_estimate": {"total_estimated_usd": 0, "line_items": [], "budget_verdict": "no_budget_set"},
        "approval": {"status": "pending"},
    }
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(bad)


def test_schema_rejects_invalid_audio_treatment_mode() -> None:
    from jsonschema import Draft202012Validator, ValidationError

    schema = json.loads(_read("schemas/artifacts/proposal_packet.schema.json"))
    bad = {
        "version": "1.0",
        "concept_options": [
            {"id": f"c{i}", "title": "t", "hook": "h",
             "narrative_structure": "story", "visual_approach": "v",
             "target_duration_seconds": 10, "why_this_works": "w"}
            for i in range(1, 4)
        ],
        "selected_concept": {"concept_id": "c1", "rationale": "r"},
        "production_plan": {
            "pipeline": "cinematic",
            "stages": [],
            "render_runtime": "remotion",
            "audio_treatment": {"mode": "silent", "rationale": "r"},  # not in enum
        },
        "cost_estimate": {"total_estimated_usd": 0, "line_items": [], "budget_verdict": "no_budget_set"},
        "approval": {"status": "pending"},
    }
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(bad)


# ---------------------------------------------------------------------------
# 2. Proposal director — HARD RULE
# ---------------------------------------------------------------------------


def test_proposal_director_has_audio_treatment_hard_rule() -> None:
    body = _read("skills/pipelines/cinematic/proposal-director.md")
    # The HARD RULE block explicitly names audio_treatment and the
    # three modes so the director reads it on a cold pass.
    assert "audio_treatment" in body
    assert "voice_led" in body and "dialogue_led" in body and "music_only" in body
    # Naming the enum by itself isn't enough — we need the decision
    # framed as a user-facing choice and escalated, not defaulted.
    assert "HARD RULE" in body
    # The reviewer-finding teeth must be named so a missed lock fails
    # post-stage review.
    assert "CRITICAL" in body or "critical" in body
    # Mode-specific commitments: voice_led must pair with
    # voice_selection, music_only must require rationale tied to user
    # sign-off.
    assert "voice_selection" in body


def test_proposal_director_names_decision_log_entry() -> None:
    """The choice must be auditable via decision_log. Reviewers use the
    log entry to distinguish 'deliberate music_only' from 'LLM
    defaulted'."""
    body = _read("skills/pipelines/cinematic/proposal-director.md")
    assert "audio_treatment_selection" in body or "decision_log" in body


# ---------------------------------------------------------------------------
# 3. Script director — reads the lock, doesn't re-decide
# ---------------------------------------------------------------------------


def test_script_director_reads_audio_treatment_instead_of_defaulting() -> None:
    body = _read("skills/pipelines/cinematic/script-director.md")
    # Must name the exact path the lock lives at.
    assert "audio_treatment" in body
    assert "proposal_packet" in body or "proposal" in body.lower()
    # Mode-specific behavior table present.
    assert "voice_led" in body and "dialogue_led" in body and "music_only" in body
    # Explicit "don't re-decide" instruction — the failure mode of #7
    # was the script stage re-deciding.
    assert "escalate" in body.lower() or "escalation" in body.lower()


def test_script_director_does_not_force_narration_as_default() -> None:
    """The original #7 fix's 'narration is the default' phrasing must
    NOT persist in the rebuilt skill. voice_led is ONE of three modes,
    not the baseline — the user picks at proposal stage."""
    body = _read("skills/pipelines/cinematic/script-director.md")
    lowered = body.lower()
    # Specific phrasings from the wrong-fix branch:
    assert "narration is the default" not in lowered, (
        "script-director must not declare narration as the default — "
        "that overrides the proposal stage's audio_treatment lock."
    )
    assert "cinematic is voice-led" not in lowered, (
        "script-director must not declare cinematic voice-led by default — "
        "it's mode-dependent per the proposal lock."
    )


def test_script_director_shows_canonical_music_only_shape() -> None:
    """Music-only is a first-class mode, not just an opt-out. The
    skill must show the canonical shape (empty text + tiled timings
    + silent_treatment_reason) so directors can emit schema-valid
    scripts in music_only mode without guessing."""
    body = _read("skills/pipelines/cinematic/script-director.md")
    assert "silent_treatment_reason" in body
    # A canonical music_only example appears somewhere.
    assert '"text": ""' in body or "empty strings" in body.lower() or "empty" in body.lower()


# ---------------------------------------------------------------------------
# 4. Asset director — dispatches on the same lock
# ---------------------------------------------------------------------------


def test_asset_director_dispatches_on_audio_treatment() -> None:
    body = _read("skills/pipelines/cinematic/asset-director.md")
    assert "audio_treatment" in body
    assert "voice_led" in body and "dialogue_led" in body and "music_only" in body
    # All three tool paths must be named.
    assert "tts_selector" in body
    # Under music_only, the skill must explicitly forbid calling TTS —
    # silently synthesizing narration the user opted out of is the
    # inverse failure of #7.
    assert "Do NOT" in body or "skip" in body.lower()


def test_asset_director_reviewer_checks_cover_every_mismatch() -> None:
    """For each mode, there's a concrete reviewer signal. Without this
    the skill is just hopeful prose."""
    body = _read("skills/pipelines/cinematic/asset-director.md")
    lowered = body.lower()
    # All three mismatches named:
    assert "voice_led" in body and "zero" in lowered
    assert "music_only" in body
    assert "dialogue_led" in body
    # The signal shape the reviewer greps for.
    assert 'type: "narration"' in body or 'type:"narration"' in body


def test_asset_director_does_not_declare_cinematic_voice_led_by_default() -> None:
    body = _read("skills/pipelines/cinematic/asset-director.md")
    assert "cinematic is voice-led" not in body.lower(), (
        "asset-director must not claim cinematic voice-led by default — "
        "the mode is picked at the proposal stage per audio_treatment."
    )
