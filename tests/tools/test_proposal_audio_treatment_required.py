"""Regression tests for fork issue #21.

On BOS fork 8cda87e (post #19+#20), a fresh cinematic run shipped a
proposal checkpoint where production_plan was missing both
audio_treatment AND voice_selection entirely:

    production_plan keys: stages, pipeline, render_runtime,
                          renderer_family, delivery_promise

No audio_treatment, no voice_selection. The proposal-director's
Step 4d called the lock a HARD RULE in prose, but the schema didn't
encode it — so complete_stage accepted the checkpoint, asset-
director had nothing to read, and the final video shipped silent.
The user-facing lie was "the voice-processing module is temporarily
faulty" — a false explanation caused by an unlocked proposal.

This is the same shape of bug fork issue #20 solved at the compose
layer (false-success checkpoints), just shifted one stage earlier.

Fix layers:

  1. proposal_packet schema — production_plan gets two `if/then`
     conditionals: pipeline=="cinematic" requires audio_treatment;
     audio_treatment.mode=="voice_led" requires voice_selection.
  2. lib/checkpoint._validate_cross_artifact_invariants — redundant
     safety net enforcing the same requirements for callers that
     bypass jsonschema validation.
  3. proposal-director skill — new Step 9 "Before calling
     complete_stage — Proposal Lock Pre-check (HARD RULE)" block.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parent.parent.parent


def _read(p: str) -> str:
    return (ROOT / p).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Schema conditional
# ---------------------------------------------------------------------------


def _minimal_cinematic_proposal(
    include_audio_treatment: bool = True,
    include_voice_selection: bool = True,
    audio_mode: str = "voice_led",
) -> dict[str, Any]:
    """Minimal schema-valid cinematic proposal packet."""
    plan: dict[str, Any] = {
        "pipeline": "cinematic",
        "stages": [],
        "render_runtime": "remotion",
    }
    if include_audio_treatment:
        plan["audio_treatment"] = {
            "mode": audio_mode,
            "rationale": "brand commercial voice register",
        }
    if include_voice_selection:
        plan["voice_selection"] = {
            "provider": "gemini",
            "voice_id": "Charon",
            "provider_candidates": [
                {"provider": "gemini", "voice_id": "Charon",
                 "rationale": "warm authoritative"},
            ],
            "rationale": "warm authoritative",
        }
    return {
        "version": "1.0",
        "concept_options": [
            {"id": f"c{i}", "title": "t", "hook": "h",
             "narrative_structure": "story", "visual_approach": "v",
             "target_duration_seconds": 10, "why_this_works": "w"}
            for i in range(1, 4)
        ],
        "selected_concept": {"concept_id": "c1", "rationale": "r"},
        "production_plan": plan,
        "cost_estimate": {
            "total_estimated_usd": 0.5, "line_items": [],
            "budget_verdict": "within_budget",
        },
        "approval": {"status": "approved"},
    }


def test_schema_rejects_cinematic_proposal_without_audio_treatment() -> None:
    """The fork issue #21 repro: cinematic proposal missing
    audio_treatment must fail schema validation. Before this fix the
    schema accepted it and the silent-video cascade began."""
    from jsonschema import Draft202012Validator, ValidationError

    schema = json.loads(_read("schemas/artifacts/proposal_packet.schema.json"))
    bad = _minimal_cinematic_proposal(include_audio_treatment=False,
                                      include_voice_selection=False)
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(bad)


def test_schema_rejects_voice_led_without_voice_selection() -> None:
    """voice_led locks TTS as the audio source. Without voice_selection
    the asset stage has no provider/voice pair and falls back to
    guessing — reintroducing fork issue #17."""
    from jsonschema import Draft202012Validator, ValidationError

    schema = json.loads(_read("schemas/artifacts/proposal_packet.schema.json"))
    bad = _minimal_cinematic_proposal(include_voice_selection=False)
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(bad)


def test_schema_accepts_voice_led_with_voice_selection() -> None:
    """Happy path: cinematic + voice_led + voice_selection is the
    canonical shape."""
    from jsonschema import Draft202012Validator

    schema = json.loads(_read("schemas/artifacts/proposal_packet.schema.json"))
    Draft202012Validator(schema).validate(_minimal_cinematic_proposal())


def test_schema_accepts_music_only_without_voice_selection() -> None:
    """music_only doesn't need voice_selection — the conditional
    `audio_treatment.mode == voice_led` gate only fires for voice_led.
    A music_only proposal is a legitimate cinematic shape."""
    from jsonschema import Draft202012Validator

    schema = json.loads(_read("schemas/artifacts/proposal_packet.schema.json"))
    ok = _minimal_cinematic_proposal(
        include_voice_selection=False,
        audio_mode="music_only",
    )
    # Music-only still needs audio_treatment, just not voice_selection.
    Draft202012Validator(schema).validate(ok)


def test_schema_accepts_dialogue_led_without_voice_selection() -> None:
    """dialogue_led extracts audio from source footage — voice_selection
    is not applicable. Only voice_led requires voice_selection."""
    from jsonschema import Draft202012Validator

    schema = json.loads(_read("schemas/artifacts/proposal_packet.schema.json"))
    ok = _minimal_cinematic_proposal(
        include_voice_selection=False,
        audio_mode="dialogue_led",
    )
    Draft202012Validator(schema).validate(ok)


def test_schema_still_allows_non_cinematic_without_audio_treatment() -> None:
    """The audio_treatment requirement is conditional on
    pipeline=="cinematic". Other pipelines don't use audio_treatment
    yet, so they must still validate without it. Guards against
    overly-broad enforcement."""
    from jsonschema import Draft202012Validator

    schema = json.loads(_read("schemas/artifacts/proposal_packet.schema.json"))
    ok = _minimal_cinematic_proposal(
        include_audio_treatment=False,
        include_voice_selection=False,
    )
    ok["production_plan"]["pipeline"] = "animated-explainer"
    Draft202012Validator(schema).validate(ok)


# ---------------------------------------------------------------------------
# 2. Cross-artifact validation in lib/checkpoint
# ---------------------------------------------------------------------------


def test_write_checkpoint_rejects_cinematic_proposal_without_audio_treatment(
    tmp_path,
) -> None:
    """The fork issue #21 repro routed through the real write_checkpoint
    path. Schema validation is the first layer that fires — the
    cross-artifact validator is the redundant safety net for callers
    that bypass jsonschema. Either layer blocking the write is
    correct; only the no-block case reproduces the bug."""
    from lib.checkpoint import write_checkpoint, CheckpointValidationError

    proposal = _minimal_cinematic_proposal(
        include_audio_treatment=False,
        include_voice_selection=False,
    )

    with pytest.raises(CheckpointValidationError) as exc:
        write_checkpoint(
            pipeline_dir=tmp_path, project_id="job-21",
            stage="proposal", status="completed",
            artifacts={"proposal_packet": proposal},
            pipeline_type="cinematic",
        )
    msg = str(exc.value).lower()
    assert "audio_treatment" in msg


def test_write_checkpoint_rejects_voice_led_without_voice_selection(
    tmp_path,
) -> None:
    """Even with audio_treatment locked, voice_led without
    voice_selection must fail. The asset stage can't produce TTS
    without a provider pick."""
    from lib.checkpoint import write_checkpoint, CheckpointValidationError

    proposal = _minimal_cinematic_proposal(include_voice_selection=False)

    with pytest.raises(CheckpointValidationError) as exc:
        write_checkpoint(
            pipeline_dir=tmp_path, project_id="job-21b",
            stage="proposal", status="completed",
            artifacts={"proposal_packet": proposal},
            pipeline_type="cinematic",
        )
    msg = str(exc.value).lower()
    assert "voice_selection" in msg


def test_cross_artifact_check_fires_when_schema_is_bypassed() -> None:
    """The cross-artifact invariant is the redundant safety net. This
    test calls _validate_cross_artifact_invariants directly — the
    same path BOS uses when they validate via an alternate writer.
    Without this check, a caller that skipped jsonschema could land
    the fork issue #21 shape."""
    from lib.checkpoint import (
        _validate_cross_artifact_invariants,
        CheckpointValidationError,
    )

    proposal = _minimal_cinematic_proposal(
        include_audio_treatment=False,
        include_voice_selection=False,
    )

    with pytest.raises(CheckpointValidationError) as exc:
        _validate_cross_artifact_invariants(
            stage="proposal", status="completed",
            artifacts={"proposal_packet": proposal},
        )
    assert "audio_treatment" in str(exc.value)
    assert "#21" in str(exc.value) or "fork issue" in str(exc.value).lower()


def test_cross_artifact_check_catches_voice_led_without_voice_selection_direct() -> None:
    """Same as above but for the voice_led → voice_selection rule."""
    from lib.checkpoint import (
        _validate_cross_artifact_invariants,
        CheckpointValidationError,
    )

    proposal = _minimal_cinematic_proposal(include_voice_selection=False)

    with pytest.raises(CheckpointValidationError) as exc:
        _validate_cross_artifact_invariants(
            stage="proposal", status="completed",
            artifacts={"proposal_packet": proposal},
        )
    assert "voice_selection" in str(exc.value)
    assert "voice_led" in str(exc.value)


def test_failed_proposal_can_omit_proposal_packet(tmp_path) -> None:
    """A `failed` checkpoint is allowed to omit the canonical
    artifact — the director calls fail_stage when they detect the
    gap, before writing a malformed proposal_packet. The cross-check
    only fires for terminal statuses (completed/awaiting_human); the
    schema only fires when the artifact is present. Together they
    mean: you can't ship a completed cinematic proposal without
    audio_treatment, but you can fail one."""
    from lib.checkpoint import write_checkpoint

    write_checkpoint(
        pipeline_dir=tmp_path, project_id="job-21c",
        stage="proposal", status="failed",
        artifacts={},
        pipeline_type="cinematic",
        error="audio_treatment not locked by Step 4d",
    )


def test_cross_artifact_check_skips_non_terminal_status_with_partial_artifact() -> None:
    """Calling the validator directly (bypassing schema) with a
    non-terminal status + partial artifact must skip the check. This
    exercises the `if status not in terminal` guard at the top of
    _validate_cross_artifact_invariants."""
    from lib.checkpoint import _validate_cross_artifact_invariants

    proposal = _minimal_cinematic_proposal(
        include_audio_treatment=False,
        include_voice_selection=False,
    )

    # Must NOT raise — in_progress/failed statuses are exempt.
    for status in ("in_progress", "failed"):
        _validate_cross_artifact_invariants(
            stage="proposal", status=status,
            artifacts={"proposal_packet": proposal},
        )


def test_cross_artifact_check_allows_happy_path(tmp_path) -> None:
    """Sanity: the common case passes without noise."""
    from lib.checkpoint import write_checkpoint

    write_checkpoint(
        pipeline_dir=tmp_path, project_id="job-21d",
        stage="proposal", status="completed",
        artifacts={"proposal_packet": _minimal_cinematic_proposal()},
        pipeline_type="cinematic",
    )


# ---------------------------------------------------------------------------
# 3. Proposal-director skill — Step 9 pre-check HARD RULE
# ---------------------------------------------------------------------------


def test_proposal_director_has_complete_stage_precheck() -> None:
    """The skill must give the director a concrete pre-check list for
    the proposal stage, matching the same pattern compose-director
    got for fork issue #20."""
    body = _read("skills/pipelines/cinematic/proposal-director.md")
    assert "Proposal Lock Pre-check" in body or "Step 9" in body
    # The HARD RULE block must name fork issue #21 by number or
    # failure description.
    lowered = body.lower()
    assert "#21" in body or "fork issue 21" in lowered or "silent" in lowered
    # fail_stage vs complete_stage must be explicit.
    assert "fail_stage" in body and "complete_stage" in body


def test_proposal_director_precheck_names_audio_treatment_fields() -> None:
    """The pre-check must enumerate the specific fields the director
    needs to verify. Without naming them the list is hopeful prose."""
    body = _read("skills/pipelines/cinematic/proposal-director.md")
    # The five required fields in the pre-check must all be named.
    assert "audio_treatment.mode" in body
    assert "audio_treatment.rationale" in body or "rationale" in body
    assert "voice_selection" in body
    assert "render_runtime" in body
    assert "renderer_family" in body


def test_proposal_director_precheck_names_schema_enforcement() -> None:
    """The skill must point the director at the schema-level
    enforcement so they understand a bypass won't silently succeed."""
    body = _read("skills/pipelines/cinematic/proposal-director.md")
    assert "if/then" in body or "schema" in body.lower()
    assert "proposal_packet.schema.json" in body or "_validate_cross_artifact" in body
