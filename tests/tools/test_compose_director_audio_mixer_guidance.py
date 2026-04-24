"""Contract test for fork issue #33 — compose-director audio_mixer guidance.

The compose-director skill used to under-specify the audio_mixer
operation choice, so directors picked `segmented_music` with one
segment per scene cut for continuous background music. That triggers
the "music dips at every cut" pumping bug because
`_segmented_music` applies a fade-in/out to EVERY segment.

This test asserts the skill now explicitly steers callers to the
right operation per intent and warns against the per-cut pattern.
"""

from __future__ import annotations

from pathlib import Path

import pytest


SKILL_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "skills" / "pipelines" / "cinematic" / "compose-director.md"
)


def _read_skill() -> str:
    return SKILL_PATH.read_text(encoding="utf-8")


def test_skill_names_fork_issue_33() -> None:
    """The HARD RULE block must reference the issue so a reader
    tracing the rule's history can find the original failure."""
    body = _read_skill()
    assert "fork issue #33" in body, (
        "compose-director must reference fork issue #33 in the audio_mixer "
        "operation-selection guidance — that's the original bug shape."
    )


def test_skill_warns_against_segmented_music_per_scene_cut() -> None:
    """The specific anti-pattern from #33 must be named: don't use
    `segmented_music` with one segment per scene cut for continuous
    music. Without explicit warning, directors will keep doing it."""
    body = _read_skill()
    lower = body.lower()
    # Some phrasing of the anti-pattern must be present.
    assert "do not use" in lower or "do not pick" in lower or "anti-pattern" in lower, (
        "skill must explicitly warn against the wrong segmented_music usage"
    )
    assert "segmented_music" in body
    # The "fade at every segment" detail should be named so directors
    # understand WHY the pattern fails.
    assert "fade" in lower
    assert "scene-cut" in lower or "scene cut" in lower or "every cut" in lower


def test_skill_lists_concrete_alternatives_for_continuous_music() -> None:
    """Per-intent operation table must include the three viable
    alternatives for continuous background music: audio_path direct,
    single-segment segmented_music, full_mix."""
    body = _read_skill()
    # The audio_path direct path (post-#32).
    assert "audio_path" in body
    # full_mix for ducked narration + music.
    assert "full_mix" in body
    # Single-segment segmented_music as an option.
    assert "single segment" in body.lower() or "SINGLE segment" in body or "one segment" in body.lower()


def test_skill_links_to_fix_32_for_audio_path_passthrough() -> None:
    """The "pass audio_path directly" recommendation only became
    viable after fix #32 muxed audio_path on the Remotion path. The
    skill should reference that so readers understand the prereq."""
    body = _read_skill()
    assert "fork issue #32" in body, (
        "skill should cite #32 — that's what made audio_path-direct viable "
        "on the cinematic-trailer (Remotion) path."
    )
