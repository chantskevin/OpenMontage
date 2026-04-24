"""Contract test for fork issue #34 — proposal-director default audio_treatment.

When the brief is ambiguous (no explicit audio cue), proposal-director
used to pick `music_only` and rationalize it post-hoc. Every cinematic
brand brief without "narration" / "voiceover" hint shipped silent.

This test asserts the skill now defaults to `voice_led` for ambiguous
briefs and requires explicit signals for `music_only`.
"""

from __future__ import annotations

from pathlib import Path

import pytest


SKILL_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "skills" / "pipelines" / "cinematic" / "proposal-director.md"
)


def _read_skill() -> str:
    return SKILL_PATH.read_text(encoding="utf-8")


def test_skill_names_fork_issue_34() -> None:
    """The HARD RULE block must reference the issue so a reader
    tracing the rule's history can find the original failure."""
    body = _read_skill()
    assert "fork issue #34" in body, (
        "proposal-director must reference fork issue #34 in the "
        "default audio_treatment guidance."
    )


def test_skill_defaults_to_voice_led_for_ambiguous_briefs() -> None:
    """The mode-selection table must explicitly tell the director to
    pick voice_led when the brief is ambiguous. Otherwise it'll
    silently default to music_only again."""
    body = _read_skill()
    # The ambiguous-default row (or equivalent prose).
    assert "ambiguous" in body.lower()
    # And the default has to be voice_led, not music_only.
    # Find the section near "ambiguous" and verify voice_led is named.
    idx = body.lower().find("ambiguous")
    nearby = body[idx:idx + 500]
    assert "voice_led" in nearby, (
        "ambiguous-brief default must be voice_led, not music_only "
        "(fork issue #34)"
    )


def test_skill_requires_explicit_signal_for_music_only() -> None:
    """music_only should require a concrete brief signal (named
    phrases like 'no narration', 'atmospheric', 'sound-design-led').
    Vague rationalizing about pacing is the failure mode."""
    body = _read_skill()
    # The "explicit signal" requirement must be named.
    assert "EXPLICIT" in body or "explicit user signal" in body.lower() or "explicit signal" in body.lower()
    # At least some of the canonical signal phrases should appear so
    # the director has examples to match against.
    signals = ["no narration", "atmospheric", "mood piece", "sound-design-led"]
    matched = sum(1 for s in signals if s in body.lower())
    assert matched >= 2, (
        f"skill should enumerate explicit-signal phrases for music_only "
        f"(matched {matched}/4: {signals}). Without examples directors "
        f"will keep inventing rationale."
    )


def test_skill_warns_against_post_hoc_rationale() -> None:
    """The "post-hoc rationale" failure pattern must be called out
    by name. Otherwise the director will rationalize the next default
    music_only pick the same way."""
    body = _read_skill()
    lower = body.lower()
    assert "post-hoc" in lower or "rationale" in lower
    # Specifically: "high-intensity pacing" was the post-hoc rationale
    # the original #34 director invented. Naming it as the
    # anti-pattern catches the next instance.
    assert "high-intensity pacing" in body or "narrative the director spins" in body or "arbitrary choice" in body


def test_skill_explains_the_one_way_door_consequence() -> None:
    """Picking music_only locks the entire downstream pipeline into
    a silent-narration shape. The skill should explain WHY this is
    serious — silent default isn't recoverable without a full re-run."""
    body = _read_skill()
    lower = body.lower()
    # Some phrasing about "can't easily fix later" / "re-run pipeline"
    # / "respect the lock".
    assert (
        "re-run the entire pipeline" in lower
        or "re-run the whole pipeline" in lower
        or "respect the lock" in lower
        or "can't easily salvage" in lower
        or "one-way door" in lower
    )


def test_skill_reviewer_check_has_teeth_for_unsupported_music_only() -> None:
    """A reviewer check must call out music_only proposals whose
    rationale doesn't quote a brief signal — otherwise the rule
    has no enforcement."""
    body = _read_skill()
    # CRITICAL finding for unsupported music_only.
    assert "CRITICAL" in body
    # The check text must mention music_only AND rationale tied to brief.
    # Find "music_only" near "CRITICAL".
    lower = body.lower()
    crit_idx = lower.find("critical finding")
    if crit_idx == -1:
        crit_idx = lower.find("critical")
    assert crit_idx >= 0
    # Within a few lines of CRITICAL there should be music_only context.
    nearby = body[max(0, crit_idx - 500):crit_idx + 500]
    assert "music_only" in nearby
