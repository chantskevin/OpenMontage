"""Multi-stage pipeline smoke — catches inter-stage shape bugs.

The render-matrix and audio-matrix suites cover individual tools end-to-end.
This suite stitches the artifact CHAIN: each stage's canonical artifact is
hand-crafted (no LLM directors), then the actual tools that consume them
are run in pipeline order, and the final output is probed for plausibility.

The bug class this catches: a stage's artifact passes its own schema but
shape-mismatches what the downstream tool reads. Fork issue #29 was exactly
this — asset-director emits cuts[].source as ID; compose treated it as a
literal path. Schema validation alone wouldn't have caught it because both
shapes are valid strings.

This isn't an end-to-end agent test — directors are explicitly out of scope.
It's a contract-chain smoke. ~10s to run, ffmpeg-only, no API calls.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

from schemas.artifacts import validate_artifact
from tools.audio.audio_mixer import AudioMixer
from tools.video.video_compose import VideoCompose


# ---------------------------------------------------------------------------
# Source synthesis (testsrc2 + sine — no API, no fixtures)
# ---------------------------------------------------------------------------


def _synth_video(path: Path, width: int, height: int, duration: float) -> Path:
    """testsrc2 video with no audio at the requested duration/dims."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i",
            f"testsrc2=size={width}x{height}:duration={duration}:rate=30",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-t", str(duration),
            str(path),
        ],
        check=True, capture_output=True, timeout=30,
    )
    return path


def _synth_speech_like(path: Path, duration: float, freq: int) -> Path:
    """Synth a speech-shaped tone (modulated sine) at the requested
    duration. Stands in for a real TTS narration clip."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i",
            f"sine=frequency={freq}:duration={duration}",
            "-c:a", "pcm_s16le", "-ar", "44100",
            str(path),
        ],
        check=True, capture_output=True, timeout=30,
    )
    return path


def _probe(path: Path) -> dict:
    """ffprobe a media file — returns duration + stream summary."""
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", str(path),
        ],
        capture_output=True, text=True, timeout=15, check=True,
    )
    data = json.loads(proc.stdout)
    streams = data.get("streams") or []
    return {
        "duration": float((data.get("format") or {}).get("duration") or 0),
        "video_count": sum(1 for s in streams if s.get("codec_type") == "video"),
        "audio_count": sum(1 for s in streams if s.get("codec_type") == "audio"),
        "video_codec": next(
            (s.get("codec_name") for s in streams if s.get("codec_type") == "video"),
            None,
        ),
        "audio_codec": next(
            (s.get("codec_name") for s in streams if s.get("codec_type") == "audio"),
            None,
        ),
    }


# ---------------------------------------------------------------------------
# Hand-crafted artifact builders — match each stage's canonical schema
# ---------------------------------------------------------------------------


def _build_proposal_packet() -> dict[str, Any]:
    """A schema-valid cinematic proposal_packet. voice_led with
    gemini TTS locked — what the proposal-director would emit after
    fork issues #17/#21 closed."""
    return {
        "version": "1.0",
        "concept_options": [
            {
                "id": f"c{i}", "title": f"Concept {i}",
                "hook": "A hook", "narrative_structure": "story",
                "visual_approach": "cinematic", "target_duration_seconds": 6,
                "why_this_works": "research-backed",
            }
            for i in range(1, 4)
        ],
        "selected_concept": {"concept_id": "c1", "rationale": "strongest"},
        "production_plan": {
            "pipeline": "cinematic",
            "stages": [],
            "render_runtime": "ffmpeg",
            "renderer_family": "cinematic-trailer",
            "audio_treatment": {
                "mode": "voice_led",
                "rationale": "brand commercial — voice carries the message",
            },
            "voice_selection": {
                "provider": "gemini",
                "voice_id": "Charon",
                "rationale": "warm authoritative",
            },
            "delivery_promise": {
                "promise_type": "motion_led",
                "motion_required": True,
                "tone_mode": "cinematic",
                "quality_floor": "presentable",
            },
        },
        "cost_estimate": {
            "total_estimated_usd": 0.5,
            "line_items": [], "budget_verdict": "within_budget",
        },
        "approval": {"status": "approved"},
    }


def _build_script(section_durations: list[float]) -> dict[str, Any]:
    """script with N sections, each with non-empty text. Used by
    asset stage to drive narration generation."""
    sections = []
    cursor = 0.0
    for i, dur in enumerate(section_durations):
        sections.append({
            "id": f"s{i + 1}",
            "text": f"Section {i + 1} narration line goes here.",
            "start_seconds": round(cursor, 3),
            "end_seconds": round(cursor + dur, 3),
            "speaker_directions": "neutral, warm",
        })
        cursor += dur
    return {
        "version": "1.0",
        "title": "Multi-stage smoke script",
        "total_duration_seconds": cursor,
        "sections": sections,
    }


def _build_scene_plan(section_ids: list[str], durations: list[float]) -> dict[str, Any]:
    """scene_plan tied to script sections via script_section_id."""
    scenes = []
    cursor = 0.0
    for sid, dur in zip(section_ids, durations):
        scenes.append({
            "id": f"scene-{sid}",
            "type": "broll",
            "description": f"B-roll for {sid}",
            "start_seconds": round(cursor, 3),
            "end_seconds": round(cursor + dur, 3),
            "script_section_id": sid,
        })
        cursor += dur
    return {"version": "1.0", "scenes": scenes}


def _build_asset_manifest(
    video_paths: list[Path],
    narration_paths: list[Path],
    music_path: Path,
    section_ids: list[str],
) -> dict[str, Any]:
    """asset_manifest with video + narration (per section) + music.
    Mirrors the shape asset-director emits after fork issues
    #19/#22/#23 closed."""
    assets = []
    for i, vp in enumerate(video_paths):
        assets.append({
            "id": f"v{i + 1}", "type": "video", "path": str(vp),
            "source_tool": "test_synth", "scene_id": section_ids[i],
        })
    for i, np_ in enumerate(narration_paths):
        assets.append({
            "id": f"n-{section_ids[i]}", "type": "narration", "path": str(np_),
            "source_tool": "test_synth", "scene_id": section_ids[i],
        })
    assets.append({
        "id": "m1", "type": "music", "path": str(music_path),
        "source_tool": "test_synth", "scene_id": section_ids[0],
    })
    return {"version": "1.0", "assets": assets}


def _build_edit_decisions(
    section_ids: list[str], durations: list[float]
) -> dict[str, Any]:
    """edit_decisions with cuts referencing video assets by ID +
    audio.narration referencing narration assets by ID + music."""
    cursor = 0.0
    cuts = []
    narration_segments = []
    for i, (sid, dur) in enumerate(zip(section_ids, durations)):
        cuts.append({
            "id": f"c{i + 1}",
            "source": f"v{i + 1}",  # ID reference (per #29 contract)
            "in_seconds": 0.0,
            "out_seconds": dur,
        })
        narration_segments.append({
            "asset_id": f"n-{sid}",
            "start_seconds": round(cursor, 3),
            "end_seconds": round(cursor + dur, 3),
        })
        cursor += dur
    return {
        "version": "1.0",
        "render_runtime": "ffmpeg",
        "renderer_family": "cinematic-trailer",
        "cuts": cuts,
        "audio": {
            "narration": {"segments": narration_segments},
            "music": {"asset_id": "m1", "volume": 0.15},
        },
    }


# ---------------------------------------------------------------------------
# 1. Each artifact validates against its schema
# ---------------------------------------------------------------------------


@pytest.mark.audio_matrix_fast  # reuse fast marker — runs in same suite
def test_smoke_each_handcrafted_artifact_passes_schema() -> None:
    """The first invariant: the artifact-builders produce
    schema-valid artifacts. If a schema tightens (like #21 made
    audio_treatment required for cinematic), this fires immediately
    so we update the smoke to match."""
    proposal = _build_proposal_packet()
    validate_artifact("proposal_packet", proposal)

    durations = [2.0, 2.0, 2.0]
    section_ids = ["s1", "s2", "s3"]
    script = _build_script(durations)
    validate_artifact("script", script)

    scene_plan = _build_scene_plan(section_ids, durations)
    validate_artifact("scene_plan", scene_plan)

    # Asset manifest needs real paths — use placeholder strings here
    # since this test is just schema-shape, not file existence.
    manifest = _build_asset_manifest(
        [Path("v1.mp4"), Path("v2.mp4"), Path("v3.mp4")],
        [Path("n1.wav"), Path("n2.wav"), Path("n3.wav")],
        Path("music.wav"),
        section_ids,
    )
    validate_artifact("asset_manifest", manifest)

    edit = _build_edit_decisions(section_ids, durations)
    validate_artifact("edit_decisions", edit)


# ---------------------------------------------------------------------------
# 2. Stage chain: audio_mixer + video_compose with the hand-crafted artifacts
# ---------------------------------------------------------------------------


@pytest.mark.audio_matrix_fast
def test_smoke_audio_mixer_to_video_compose_chain(tmp_path: Path) -> None:
    """The load-bearing smoke: build all 5 artifacts → synthesize
    sources → audio_mixer mixes narration + music → video_compose
    composites video + premixed audio → probe final mp4.

    This catches the inter-stage shape bug class (#29 et al). If
    asset-director emits cuts in shape X but compose reads cuts in
    shape Y, this test fires regardless of whether either schema
    independently validates."""
    section_ids = ["s1", "s2", "s3"]
    durations = [2.0, 2.0, 2.0]
    total = sum(durations)

    # --- Synthesize sources ---
    videos = [
        _synth_video(tmp_path / f"v{i+1}.mp4", 1280, 720, dur)
        for i, dur in enumerate(durations)
    ]
    narrations = [
        _synth_speech_like(tmp_path / f"n_{sid}.wav", dur, 300 + i * 50)
        for i, (sid, dur) in enumerate(zip(section_ids, durations))
    ]
    music = _synth_speech_like(tmp_path / "music.wav", total, 440)

    # --- Build the artifact chain ---
    proposal = _build_proposal_packet()
    script = _build_script(durations)
    scene_plan = _build_scene_plan(section_ids, durations)
    manifest = _build_asset_manifest(videos, narrations, music, section_ids)
    edit_decisions = _build_edit_decisions(section_ids, durations)
    # Validate each — if any drifts from schema, fail loud here, not
    # mysteriously inside a tool call.
    validate_artifact("proposal_packet", proposal)
    validate_artifact("script", script)
    validate_artifact("scene_plan", scene_plan)
    validate_artifact("asset_manifest", manifest)
    validate_artifact("edit_decisions", edit_decisions)

    # --- Stage 1: audio_mixer mixes narration + music ---
    # Mirrors what compose-director runs before video_compose. This
    # is the contract surface where audio_mixer must understand the
    # narration_segments shape from edit_decisions.
    mixed_audio = tmp_path / "mixed.wav"
    mix_result = AudioMixer().execute({
        "operation": "full_mix",
        "tracks": [
            *[
                {"path": str(n), "role": "speech",
                 "start_seconds": sum(durations[:i])}
                for i, n in enumerate(narrations)
            ],
            {"path": str(music), "role": "music"},
        ],
        "ducking": {"enabled": True, "music_volume_during_speech": 0.15},
        "normalize": True,
        "output_path": str(mixed_audio),
    })
    assert mix_result.success, f"audio_mixer failed: {mix_result.error}"
    assert mixed_audio.exists() and mixed_audio.stat().st_size > 0

    # --- Stage 2: video_compose with premixed audio ---
    output = tmp_path / "final.mp4"
    compose_result = VideoCompose().execute({
        "operation": "render",
        "edit_decisions": edit_decisions,
        "asset_manifest": manifest,
        "proposal_packet": proposal,
        "audio_path": str(mixed_audio),  # the premixed audio gets muxed in
        "output_path": str(output),
    })
    assert compose_result.success, (
        f"video_compose failed: {compose_result.error}\n\n"
        f"This is the inter-stage shape bug class. The artifacts each "
        f"validate independently but the chain fails — same shape as "
        f"fork issue #29."
    )

    # --- Probe the final output ---
    probe = _probe(output)
    assert probe["video_count"] >= 1, (
        f"final mp4 missing video stream: {probe}"
    )
    assert probe["audio_count"] >= 1, (
        f"final mp4 missing audio stream — the audio_mixer → video_compose "
        f"audio handoff regressed: {probe}"
    )
    assert abs(probe["duration"] - total) < 0.5, (
        f"final mp4 duration {probe['duration']}s differs from edit "
        f"timeline {total}s by more than 0.5s"
    )


# ---------------------------------------------------------------------------
# 3. Negative test — schema drift between proposal and asset surfaces
# ---------------------------------------------------------------------------


@pytest.mark.audio_matrix_fast
def test_smoke_voice_led_proposal_without_narration_blocks_render(
    tmp_path: Path,
) -> None:
    """The fork issue #22/#23 invariant tested at the chain level:
    a voice_led proposal with an asset_manifest missing narration
    must block at compose. The trio of guards (#19 cross-stage, #22
    compose-time, #23 set-coverage) should fire here."""
    section_ids = ["s1", "s2"]
    durations = [2.0, 2.0]
    videos = [
        _synth_video(tmp_path / f"v{i+1}.mp4", 1280, 720, dur)
        for i, dur in enumerate(durations)
    ]

    proposal = _build_proposal_packet()  # voice_led
    script = _build_script(durations)
    # Asset manifest with NO narration entries — the bug shape.
    manifest = {
        "version": "1.0",
        "assets": [
            {"id": f"v{i+1}", "type": "video", "path": str(vp),
             "source_tool": "test", "scene_id": section_ids[i]}
            for i, vp in enumerate(videos)
        ],
    }
    edit_decisions = _build_edit_decisions(section_ids, durations)

    result = VideoCompose().execute({
        "operation": "render",
        "edit_decisions": edit_decisions,
        "asset_manifest": manifest,
        "proposal_packet": proposal,
        "script": script,  # required for #23 per-section coverage check
        "output_path": str(tmp_path / "final.mp4"),
    })
    assert not result.success, (
        "voice_led + zero narration assets must block render. The "
        "#22/#23 guards exist precisely to prevent this silent ship."
    )
    err = (result.error or "").lower()
    assert "narration" in err or "voice_led" in err
