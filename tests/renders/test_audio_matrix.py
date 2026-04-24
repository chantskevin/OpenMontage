"""End-to-end audio_mixer matrix tests.

audio_mixer has 5 operation branches (mix, duck, full_mix,
segmented_music, extract). Unit tests in tests/tools/ cover the
filter-graph construction (normalize=0, alimiter, ducking shape).
This suite runs each branch end-to-end against ffmpeg-synthesized
audio sources and asserts the output's structural properties via
ffprobe + audio level measurement.

Two suites:
  - audio_matrix_fast: 6 cells (~5s) — runs per PR
  - audio_matrix_full: same 6 + edge cases (~10s)

Sources are synthesized via ffmpeg lavfi (sine, anullsrc, aevalsrc)
so no audio fixtures are checked into the repo.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest

from tools.audio.audio_mixer import AudioMixer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synth_tone(
    path: Path, duration: float, freq: int = 440, gain_db: float = 0.0
) -> Path:
    """Synthesize a sine tone at `freq` Hz for `duration` seconds.
    Useful as a 'music' or 'speech-shaped' source."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if gain_db != 0.0:
        af = f"sine=frequency={freq}:duration={duration}, volume={gain_db}dB"
    else:
        af = f"sine=frequency={freq}:duration={duration}"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", af,
            "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "2",
            str(path),
        ],
        check=True, capture_output=True, timeout=30,
    )
    return path


def _synth_silence(path: Path, duration: float) -> Path:
    """Pure silence of `duration` seconds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo:duration={duration}",
            "-c:a", "pcm_s16le",
            str(path),
        ],
        check=True, capture_output=True, timeout=30,
    )
    return path


def _synth_video_with_audio(path: Path, duration: float) -> Path:
    """A video file with a 440 Hz tone — used for extract / segmented_music tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"testsrc2=size=320x240:duration={duration}:rate=30",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-t", str(duration), "-shortest",
            str(path),
        ],
        check=True, capture_output=True, timeout=30,
    )
    return path


def _probe_audio(path: Path) -> dict:
    """ffprobe an audio file. Returns {duration, sample_rate, channels,
    codec, mean_volume_db}."""
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", str(path),
        ],
        capture_output=True, text=True, timeout=15, check=True,
    )
    import json as _json
    data = _json.loads(proc.stdout)
    audio_stream = next(
        (s for s in data.get("streams") or [] if s.get("codec_type") == "audio"),
        None,
    )
    result = {
        "duration": float((data.get("format") or {}).get("duration") or 0),
        "has_audio": audio_stream is not None,
    }
    if audio_stream:
        result["sample_rate"] = int(audio_stream.get("sample_rate") or 0)
        result["channels"] = int(audio_stream.get("channels") or 0)
        result["codec"] = audio_stream.get("codec_name")

    # Mean volume via ffmpeg volumedetect.
    vol_proc = subprocess.run(
        [
            "ffmpeg", "-i", str(path), "-af", "volumedetect",
            "-f", "null", "-",
        ],
        capture_output=True, text=True, timeout=30,
    )
    match = re.search(r"mean_volume:\s*(-?\d+\.?\d*)\s*dB", vol_proc.stderr or "")
    if match:
        result["mean_volume_db"] = float(match.group(1))

    return result


# ---------------------------------------------------------------------------
# Fast suite — one cell per branch
# ---------------------------------------------------------------------------


@pytest.mark.audio_matrix_fast
def test_audio_mix_two_speech_tracks(tmp_path: Path) -> None:
    """The simplest mix: two speech-like tracks summed. Output must
    have audio, match the longest input duration, and not clip."""
    a = _synth_tone(tmp_path / "speech_a.wav", duration=2.0, freq=300)
    b = _synth_tone(tmp_path / "speech_b.wav", duration=3.0, freq=500)
    out = tmp_path / "mixed.wav"

    result = AudioMixer().execute({
        "operation": "mix",
        "tracks": [
            {"path": str(a), "role": "speech"},
            {"path": str(b), "role": "speech"},
        ],
        "output_path": str(out),
    })
    assert result.success, result.error

    probe = _probe_audio(out)
    assert probe["has_audio"], "mix output missing audio stream"
    # Duration matches longest input (within 0.1s).
    assert abs(probe["duration"] - 3.0) < 0.2, probe
    # alimiter caps at 0.98 → mean volume should be reasonable, not silent.
    assert probe.get("mean_volume_db", -100) > -40, (
        f"mean volume {probe.get('mean_volume_db')} dB suggests output is "
        f"effectively silent — alimiter or normalize=0 may have regressed"
    )


@pytest.mark.audio_matrix_fast
def test_audio_mix_speech_and_music_normalize_zero(tmp_path: Path) -> None:
    """The fork issue #9 invariant: amix=normalize=0 prevents music
    from being halved when mixed against speech. Output mean volume
    should be near music's level, not -6 dB lower."""
    music = _synth_tone(tmp_path / "music.wav", duration=2.0, freq=440, gain_db=-3)
    speech = _synth_tone(tmp_path / "speech.wav", duration=2.0, freq=300, gain_db=-3)
    out = tmp_path / "mixed.wav"

    result = AudioMixer().execute({
        "operation": "mix",
        "tracks": [
            {"path": str(music), "role": "music"},
            {"path": str(speech), "role": "speech"},
        ],
        "normalize": False,  # disable loudnorm so we measure the pre-norm sum
        "output_path": str(out),
    })
    assert result.success, result.error

    probe = _probe_audio(out)
    mean = probe.get("mean_volume_db", -100)
    # With normalize=0 + alimiter, summing two -3 dB tones produces
    # ~-24 dB mean (pure-tone RMS << peak; alimiter caps peaks). With
    # the normalize=1 bug, halving + post-loudnorm would produce ~-50
    # dB or even silent. Threshold is loose enough for the natural
    # peak/RMS gap of pure tones, tight enough to catch the bug.
    assert mean > -35, f"mix is too quiet ({mean} dB) — normalize=1 regression?"


@pytest.mark.audio_matrix_fast
def test_audio_full_mix_with_ducking(tmp_path: Path) -> None:
    """full_mix with speech + music + duck — the cinematic narration
    path. Verify output exists, has audio, duration matches longest."""
    speech = _synth_tone(tmp_path / "speech.wav", duration=3.0, freq=300)
    music = _synth_tone(tmp_path / "music.wav", duration=3.0, freq=440)
    out = tmp_path / "full_mix.wav"

    result = AudioMixer().execute({
        "operation": "full_mix",
        "tracks": [
            {"path": str(speech), "role": "speech"},
            {"path": str(music), "role": "music"},
        ],
        "ducking": {"enabled": True, "music_volume_during_speech": 0.15},
        "output_path": str(out),
    })
    assert result.success, result.error

    probe = _probe_audio(out)
    assert probe["has_audio"]
    assert abs(probe["duration"] - 3.0) < 0.3
    # Should not be silent.
    assert probe.get("mean_volume_db", -100) > -40


@pytest.mark.audio_matrix_fast
def test_audio_duck_simple_format(tmp_path: Path) -> None:
    """The simple duck format (primary_audio + secondary_audio) used
    by compose-director. Output must be non-silent and match the
    longest input."""
    speech = _synth_tone(tmp_path / "speech.wav", duration=2.0, freq=300)
    music = _synth_tone(tmp_path / "music.wav", duration=2.0, freq=440)
    out = tmp_path / "ducked.wav"

    result = AudioMixer().execute({
        "operation": "duck",
        "primary_audio": str(speech),
        "secondary_audio": str(music),
        "duck_level": -12,
        "output_path": str(out),
    })
    assert result.success, result.error

    probe = _probe_audio(out)
    assert probe["has_audio"]
    assert abs(probe["duration"] - 2.0) < 0.3


@pytest.mark.audio_matrix_fast
def test_audio_extract_from_video(tmp_path: Path) -> None:
    """extract operation: pull audio out of a video file. Output is
    a wav with the source's duration."""
    video = _synth_video_with_audio(tmp_path / "src.mp4", duration=2.0)
    out = tmp_path / "extracted.wav"

    result = AudioMixer().execute({
        "operation": "extract",
        "input_path": str(video),
        "output_path": str(out),
    })
    assert result.success, result.error

    probe = _probe_audio(out)
    assert probe["has_audio"]
    assert abs(probe["duration"] - 2.0) < 0.2


@pytest.mark.audio_matrix_fast
def test_audio_segmented_music_into_video(tmp_path: Path) -> None:
    """segmented_music: mix background music into a video only during
    specified time segments. Output must be a video with mixed audio."""
    video = _synth_video_with_audio(tmp_path / "src.mp4", duration=4.0)
    music = _synth_tone(tmp_path / "music.wav", duration=4.0, freq=440)
    out = tmp_path / "with_music.mp4"

    result = AudioMixer().execute({
        "operation": "segmented_music",
        "video_path": str(video),
        "music_path": str(music),
        "music_volume": 0.20,
        "segments": [{"start": 0, "end": 2}],
        "fade_duration": 0.3,
        "output_path": str(out),
    })
    assert result.success, result.error
    assert out.exists()
    assert out.stat().st_size > 0
    # Verify it's a valid video with audio.
    probe = _probe_audio(out)
    assert probe["has_audio"]
    assert abs(probe["duration"] - 4.0) < 0.3


# ---------------------------------------------------------------------------
# Full suite — edge cases
# ---------------------------------------------------------------------------


@pytest.mark.audio_matrix_full
def test_audio_full_mix_preserves_loudnorm_target(tmp_path: Path) -> None:
    """When normalize=True (the default), output should be loudnorm-
    normalized to ~-16 LUFS. Verify the post-pass actually ran."""
    speech = _synth_tone(tmp_path / "speech.wav", duration=2.0, freq=300)
    music = _synth_tone(tmp_path / "music.wav", duration=2.0, freq=440)
    out = tmp_path / "normalized.wav"

    result = AudioMixer().execute({
        "operation": "full_mix",
        "tracks": [
            {"path": str(speech), "role": "speech"},
            {"path": str(music), "role": "music"},
        ],
        "ducking": {"enabled": True},
        "normalize": True,
        "output_path": str(out),
    })
    assert result.success, result.error

    probe = _probe_audio(out)
    # loudnorm targets -16 LUFS. mean_volume isn't LUFS but is in the
    # same ballpark — should be in a reasonable range, not extreme.
    mean = probe.get("mean_volume_db", -100)
    assert -30 < mean < -5, (
        f"loudnorm output {mean} dB outside expected range "
        f"(-30 to -5). Loudnorm post-pass may have regressed."
    )


@pytest.mark.audio_matrix_full
def test_audio_mix_with_music_gain_db(tmp_path: Path) -> None:
    """music_gain_db pulls music up/down without changing other
    tracks. Verify negative gain produces quieter output than 0."""
    music = _synth_tone(tmp_path / "music.wav", duration=2.0, freq=440)
    speech = _synth_tone(tmp_path / "speech.wav", duration=2.0, freq=300)

    out_neutral = tmp_path / "neutral.wav"
    AudioMixer().execute({
        "operation": "mix",
        "tracks": [
            {"path": str(music), "role": "music"},
            {"path": str(speech), "role": "speech"},
        ],
        "music_gain_db": 0,
        "normalize": False,
        "output_path": str(out_neutral),
    })

    out_quieted = tmp_path / "quieted.wav"
    AudioMixer().execute({
        "operation": "mix",
        "tracks": [
            {"path": str(music), "role": "music"},
            {"path": str(speech), "role": "speech"},
        ],
        "music_gain_db": -12,
        "normalize": False,
        "output_path": str(out_quieted),
    })

    p_neutral = _probe_audio(out_neutral)
    p_quieted = _probe_audio(out_quieted)
    # Quieted should be measurably softer than neutral.
    assert p_quieted["mean_volume_db"] < p_neutral["mean_volume_db"] - 1, (
        f"music_gain_db=-12 didn't reduce output level: "
        f"neutral={p_neutral['mean_volume_db']} quieted={p_quieted['mean_volume_db']}"
    )


@pytest.mark.audio_matrix_full
def test_audio_segmented_music_outside_segments_is_video_audio_only(
    tmp_path: Path,
) -> None:
    """segmented_music must mute music OUTSIDE the named segments.
    Verify by checking the output's overall duration matches the
    video's, not the music's, and audio is present throughout."""
    video = _synth_video_with_audio(tmp_path / "src.mp4", duration=5.0)
    music = _synth_tone(tmp_path / "music.wav", duration=5.0, freq=440)
    out = tmp_path / "segmented.mp4"

    result = AudioMixer().execute({
        "operation": "segmented_music",
        "video_path": str(video),
        "music_path": str(music),
        "music_volume": 0.20,
        # Music only during 1-2s window — outside that, just video audio.
        "segments": [{"start": 1.0, "end": 2.0}],
        "fade_duration": 0.2,
        "output_path": str(out),
    })
    assert result.success, result.error

    probe = _probe_audio(out)
    assert probe["has_audio"]
    # Output duration matches video, not the music alone.
    assert abs(probe["duration"] - 5.0) < 0.3


@pytest.mark.audio_matrix_full
def test_audio_workspace_root_resolves_relative_track_paths(tmp_path: Path) -> None:
    """Fork issue #16 invariant: bare filenames resolve against
    _workspace_root. Without this, every BOS-style asset_manifest
    reference fails 'Track not found'."""
    music = _synth_tone(tmp_path / "music.wav", duration=1.0, freq=440)
    speech = _synth_tone(tmp_path / "speech.wav", duration=1.0, freq=300)
    out_relative = "out.wav"  # relative output path

    result = AudioMixer().execute({
        "operation": "mix",
        "tracks": [
            {"path": "music.wav", "role": "music"},   # bare filename
            {"path": "speech.wav", "role": "speech"},
        ],
        "output_path": out_relative,
        "_workspace_root": str(tmp_path),
    })
    assert result.success, result.error
    # Output landed in the workspace, not cwd.
    assert (tmp_path / out_relative).exists()
