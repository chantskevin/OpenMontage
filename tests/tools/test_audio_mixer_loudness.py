"""Regression tests for fork issue #9.

audio_mixer's mix/full_mix operations used FFmpeg's `amix` filter with
the default `normalize=1`, which averages inputs by count. Mixing a
loud music track with Veo clips' near-silent ambient audio halved
the music's perceived loudness — exactly the "music is barely
audible" symptom the issue reports.

Fix:
  1. amix built with `normalize=0` so each input rides at its own
     level rather than being divided by input count.
  2. `alimiter=limit=0.98` after the sum so the un-normalized mix
     can't clip.
  3. New `music_gain_db` convenience input: applies per-role gain
     to music-role tracks before mixing, for callers that want to
     pull music above/below other layers without computing linear
     volume ratios.

These tests stub `run_command` and inspect the assembled ffmpeg
filter_complex to lock in the normalize=0 + alimiter shape, plus
cover the music_gain_db per-role plumbing.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from tools.audio.audio_mixer import AudioMixer
from tools.base_tool import ToolResult


def _make_fake_track(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00")
    return path


@pytest.fixture
def captured_cmd(monkeypatch):
    """Stub run_command to capture the ffmpeg command assembled by the
    mixer. Returns a list that gets appended to on each call."""
    calls: list[list[str]] = []

    def fake_run_command(self, cmd, *, timeout=None, cwd=None):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(AudioMixer, "run_command", fake_run_command)
    return calls


def _filter_complex(cmd: list[str]) -> str:
    return cmd[cmd.index("-filter_complex") + 1]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_schema_declares_music_gain_db_with_zero_default() -> None:
    schema = AudioMixer.input_schema
    prop = schema["properties"].get("music_gain_db")
    assert prop is not None, (
        "audio_mixer must declare music_gain_db — the convenience "
        "knob for the music-against-near-silent-video case."
    )
    assert prop["type"] == "number"
    assert prop.get("default") == 0


# ---------------------------------------------------------------------------
# Amix normalize=0 — the core of the issue #9 fix
# ---------------------------------------------------------------------------


def test_mix_uses_amix_normalize_zero(tmp_path, captured_cmd) -> None:
    """amix=normalize=0 keeps each track at its own level rather than
    dividing by input count. Regression guard against reverting to the
    historical default."""
    a = _make_fake_track(tmp_path / "music.wav")
    b = _make_fake_track(tmp_path / "video_audio.wav")

    AudioMixer().execute({
        "operation": "mix",
        "tracks": [
            {"path": str(a), "role": "music"},
            {"path": str(b), "role": "primary"},
        ],
        "output_path": str(tmp_path / "out.wav"),
    })

    assert captured_cmd, "mixer did not invoke ffmpeg"
    fc = _filter_complex(captured_cmd[0])
    assert "normalize=0" in fc, (
        f"amix must use normalize=0 so music isn't halved when mixed "
        f"against near-silent video audio. filter_complex: {fc}"
    )


def test_mix_adds_alimiter_after_amix(tmp_path, captured_cmd) -> None:
    """Without amix's normalize=1 dividing, peaks can clip. alimiter
    must catch them. If a future change removes the limiter this test
    fires."""
    a = _make_fake_track(tmp_path / "music.wav")
    b = _make_fake_track(tmp_path / "voice.wav")

    AudioMixer().execute({
        "operation": "mix",
        "tracks": [{"path": str(a)}, {"path": str(b)}],
        "output_path": str(tmp_path / "out.wav"),
    })

    fc = _filter_complex(captured_cmd[0])
    assert "alimiter" in fc, (
        f"post-amix alimiter must be present to guard against peaks "
        f"when amix has normalize=0. filter_complex: {fc}"
    )


def test_mix_preserves_loudnorm_when_normalize_true(tmp_path, captured_cmd) -> None:
    """The existing `normalize` input (boolean) controls loudnorm
    post-pass. Distinct from amix's internal `normalize` setting,
    which was the bug. Loudnorm still runs at -16 LUFS when the
    caller opts in."""
    a = _make_fake_track(tmp_path / "music.wav")

    AudioMixer().execute({
        "operation": "mix",
        "tracks": [{"path": str(a)}],
        "normalize": True,
        "output_path": str(tmp_path / "out.wav"),
    })

    fc = _filter_complex(captured_cmd[0])
    assert "loudnorm=I=-16" in fc


def test_full_mix_uses_amix_normalize_zero_in_no_duck_branch(
    tmp_path, captured_cmd
) -> None:
    """_full_mix has two branches: with ducking (sidechaincompress)
    and without. The no-duck branch used to amix-with-normalize-1.
    Both branches now respect normalize=0."""
    music = _make_fake_track(tmp_path / "music.wav")
    sfx = _make_fake_track(tmp_path / "sfx.wav")

    AudioMixer().execute({
        "operation": "full_mix",
        "tracks": [
            {"path": str(music), "role": "music"},
            {"path": str(sfx), "role": "sfx"},
        ],
        "ducking": {"enabled": False},
        "output_path": str(tmp_path / "out.wav"),
    })

    fc = _filter_complex(captured_cmd[0])
    assert "normalize=0" in fc
    assert "alimiter" in fc


# ---------------------------------------------------------------------------
# music_gain_db — per-role convenience gain
# ---------------------------------------------------------------------------


def test_music_gain_db_applies_volume_in_db_to_music_role(
    tmp_path, captured_cmd
) -> None:
    """Positive music_gain_db boosts music. Filter chain should
    include a volume=<db>dB filter on the music-role track."""
    music = _make_fake_track(tmp_path / "music.wav")
    voice = _make_fake_track(tmp_path / "voice.wav")

    AudioMixer().execute({
        "operation": "mix",
        "tracks": [
            {"path": str(music), "role": "music"},
            {"path": str(voice), "role": "primary"},
        ],
        "music_gain_db": 3,
        "output_path": str(tmp_path / "out.wav"),
    })

    fc = _filter_complex(captured_cmd[0])
    assert "volume=3.0dB" in fc or "volume=3dB" in fc, (
        f"music_gain_db=3 must produce `volume=3dB` on the music-role "
        f"filter chain. filter_complex: {fc}"
    )


def test_music_gain_db_does_not_touch_non_music_roles(
    tmp_path, captured_cmd
) -> None:
    """The gain knob is role-scoped to music / secondary. Speech and
    sfx must not receive the boost even when music_gain_db is set."""
    voice = _make_fake_track(tmp_path / "voice.wav")
    sfx = _make_fake_track(tmp_path / "sfx.wav")

    AudioMixer().execute({
        "operation": "mix",
        "tracks": [
            {"path": str(voice), "role": "primary"},
            {"path": str(sfx), "role": "sfx"},
        ],
        "music_gain_db": -6,
        "output_path": str(tmp_path / "out.wav"),
    })

    fc = _filter_complex(captured_cmd[0])
    # No dB-form volume filter — neither track is music-role.
    assert "dB" not in fc, (
        f"music_gain_db must not apply to speech/sfx tracks. "
        f"filter_complex: {fc}"
    )


def test_music_gain_db_default_zero_emits_no_volume_filter(
    tmp_path, captured_cmd
) -> None:
    """Default 0 → no extra volume filter emitted. Keeps filter graph
    clean for the common case."""
    music = _make_fake_track(tmp_path / "music.wav")

    AudioMixer().execute({
        "operation": "mix",
        "tracks": [{"path": str(music), "role": "music"}],
        "output_path": str(tmp_path / "out.wav"),
    })

    fc = _filter_complex(captured_cmd[0])
    assert "dB" not in fc, f"default music_gain_db=0 leaked: {fc}"


def test_music_gain_db_in_full_mix(tmp_path, captured_cmd) -> None:
    """full_mix must also honor music_gain_db on music-role tracks.
    Symmetry with _mix."""
    music = _make_fake_track(tmp_path / "music.wav")
    voice = _make_fake_track(tmp_path / "voice.wav")

    AudioMixer().execute({
        "operation": "full_mix",
        "tracks": [
            {"path": str(music), "role": "music"},
            {"path": str(voice), "role": "speech"},
        ],
        "music_gain_db": -6,
        "output_path": str(tmp_path / "out.wav"),
    })

    fc = _filter_complex(captured_cmd[0])
    assert "volume=-6.0dB" in fc or "volume=-6dB" in fc


def test_full_mix_duck_branch_uses_amix_normalize_zero_everywhere(
    tmp_path, captured_cmd
) -> None:
    """Duck branch has FOUR amix operations (speech submix, music
    submix, final speech+music mix, optional SFX add). Every one
    of them must ride with normalize=0 so the music doesn't get
    halved at ANY of the nodes. Regression guard — the first
    version of the #9 fix only covered the no-duck branch."""
    speech_a = _make_fake_track(tmp_path / "narr_a.wav")
    speech_b = _make_fake_track(tmp_path / "narr_b.wav")
    music_a = _make_fake_track(tmp_path / "music_a.wav")
    music_b = _make_fake_track(tmp_path / "music_b.wav")

    AudioMixer().execute({
        "operation": "full_mix",
        "tracks": [
            {"path": str(speech_a), "role": "speech"},
            {"path": str(speech_b), "role": "speech"},
            {"path": str(music_a), "role": "music"},
            {"path": str(music_b), "role": "music"},
        ],
        "ducking": {"enabled": True},
        "output_path": str(tmp_path / "out.wav"),
    })

    fc = _filter_complex(captured_cmd[0])
    # Every amix in this filter graph must carry normalize=0.
    amix_count = fc.count("amix=inputs=")
    normalize_zero_count = fc.count("normalize=0")
    assert amix_count >= 3, (
        f"expected at least 3 amix operations in the duck branch, "
        f"found {amix_count}. filter_complex: {fc}"
    )
    assert normalize_zero_count >= amix_count, (
        f"every amix must include normalize=0, but counted "
        f"{amix_count} amix operations and only {normalize_zero_count} "
        f"normalize=0 occurrences. filter_complex: {fc}"
    )


def test_full_mix_duck_branch_adds_alimiter_before_loudnorm(
    tmp_path, captured_cmd
) -> None:
    """The duck branch used to terminate in plain amix with no peak
    limiter, then run loudnorm directly on that. Without alimiter,
    summed peaks from normalize=0 can clip before loudnorm sees
    them. The limiter must appear in the graph."""
    speech = _make_fake_track(tmp_path / "narr.wav")
    music = _make_fake_track(tmp_path / "music.wav")

    AudioMixer().execute({
        "operation": "full_mix",
        "tracks": [
            {"path": str(speech), "role": "speech"},
            {"path": str(music), "role": "music"},
        ],
        "ducking": {"enabled": True},
        "output_path": str(tmp_path / "out.wav"),
    })

    fc = _filter_complex(captured_cmd[0])
    assert "alimiter" in fc, (
        f"duck branch must include alimiter. filter_complex: {fc}"
    )
    # Order: alimiter must come BEFORE loudnorm (if loudnorm is present).
    if "loudnorm" in fc:
        assert fc.index("alimiter") < fc.index("loudnorm"), (
            f"alimiter must precede loudnorm so peaks are bounded "
            f"before the loudness normalization pass. fc: {fc}"
        )


def test_full_mix_duck_branch_with_sfx_still_limits(tmp_path, captured_cmd) -> None:
    """Adding sfx routes through a second amix node before the limiter.
    The limiter must still fire at the end."""
    speech = _make_fake_track(tmp_path / "narr.wav")
    music = _make_fake_track(tmp_path / "music.wav")
    sfx = _make_fake_track(tmp_path / "sfx.wav")

    AudioMixer().execute({
        "operation": "full_mix",
        "tracks": [
            {"path": str(speech), "role": "speech"},
            {"path": str(music), "role": "music"},
            {"path": str(sfx), "role": "sfx"},
        ],
        "ducking": {"enabled": True},
        "output_path": str(tmp_path / "out.wav"),
    })

    fc = _filter_complex(captured_cmd[0])
    assert "alimiter" in fc
    # Every amix still carries normalize=0.
    assert fc.count("amix=inputs=") == fc.count("normalize=0")


def test_mix_result_reports_music_gain_db(tmp_path, captured_cmd) -> None:
    """Result envelope should name the applied gain so the compose
    director can include it in the checkpoint for audit."""
    music = _make_fake_track(tmp_path / "music.wav")

    r = AudioMixer().execute({
        "operation": "mix",
        "tracks": [{"path": str(music), "role": "music"}],
        "music_gain_db": -6,
        "output_path": str(tmp_path / "out.wav"),
    })

    assert r.success
    assert r.data.get("music_gain_db") == -6
