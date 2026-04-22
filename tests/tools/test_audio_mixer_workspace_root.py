"""Regression tests for fork issue #16.

audio_mixer used to call `Path(track["path"]).exists()` on the raw
`path` field, even when the caller had passed `_workspace_root`. The
BOS asset manifest registers music as `{"id": "a7", "path":
"background-music.mp3"}` — a bare filename with no directory
prefix — and the compose director hands the dereferenced path to
audio_mixer unchanged. Without workspace-relative resolution, the
mixer failed with "Track not found: background-music.mp3" and the
final mp4 shipped silent.

Fix: `_resolve_workspace_paths` runs at the top of `execute` and
resolves relative paths (tracks[].path, primary_audio/secondary_audio,
input_path, video_path, music_path, output_path) against
`_workspace_root` when it's a valid directory. Absolute paths pass
through unchanged so local/test callers aren't affected.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from tools.audio.audio_mixer import AudioMixer


def _make_fake_track(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00")
    return path


@pytest.fixture
def captured_cmd(monkeypatch):
    """Stub run_command to capture assembled ffmpeg commands."""
    calls: list[list[str]] = []

    def fake_run_command(self, cmd, *, timeout=None, cwd=None):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="30.0", stderr="")

    monkeypatch.setattr(AudioMixer, "run_command", fake_run_command)
    return calls


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_schema_declares_workspace_root() -> None:
    """audio_mixer must accept `_workspace_root` so the media-worker
    HTTP route's job-id injection reaches it — same contract as
    video_compose after fix #13."""
    schema = AudioMixer.input_schema
    prop = schema["properties"].get("_workspace_root")
    assert prop is not None, (
        "audio_mixer.input_schema must declare `_workspace_root` so "
        "relative track paths can be resolved against the real "
        "workspace directory (issue #16)."
    )
    assert prop["type"] == "string"


# ---------------------------------------------------------------------------
# The failure mode from issue #16
# ---------------------------------------------------------------------------


def test_mix_with_bare_filename_and_workspace_root_resolves(
    tmp_path, captured_cmd
) -> None:
    """The issue-#16 repro: asset manifest registered music as a bare
    filename (`background-music.mp3`), edit decisions dereferenced it,
    and the director handed the bare name to audio_mixer along with
    `_workspace_root`. It must resolve against the workspace root."""
    music_name = "background-music.mp3"
    music_abs = _make_fake_track(tmp_path / music_name)
    voice_name = "narration.mp3"
    _make_fake_track(tmp_path / voice_name)

    result = AudioMixer().execute({
        "operation": "mix",
        "tracks": [
            {"path": music_name, "role": "music"},
            {"path": voice_name, "role": "primary"},
        ],
        "output_path": "mixed.wav",
        "_workspace_root": str(tmp_path),
    })

    assert result.success, (
        f"mix with bare filenames + _workspace_root must succeed; got "
        f"error: {result.error}"
    )
    assert captured_cmd, "ffmpeg was not invoked"
    # The input file after -i should be the absolute resolved path.
    cmd = captured_cmd[0]
    inputs_in_cmd = [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "-i"]
    assert str(music_abs) in inputs_in_cmd, (
        f"workspace_root resolution did not rewrite '{music_name}' to "
        f"{music_abs}. inputs passed to ffmpeg: {inputs_in_cmd}"
    )


def test_mix_without_workspace_root_still_fails_when_path_missing(
    tmp_path, captured_cmd, monkeypatch
) -> None:
    """Without _workspace_root the old cwd-relative behavior holds:
    a bare filename that doesn't exist under cwd returns a clear
    "Track not found" error. This keeps local/test callers working."""
    monkeypatch.chdir(tmp_path)

    result = AudioMixer().execute({
        "operation": "mix",
        "tracks": [{"path": "nope.mp3", "role": "music"}],
        "output_path": "mixed.wav",
    })

    assert not result.success
    assert "Track not found" in (result.error or "")
    assert "nope.mp3" in (result.error or "")


def test_mix_absolute_path_is_not_rewritten(tmp_path, captured_cmd) -> None:
    """When the caller has already absolutized a path, workspace-root
    resolution must leave it alone."""
    music_abs = _make_fake_track(tmp_path / "subdir" / "music.mp3")

    # Make a second workspace root that does NOT contain the absolute
    # path, to prove the absolute path wins.
    other_root = tmp_path / "other_workspace"
    other_root.mkdir()

    result = AudioMixer().execute({
        "operation": "mix",
        "tracks": [{"path": str(music_abs), "role": "music"}],
        "output_path": str(tmp_path / "out.wav"),
        "_workspace_root": str(other_root),
    })

    assert result.success, result.error
    cmd = captured_cmd[0]
    inputs_in_cmd = [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "-i"]
    assert str(music_abs) in inputs_in_cmd


def test_full_mix_ducking_branch_resolves_relative_tracks(
    tmp_path, captured_cmd
) -> None:
    """The full_mix ducking branch is the one BOS hits on cinematic
    runs. It must also resolve relative track paths against
    workspace_root."""
    music_abs = _make_fake_track(tmp_path / "bg.mp3")
    voice_abs = _make_fake_track(tmp_path / "voice.mp3")

    result = AudioMixer().execute({
        "operation": "full_mix",
        "tracks": [
            {"path": "voice.mp3", "role": "speech"},
            {"path": "bg.mp3", "role": "music"},
        ],
        "ducking": {"enabled": True},
        "output_path": "final_mix.wav",
        "_workspace_root": str(tmp_path),
    })

    assert result.success, result.error
    cmd = captured_cmd[0]
    inputs_in_cmd = [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "-i"]
    assert str(music_abs) in inputs_in_cmd
    assert str(voice_abs) in inputs_in_cmd
    # Output also resolves under the workspace.
    assert str((tmp_path / "final_mix.wav").resolve()) in cmd


def test_duck_simple_format_resolves_primary_and_secondary(
    tmp_path, captured_cmd
) -> None:
    """The simple duck API (`primary_audio`/`secondary_audio`) must
    also respect workspace_root."""
    speech_abs = _make_fake_track(tmp_path / "speech.mp3")
    music_abs = _make_fake_track(tmp_path / "music.mp3")

    result = AudioMixer().execute({
        "operation": "duck",
        "primary_audio": "speech.mp3",
        "secondary_audio": "music.mp3",
        "output_path": "ducked.wav",
        "_workspace_root": str(tmp_path),
    })

    assert result.success, result.error
    cmd = captured_cmd[0]
    inputs_in_cmd = [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "-i"]
    assert str(speech_abs) in inputs_in_cmd
    assert str(music_abs) in inputs_in_cmd


def test_segmented_music_resolves_video_and_music(
    tmp_path, captured_cmd
) -> None:
    """segmented_music hits different fields (video_path, music_path)
    — the resolver must handle them too so the same BOS-style bare
    filenames work.

    The stubbed ffmpeg doesn't actually produce an output file so
    `_segmented_music`'s existence check fails downstream; that's
    fine — we only care that the ffmpeg command was built with the
    resolved paths (the thing the fix changes)."""
    video_abs = _make_fake_track(tmp_path / "assembled.mp4")
    music_abs = _make_fake_track(tmp_path / "bg.mp3")

    AudioMixer().execute({
        "operation": "segmented_music",
        "video_path": "assembled.mp4",
        "music_path": "bg.mp3",
        "segments": [{"start": 0, "end": 5}],
        "output_path": "final.mp4",
        "_workspace_root": str(tmp_path),
    })

    assert len(captured_cmd) >= 2, (
        "segmented_music should invoke ffprobe then ffmpeg; got "
        f"{len(captured_cmd)} call(s)"
    )
    ffmpeg_cmd = captured_cmd[-1]
    inputs_in_cmd = [
        ffmpeg_cmd[i + 1] for i, arg in enumerate(ffmpeg_cmd) if arg == "-i"
    ]
    assert str(video_abs) in inputs_in_cmd
    assert str(music_abs) in inputs_in_cmd


def test_extract_resolves_input_path(tmp_path, captured_cmd) -> None:
    """extract's input_path is the one relative path it uses — the
    resolver must cover it."""
    video_abs = _make_fake_track(tmp_path / "clip.mp4")

    result = AudioMixer().execute({
        "operation": "extract",
        "input_path": "clip.mp4",
        "output_path": "audio.wav",
        "_workspace_root": str(tmp_path),
    })

    assert result.success, result.error
    cmd = captured_cmd[0]
    inputs_in_cmd = [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "-i"]
    assert str(video_abs) in inputs_in_cmd


def test_invalid_workspace_root_falls_back_to_cwd_behavior(
    tmp_path, captured_cmd, monkeypatch
) -> None:
    """If `_workspace_root` points at a non-existent directory, treat
    it as unset rather than blowing up. The caller's cwd-relative
    behavior continues — the legacy error message surfaces instead of
    a silent misroute into a phantom workspace."""
    monkeypatch.chdir(tmp_path)

    result = AudioMixer().execute({
        "operation": "mix",
        "tracks": [{"path": "nope.mp3", "role": "music"}],
        "output_path": "out.wav",
        "_workspace_root": "/path/that/definitely/does/not/exist/abc123",
    })

    # bare "nope.mp3" still fails clearly because cwd doesn't have it.
    assert not result.success
    assert "nope.mp3" in (result.error or "")
