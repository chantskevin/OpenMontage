"""Ground-truth probe tests for video_compose.

The class of bugs #26 / #27 / #28 / #30 all shipped because the
director's render_report claimed something the actual output file
didn't agree with. Closing this class structurally requires the
tool surface ground-truth probe data so downstream consumers can
use it instead of fabricating fields.

`_probe_output_metadata` is the helper. Both `_compose` and
`_remotion_render` success returns now include
`data["actual_output"] = {resolution, duration_seconds, video_codec,
audio_codec, fps, file_size_bytes, has_audio, ...}` derived from
ffprobe.

These tests lock the probe semantics directly. End-to-end tests
that the ffprobe output makes it into ToolResult.data live in the
existing compose-path / cinematic-trio suites; this file focuses on
the helper itself.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tools.video.video_compose import VideoCompose


def _write_real_mp4(
    path: Path, width: int, height: int, duration: float = 0.1, fps: int = 30
) -> Path:
    """ffmpeg testsrc → real MP4 with the requested dims/duration."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i",
            f"color=size={width}x{height}:duration={duration}:rate={fps}:color=blue",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-t", str(duration),
            str(path),
        ],
        check=True, capture_output=True, timeout=30,
    )
    return path


def _write_mp4_with_audio(
    path: Path, width: int, height: int, duration: float = 0.5
) -> Path:
    """MP4 with a silent audio track, for has_audio assertions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"color=size={width}x{height}:duration={duration}:rate=30:color=blue",
            "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-t", str(duration), "-shortest",
            str(path),
        ],
        check=True, capture_output=True, timeout=30,
    )
    return path


# ---------------------------------------------------------------------------
# 1. Resolution / dimensions probe — closes #27 / #30 class
# ---------------------------------------------------------------------------


def test_probe_returns_actual_resolution_for_landscape(tmp_path: Path) -> None:
    """The simplest #27 invariant: probe of a 1920x1080 file returns
    that exact resolution. Director claims notwithstanding."""
    f = _write_real_mp4(tmp_path / "landscape.mp4", 1920, 1080)
    actual = VideoCompose._probe_output_metadata(f)
    assert actual is not None
    assert actual["resolution"] == "1920x1080"
    assert actual["width"] == 1920
    assert actual["height"] == 1080


def test_probe_returns_actual_resolution_for_portrait(tmp_path: Path) -> None:
    """The #27 case: portrait file probed returns 720x1280, not the
    director's mistaken 1920x1080 claim."""
    f = _write_real_mp4(tmp_path / "portrait.mp4", 720, 1280)
    actual = VideoCompose._probe_output_metadata(f)
    assert actual is not None
    assert actual["resolution"] == "720x1280"
    assert actual["width"] == 720
    assert actual["height"] == 1280


# ---------------------------------------------------------------------------
# 2. Duration probe — closes #26 class
# ---------------------------------------------------------------------------


def test_probe_returns_actual_duration(tmp_path: Path) -> None:
    """The #26 repro: director claims duration_seconds=10 but the
    file is 30s. Probe returns the real file duration regardless of
    what the director plans."""
    f = _write_real_mp4(tmp_path / "two_sec.mp4", 640, 360, duration=2.0)
    actual = VideoCompose._probe_output_metadata(f)
    assert actual is not None
    # Allow tiny ffmpeg encoding rounding (within 0.2s).
    assert abs(actual["duration_seconds"] - 2.0) < 0.2


# ---------------------------------------------------------------------------
# 3. Audio stream presence — closes #28 class (silent renders)
# ---------------------------------------------------------------------------


def test_probe_detects_audio_stream_present(tmp_path: Path) -> None:
    """When the render has audio, has_audio: True. Used downstream by
    audio_spotcheck / final_review to verify narration/music landed."""
    f = _write_mp4_with_audio(tmp_path / "with_audio.mp4", 640, 360)
    actual = VideoCompose._probe_output_metadata(f)
    assert actual is not None
    assert actual["has_audio"] is True
    assert actual["audio_codec"] == "aac"


def test_probe_detects_audio_stream_absent(tmp_path: Path) -> None:
    """When the render has no audio (video-only), has_audio: False.
    Catches the silent-render shape from #28 — render appears successful
    but there's no audio stream at all."""
    f = _write_real_mp4(tmp_path / "no_audio.mp4", 640, 360)
    actual = VideoCompose._probe_output_metadata(f)
    assert actual is not None
    assert actual["has_audio"] is False


# ---------------------------------------------------------------------------
# 4. Codec + fps + file size — extra signal for downstream consumers
# ---------------------------------------------------------------------------


def test_probe_returns_codec_and_fps_and_size(tmp_path: Path) -> None:
    f = _write_real_mp4(tmp_path / "clip.mp4", 640, 360, duration=1.0, fps=30)
    actual = VideoCompose._probe_output_metadata(f)
    assert actual is not None
    assert actual["video_codec"] == "h264"
    assert actual["fps"] == 30.0
    assert actual["file_size_bytes"] > 0


# ---------------------------------------------------------------------------
# 5. Failure modes — return None, don't crash
# ---------------------------------------------------------------------------


def test_probe_returns_none_on_missing_file(tmp_path: Path) -> None:
    """Missing file → None. Caller should treat None as a red flag
    (a 'successful' render with no probable output is suspect)."""
    actual = VideoCompose._probe_output_metadata(tmp_path / "nonexistent.mp4")
    assert actual is None


def test_probe_returns_none_on_unparseable_file(tmp_path: Path) -> None:
    """Garbage file content → None, not crash. Robust to malformed
    outputs from a broken encoder."""
    f = tmp_path / "garbage.mp4"
    f.write_bytes(b"this is not a valid mp4 at all")
    actual = VideoCompose._probe_output_metadata(f)
    # ffprobe exits non-zero on this; helper returns None.
    assert actual is None


# ---------------------------------------------------------------------------
# 6. Integration: actual_output appears in ToolResult.data
# ---------------------------------------------------------------------------


def test_compose_includes_actual_output_in_result_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The end-to-end invariant: when _compose succeeds, the result's
    data dict has actual_output populated from a probe of the real
    output file. Downstream consumers (compose-director skill, BOS
    render_report writer) should read actual_output instead of
    fabricating values."""

    # Stub ffmpeg run so the encoder doesn't actually run, but write a
    # real mp4 to the output_path so the probe has something to read.
    real_clip = _write_real_mp4(tmp_path / "src.mp4", 1920, 1080, duration=1.0)
    output_path = tmp_path / "out.mp4"

    def fake_run(self, cmd, *, timeout=None, cwd=None):
        # Last positional arg of compose's final ffmpeg call is the output
        # path. Copy our prepared real_clip to it so the probe succeeds.
        import shutil
        shutil.copy(real_clip, output_path)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(VideoCompose, "run_command", fake_run)

    inputs = {
        "operation": "compose",
        "edit_decisions": {
            "cuts": [
                {"source": str(real_clip), "in_seconds": 0, "out_seconds": 1},
            ],
        },
        "output_path": str(output_path),
    }
    result = VideoCompose().execute(inputs)
    assert result.success, result.error

    actual = result.data.get("actual_output")
    assert actual is not None, (
        "compose result must include actual_output from ffprobe so "
        "downstream consumers don't have to fabricate render_report "
        "fields. data keys: " + str(list(result.data.keys()))
    )
    assert actual["resolution"] == "1920x1080"
    assert actual["video_codec"] == "h264"
