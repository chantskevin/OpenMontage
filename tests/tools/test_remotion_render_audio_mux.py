"""Regression tests for fork issue #32.

`_remotion_render` ignored `audio_path` — every cinematic-trailer
render shipped with a silent AAC track regardless of what the
caller passed. The compose path muxes audio natively; the Remotion
path needed a parallel post-render mux.

Three identical Remotion renders called with progressively more
explicit audio inputs produced byte-for-byte identical mp4s
(MD5-confirmed in the issue), every one at -91 dB mean volume.

Fix: post-process Remotion's video-only output with an ffmpeg mux
when audio_path is provided. No-op when not.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from tools.video.video_compose import VideoCompose


def _write_testsrc_silent(path: Path, w: int, h: int, dur: float = 1.0) -> Path:
    """Video-only testsrc — no audio stream. Stands in for a Veo
    clip whose audio is silent / absent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"testsrc2=size={w}x{h}:duration={dur}:rate=30",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-t", str(dur), str(path),
        ],
        check=True, capture_output=True, timeout=60,
    )
    return path


def _write_audible_audio(path: Path, dur: float = 1.0, freq: int = 440) -> Path:
    """A clearly-audible tone — used as the premixed audio_path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={dur}",
            "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "2",
            str(path),
        ],
        check=True, capture_output=True, timeout=30,
    )
    return path


def _measure_mean_volume_db(path: Path) -> float:
    """Run ffmpeg volumedetect and return mean_volume in dB.
    Returns -100.0 if no audio stream / can't measure."""
    import re
    proc = subprocess.run(
        [
            "ffmpeg", "-i", str(path),
            "-af", "volumedetect", "-f", "null", "-",
        ],
        capture_output=True, text=True, timeout=30,
    )
    match = re.search(r"mean_volume:\s*(-?\d+\.?\d*)\s*dB", proc.stderr or "")
    if match:
        return float(match.group(1))
    return -100.0


# ---------------------------------------------------------------------------
# 1. Direct repro — Remotion render with audio_path produces audible output
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    shutil.which("npx") is None, reason="Remotion render requires npx (Node.js)"
)
def test_remotion_render_muxes_audio_path(tmp_path: Path) -> None:
    """The exact #32 repro: render a video-only Remotion composition
    with an audio_path. Without the fix, the output's mean volume
    would be ≈ -91 dB (silent AAC). With the mux pass, it's the
    audio_path's level."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")

    src = _write_testsrc_silent(tmp_path / "scene.mp4", 1280, 720, dur=1.0)
    audio = _write_audible_audio(tmp_path / "music.wav", dur=1.0, freq=440)
    output = tmp_path / "out.mp4"

    inputs = {
        "operation": "render",
        "edit_decisions": {
            "render_runtime": "remotion",
            "renderer_family": "cinematic-trailer",
            "cuts": [{"id": "c1", "source": "a0",
                      "in_seconds": 0, "out_seconds": 1.0}],
        },
        "asset_manifest": {
            "version": "1.0",
            "assets": [{
                "id": "a0", "type": "video", "path": str(src),
                "source_tool": "test", "scene_id": "s1",
            }],
        },
        "audio_path": str(audio),
        "output_path": str(output),
    }
    result = VideoCompose().execute(inputs)
    assert result.success, f"render failed: {result.error}"

    mean_db = _measure_mean_volume_db(output)
    assert mean_db > -50, (
        f"Remotion output has silent audio ({mean_db} dB) despite "
        f"audio_path being provided. fork issue #32 — _remotion_render "
        f"ignored the audio mux step."
    )


# ---------------------------------------------------------------------------
# 2. The result data marks audio_muxed=True so callers can verify
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    shutil.which("npx") is None, reason="Remotion render requires npx (Node.js)"
)
def test_remotion_render_reports_audio_muxed_in_data(tmp_path: Path) -> None:
    """The result data should mark audio_muxed=True when the mux
    succeeded, so downstream consumers (compose-director self-check,
    final_review) can verify the mux actually ran rather than
    re-probing."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")

    src = _write_testsrc_silent(tmp_path / "scene.mp4", 1280, 720, dur=1.0)
    audio = _write_audible_audio(tmp_path / "music.wav", dur=1.0)
    output = tmp_path / "out.mp4"

    inputs = {
        "operation": "render",
        "edit_decisions": {
            "render_runtime": "remotion",
            "renderer_family": "cinematic-trailer",
            "cuts": [{"id": "c1", "source": "a0",
                      "in_seconds": 0, "out_seconds": 1.0}],
        },
        "asset_manifest": {"version": "1.0", "assets": [{
            "id": "a0", "type": "video", "path": str(src),
            "source_tool": "test", "scene_id": "s1",
        }]},
        "audio_path": str(audio),
        "output_path": str(output),
    }
    result = VideoCompose().execute(inputs)
    assert result.success
    assert result.data.get("audio_muxed") is True


# ---------------------------------------------------------------------------
# 3. Backwards compat — no audio_path → no mux pass, no extra cost
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    shutil.which("npx") is None, reason="Remotion render requires npx (Node.js)"
)
def test_remotion_render_without_audio_path_unchanged(tmp_path: Path) -> None:
    """When audio_path isn't provided, behavior is unchanged — no
    mux runs, audio_muxed=False. The fix is additive."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")

    src = _write_testsrc_silent(tmp_path / "scene.mp4", 1280, 720, dur=1.0)
    output = tmp_path / "out.mp4"

    inputs = {
        "operation": "render",
        "edit_decisions": {
            "render_runtime": "remotion",
            "renderer_family": "cinematic-trailer",
            "cuts": [{"id": "c1", "source": "a0",
                      "in_seconds": 0, "out_seconds": 1.0}],
        },
        "asset_manifest": {"version": "1.0", "assets": [{
            "id": "a0", "type": "video", "path": str(src),
            "source_tool": "test", "scene_id": "s1",
        }]},
        "output_path": str(output),
    }
    result = VideoCompose().execute(inputs)
    assert result.success
    assert result.data.get("audio_muxed") is False


# ---------------------------------------------------------------------------
# 4. Path resolution — audio_path bare filename resolves against cwd
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    shutil.which("npx") is None, reason="Remotion render requires npx (Node.js)"
)
def test_remotion_render_audio_path_bare_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bare-filename audio_path resolves against cwd, mirroring the
    #31 fix for cut sources. Same workspace-cwd convention."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")

    monkeypatch.chdir(tmp_path)
    src = _write_testsrc_silent(tmp_path / "scene.mp4", 1280, 720)
    _write_audible_audio(tmp_path / "music.wav", dur=1.0)
    output = tmp_path / "out.mp4"

    inputs = {
        "operation": "render",
        "edit_decisions": {
            "render_runtime": "remotion",
            "renderer_family": "cinematic-trailer",
            "cuts": [{"id": "c1", "source": "a0",
                      "in_seconds": 0, "out_seconds": 1.0}],
        },
        "asset_manifest": {"version": "1.0", "assets": [{
            "id": "a0", "type": "video", "path": str(src),
            "source_tool": "test", "scene_id": "s1",
        }]},
        "audio_path": "music.wav",  # bare filename
        "output_path": str(output),
    }
    result = VideoCompose().execute(inputs)
    assert result.success
    assert result.data.get("audio_muxed") is True
    assert _measure_mean_volume_db(output) > -50
