"""Regression tests for fork issues #29 and #30 — the FFmpeg `compose`
operation parity with the `render` path.

#29: cuts[].source is treated as a literal file path; ID-shaped
     references from asset_manifest are not resolved. Skill-conformant
     edit_decisions fail with "Cut source not found: <id>".

#30: target resolution defaults to 1920x1080 regardless of source
     orientation. Portrait sources get squeezed into landscape.

The render / remotion_render paths got both fixes via the cuts→scenes
adapter (#26-#28) and the canvas autodetect (#27 / #8). These tests
lock in the analogous fixes on the compose path.
"""

from __future__ import annotations

import io
import struct
from pathlib import Path
from typing import Any

import pytest

from tools.video.video_compose import VideoCompose


# ---------------------------------------------------------------------------
# Helpers — write minimal real video files so ffprobe (used by canvas
# autodetect) can read width/height. We don't actually run ffmpeg in
# these tests; we monkeypatch run_command and stop before encoding.
# ---------------------------------------------------------------------------


def _write_real_mp4(path: Path, width: int, height: int) -> Path:
    """Use ffmpeg testsrc to write a real 1-frame MP4 at the requested
    dimensions. Needed because the canvas-autodetect path actually
    ffprobes the file; a stub byte file would fail to parse and report
    None orientation."""
    import subprocess
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"color=size={width}x{height}:duration=0.1:rate=1:color=blue",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-t", "0.1",
            str(path),
        ],
        check=True, capture_output=True, timeout=30,
    )
    return path


def _stub_run_command(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Capture ffmpeg invocations and short-circuit them so the
    encoder doesn't actually run. Returns the captured cmds."""
    import subprocess
    captured: dict[str, Any] = {"cmds": []}
    real_run = VideoCompose.run_command

    def fake_run(self, cmd, *, timeout=None, cwd=None):
        captured["cmds"].append(list(cmd))
        # Pretend success — write any -t output as an empty mp4 so
        # downstream existence checks pass.
        for i, arg in enumerate(cmd):
            if arg == "-y" and i + 1 < len(cmd):
                # Last positional arg is the output file in our usage.
                out = Path(cmd[-1])
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(b"\x00")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(VideoCompose, "run_command", fake_run)
    return captured


# ---------------------------------------------------------------------------
# 1. #29 — compose resolves cuts[].source against asset_manifest IDs
# ---------------------------------------------------------------------------


def test_compose_resolves_id_shaped_source_against_asset_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exact #29 repro: cuts reference assets by ID, asset_manifest
    maps each ID to a real path. compose used to fail with 'Cut source
    not found: <id>'. After the fix, IDs resolve and compose proceeds."""
    _stub_run_command(monkeypatch)
    real_clip = _write_real_mp4(tmp_path / "scene.mp4", 1920, 1080)

    inputs = {
        "operation": "compose",
        "edit_decisions": {
            "cuts": [
                {"source": "s1_hook_video", "in_seconds": 0, "out_seconds": 3},
            ],
        },
        "asset_manifest": {
            "assets": [
                {"id": "s1_hook_video", "path": str(real_clip)},
            ],
        },
        "output_path": str(tmp_path / "out.mp4"),
    }
    result = VideoCompose().execute(inputs)
    assert result.success, (
        f"compose must resolve ID-shaped source against asset_manifest. "
        f"Got error: {result.error}"
    )


def test_compose_literal_path_still_works_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Backwards compat: literal file paths in cuts[].source must
    continue to work. Resolution is best-effort — when the source
    doesn't match an ID, fall through to literal-path treatment."""
    _stub_run_command(monkeypatch)
    real_clip = _write_real_mp4(tmp_path / "scene.mp4", 1920, 1080)

    inputs = {
        "operation": "compose",
        "edit_decisions": {
            "cuts": [
                {"source": str(real_clip), "in_seconds": 0, "out_seconds": 3},
            ],
        },
        "output_path": str(tmp_path / "out.mp4"),
    }
    result = VideoCompose().execute(inputs)
    assert result.success, result.error


def test_compose_unresolved_id_surfaces_helpful_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the source neither exists as a path NOR matches an ID,
    the error should name both shapes so the caller can tell whether
    the lookup ran. Old error was just `Cut source not found: <id>`
    with no signal that an ID lookup was attempted."""
    _stub_run_command(monkeypatch)

    inputs = {
        "operation": "compose",
        "edit_decisions": {
            "cuts": [
                {"source": "phantom_id", "in_seconds": 0, "out_seconds": 3},
            ],
        },
        "asset_manifest": {"assets": [{"id": "different_id", "path": "x.mp4"}]},
        "output_path": str(tmp_path / "out.mp4"),
    }
    result = VideoCompose().execute(inputs)
    assert not result.success
    err = result.error or ""
    assert "phantom_id" in err
    assert "id→path" in err or "asset_manifest" in err


# ---------------------------------------------------------------------------
# 2. #30 — compose autodetects canvas orientation from source cuts
# ---------------------------------------------------------------------------


def test_compose_portrait_sources_pick_portrait_canvas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exact #30 repro: three 720×1280 portrait sources, no
    explicit profile. compose used to default to 1920×1080 landscape,
    silently squeezing portrait into landscape. After the fix, the
    canvas autodetect picks instagram_reels (1080×1920)."""
    captured = _stub_run_command(monkeypatch)
    p1 = _write_real_mp4(tmp_path / "s1.mp4", 720, 1280)
    p2 = _write_real_mp4(tmp_path / "s2.mp4", 720, 1280)
    p3 = _write_real_mp4(tmp_path / "s3.mp4", 720, 1280)

    inputs = {
        "operation": "compose",
        "edit_decisions": {
            "cuts": [
                {"source": str(p1), "in_seconds": 0, "out_seconds": 2},
                {"source": str(p2), "in_seconds": 0, "out_seconds": 3},
                {"source": str(p3), "in_seconds": 0, "out_seconds": 3},
            ],
        },
        "output_path": str(tmp_path / "out.mp4"),
    }
    result = VideoCompose().execute(inputs)
    assert result.success, result.error

    # Find an ffmpeg encoder cmd that names the resolution. The compose
    # path passes -s WIDTHxHEIGHT to size segments. With portrait
    # autodetect, that should be 1080x1920 instead of 1920x1080.
    saw_portrait = False
    saw_landscape = False
    for cmd in captured["cmds"]:
        for arg in cmd:
            if isinstance(arg, str):
                if "1080x1920" in arg or "1080:1920" in arg:
                    saw_portrait = True
                if "1920x1080" in arg and "1920x1080:" not in arg:
                    saw_landscape = True
    assert saw_portrait, (
        f"portrait sources must yield a 1080x1920 canvas (instagram_reels "
        f"profile). cmds: {captured['cmds']}"
    )


def test_compose_landscape_sources_pick_landscape_canvas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sanity: landscape sources still pick 1920x1080 (generic_hd).
    The autodetect must not regress the existing default for the
    common landscape case."""
    captured = _stub_run_command(monkeypatch)
    a = _write_real_mp4(tmp_path / "a.mp4", 1920, 1080)

    inputs = {
        "operation": "compose",
        "edit_decisions": {
            "cuts": [
                {"source": str(a), "in_seconds": 0, "out_seconds": 3},
            ],
        },
        "output_path": str(tmp_path / "out.mp4"),
    }
    result = VideoCompose().execute(inputs)
    assert result.success, result.error

    saw_landscape = any(
        any(isinstance(arg, str) and "1920x1080" in arg for arg in cmd)
        for cmd in captured["cmds"]
    )
    assert saw_landscape, (
        f"landscape sources must yield 1920x1080 (generic_hd default). "
        f"cmds: {captured['cmds']}"
    )


def test_compose_explicit_profile_overrides_autodetect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the caller pins a profile, the autodetect must NOT fire.
    The orientation guess is a fallback for unspecified intent, not a
    policy override."""
    captured = _stub_run_command(monkeypatch)
    # Portrait source — autodetect would pick instagram_reels.
    p = _write_real_mp4(tmp_path / "p.mp4", 720, 1280)

    inputs = {
        "operation": "compose",
        # Explicit landscape profile overrides the portrait autodetect.
        "profile": "generic_hd",
        "edit_decisions": {
            "cuts": [
                {"source": str(p), "in_seconds": 0, "out_seconds": 3},
            ],
        },
        "output_path": str(tmp_path / "out.mp4"),
    }
    result = VideoCompose().execute(inputs)
    assert result.success, result.error

    # Should land on 1920x1080 (the explicit profile), not 1080x1920.
    saw_landscape = any(
        any(isinstance(arg, str) and "1920x1080" in arg for arg in cmd)
        for cmd in captured["cmds"]
    )
    assert saw_landscape


# ---------------------------------------------------------------------------
# 3. Combined: #29 + #30 — ID resolution AND canvas autodetect together
# ---------------------------------------------------------------------------


def test_compose_id_referenced_portrait_cuts_get_portrait_canvas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Realistic skill-conformant repro: cuts reference IDs, sources
    are portrait. Without #29 the lookup fails before autodetect runs.
    Without #30 the autodetect falls back to landscape. Both fixes
    together produce the right shape: portrait canvas for portrait
    sources referenced by ID."""
    captured = _stub_run_command(monkeypatch)
    p1 = _write_real_mp4(tmp_path / "scene1.mp4", 720, 1280)
    p2 = _write_real_mp4(tmp_path / "scene2.mp4", 720, 1280)

    inputs = {
        "operation": "compose",
        "edit_decisions": {
            "cuts": [
                {"source": "s1_video", "in_seconds": 0, "out_seconds": 3},
                {"source": "s2_video", "in_seconds": 0, "out_seconds": 3},
            ],
        },
        "asset_manifest": {
            "assets": [
                {"id": "s1_video", "path": str(p1)},
                {"id": "s2_video", "path": str(p2)},
            ],
        },
        "output_path": str(tmp_path / "out.mp4"),
    }
    result = VideoCompose().execute(inputs)
    assert result.success, result.error

    saw_portrait = any(
        any(isinstance(arg, str) and "1080x1920" in arg for arg in cmd)
        for cmd in captured["cmds"]
    )
    assert saw_portrait
