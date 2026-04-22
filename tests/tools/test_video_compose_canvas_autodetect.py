"""Regression tests for fork issue #8.

The observed symptom was: agent passes aspect_ratio=9:16, Veo returns
portrait 720×1280 clips, final render comes back 1920×1080 landscape.
Tracing showed the problem wasn't aspect_ratio being dropped at Veo
— the model name derivation correctly routes 9:16 to a portrait Veo
model. The real gap was that video_compose's canvas defaults to
1920×1080 regardless of source orientation, so portrait source was
scaled/letterboxed into a landscape canvas.

Fix: when no `profile` is explicitly passed, auto-detect canvas
orientation by ffprobing primary-layer cuts and picking the matching
default profile. Explicit `profile` still overrides.

These tests lock in:

  1. All-portrait cuts → instagram_reels (1080×1920) profile selected.
  2. All-landscape cuts → generic_hd (1920×1080) profile selected.
  3. Square cuts → instagram_feed (1080×1080).
  4. Mixed orientations with no majority → None (preserve historical
     1920×1080 fallback, don't coin-flip).
  5. Explicit caller profile wins over auto-detect.
  6. Remote URLs / missing files skipped (no stall).
  7. Overlay/background-layer cuts don't influence the vote — only
     primary-layer.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tools.video.video_compose import VideoCompose


def _has_ffmpeg_ffprobe() -> bool:
    import shutil
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def _make_clip(path: Path, width: int, height: int, duration: float = 1.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i",
            f"testsrc2=size={width}x{height}:rate=24:duration={duration}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(path),
        ],
        check=True, timeout=30,
    )
    return path


# ---------------------------------------------------------------------------
# _probe_video_orientation
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_ffmpeg_ffprobe(), reason="ffmpeg/ffprobe not on PATH")
def test_probe_orientation_recognizes_portrait(tmp_path: Path) -> None:
    clip = _make_clip(tmp_path / "portrait.mp4", 720, 1280)
    assert VideoCompose()._probe_video_orientation(str(clip)) == "portrait"


@pytest.mark.skipif(not _has_ffmpeg_ffprobe(), reason="ffmpeg/ffprobe not on PATH")
def test_probe_orientation_recognizes_landscape(tmp_path: Path) -> None:
    clip = _make_clip(tmp_path / "landscape.mp4", 1280, 720)
    assert VideoCompose()._probe_video_orientation(str(clip)) == "landscape"


@pytest.mark.skipif(not _has_ffmpeg_ffprobe(), reason="ffmpeg/ffprobe not on PATH")
def test_probe_orientation_recognizes_square(tmp_path: Path) -> None:
    clip = _make_clip(tmp_path / "square.mp4", 720, 720)
    assert VideoCompose()._probe_video_orientation(str(clip)) == "square"


def test_probe_orientation_returns_none_on_missing_file(tmp_path: Path) -> None:
    """Non-existent files must produce None, not crash. The auto-detect
    caller treats None as 'skip this cut' rather than halting."""
    assert VideoCompose()._probe_video_orientation(
        str(tmp_path / "does-not-exist.mp4")
    ) is None


@pytest.mark.skipif(not _has_ffmpeg_ffprobe(), reason="ffprobe not on PATH")
def test_probe_orientation_returns_none_on_zero_dimensions(tmp_path: Path) -> None:
    """Regression guard: ffprobe sometimes exits 0 with 'width,height'
    = '0,0' on a malformed file (e.g. invalid PNG signature). The
    naive `w == h` check would classify that as "square" and drag
    auto-detect into the wrong canvas. Must return None so the cut
    is ignored in the orientation vote."""
    bad = tmp_path / "not-a-real-png.png"
    bad.write_bytes(b"\x00")  # invalid PNG signature
    assert VideoCompose()._probe_video_orientation(str(bad)) is None


# ---------------------------------------------------------------------------
# _auto_detect_canvas_profile
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_ffmpeg_ffprobe(), reason="ffmpeg/ffprobe not on PATH")
def test_autodetect_all_portrait_picks_instagram_reels(tmp_path: Path) -> None:
    """The #8 scenario: Veo returned portrait clips, compose auto-
    detects portrait majority → instagram_reels canvas (1080×1920)."""
    clips = [
        _make_clip(tmp_path / f"c{i}.mp4", 720, 1280)
        for i in range(3)
    ]
    cuts = [{"id": f"c{i}", "source": str(c), "in_seconds": 0, "out_seconds": 1}
            for i, c in enumerate(clips)]
    assert VideoCompose()._auto_detect_canvas_profile(cuts) == "instagram_reels"


@pytest.mark.skipif(not _has_ffmpeg_ffprobe(), reason="ffmpeg/ffprobe not on PATH")
def test_autodetect_all_landscape_picks_generic_hd(tmp_path: Path) -> None:
    """Landscape majority → generic_hd. No behavior change from the
    historical default, but now driven by source truth, not static
    assumption."""
    clips = [
        _make_clip(tmp_path / f"c{i}.mp4", 1280, 720)
        for i in range(3)
    ]
    cuts = [{"id": f"c{i}", "source": str(c), "in_seconds": 0, "out_seconds": 1}
            for i, c in enumerate(clips)]
    assert VideoCompose()._auto_detect_canvas_profile(cuts) == "generic_hd"


@pytest.mark.skipif(not _has_ffmpeg_ffprobe(), reason="ffmpeg/ffprobe not on PATH")
def test_autodetect_all_square_picks_instagram_feed(tmp_path: Path) -> None:
    clips = [_make_clip(tmp_path / f"c{i}.mp4", 720, 720) for i in range(2)]
    cuts = [{"id": f"c{i}", "source": str(c), "in_seconds": 0, "out_seconds": 1}
            for i, c in enumerate(clips)]
    assert VideoCompose()._auto_detect_canvas_profile(cuts) == "instagram_feed"


@pytest.mark.skipif(not _has_ffmpeg_ffprobe(), reason="ffmpeg/ffprobe not on PATH")
def test_autodetect_ties_return_none(tmp_path: Path) -> None:
    """2 portrait + 2 landscape → no majority → None → caller keeps
    historical 1920×1080 fallback. Better than coin-flipping a canvas
    that would crop half the sources either way."""
    cuts = [
        {"id": "c0", "source": str(_make_clip(tmp_path / "a.mp4", 720, 1280)),
         "in_seconds": 0, "out_seconds": 1},
        {"id": "c1", "source": str(_make_clip(tmp_path / "b.mp4", 720, 1280)),
         "in_seconds": 0, "out_seconds": 1},
        {"id": "c2", "source": str(_make_clip(tmp_path / "c.mp4", 1280, 720)),
         "in_seconds": 0, "out_seconds": 1},
        {"id": "c3", "source": str(_make_clip(tmp_path / "d.mp4", 1280, 720)),
         "in_seconds": 0, "out_seconds": 1},
    ]
    assert VideoCompose()._auto_detect_canvas_profile(cuts) is None


def test_autodetect_empty_cuts_return_none() -> None:
    """No cuts → no signal → None."""
    assert VideoCompose()._auto_detect_canvas_profile([]) is None


def test_autodetect_skips_remote_urls(tmp_path: Path) -> None:
    """URLs stall ffprobe. Don't let them contribute to the vote.
    An all-URL cut list with no probeable sources returns None
    (historical fallback applies)."""
    cuts = [
        {"id": "c0", "source": "https://example.com/remote.mp4",
         "in_seconds": 0, "out_seconds": 1},
    ]
    assert VideoCompose()._auto_detect_canvas_profile(cuts) is None


@pytest.mark.skipif(not _has_ffmpeg_ffprobe(), reason="ffmpeg/ffprobe not on PATH")
def test_autodetect_only_counts_primary_layer_cuts(tmp_path: Path) -> None:
    """Background / overlay layers don't drive the canvas — they're
    subordinate. Mixed primary+overlay is NOT a mixed signal; only
    the primary vote counts."""
    primary_portrait = _make_clip(tmp_path / "p.mp4", 720, 1280)
    overlay_landscape = _make_clip(tmp_path / "o.mp4", 1280, 720)
    cuts = [
        {"id": "c0", "source": str(primary_portrait),
         "in_seconds": 0, "out_seconds": 1, "layer": "primary"},
        {"id": "o1", "source": str(overlay_landscape),
         "in_seconds": 0, "out_seconds": 1, "layer": "overlay"},
        {"id": "o2", "source": str(overlay_landscape),
         "in_seconds": 0, "out_seconds": 1, "layer": "overlay"},
        {"id": "o3", "source": str(overlay_landscape),
         "in_seconds": 0, "out_seconds": 1, "layer": "overlay"},
    ]
    # 1 portrait primary vs 3 landscape overlays → overlays ignored →
    # portrait canvas.
    assert VideoCompose()._auto_detect_canvas_profile(cuts) == "instagram_reels"


# ---------------------------------------------------------------------------
# Integration via _render — explicit profile still wins
# ---------------------------------------------------------------------------


def test_explicit_profile_overrides_autodetect(monkeypatch, tmp_path):
    """Callers that pin a canvas (via `profile`) must keep their
    choice even when sources disagree. The auto-detect is a
    no-caller-intent fallback, not a policy override."""
    from tools.base_tool import ToolResult

    # Stub _auto_detect to surface if it was called; we expect it to
    # NOT be called when a profile is explicitly passed.
    called = {"autodetect": False}

    def fake_autodetect(self, cuts):
        called["autodetect"] = True
        return "instagram_reels"

    monkeypatch.setattr(
        VideoCompose, "_auto_detect_canvas_profile", fake_autodetect
    )

    # Short-circuit everything past the profile-resolution point.
    def fake_pre(self, *a, **kw):
        # Force an early return via the hyperframes "unavailable" path —
        # simplest way to probe profile resolution without running a
        # real render.
        return None

    captured: dict = {}

    def fake_ffmpeg_render(self, *, inputs, edit_decisions, resolved_cuts,
                            output_path, profile):
        captured["profile"] = profile
        return ToolResult(success=True, data={"output": str(output_path)})

    monkeypatch.setattr(VideoCompose, "_pre_compose_validation", fake_pre)
    monkeypatch.setattr(VideoCompose, "_render_via_ffmpeg", fake_ffmpeg_render)

    clip = tmp_path / "src.mp4"
    clip.write_bytes(b"\x00")
    out = tmp_path / "out.mp4"

    VideoCompose()._render({
        "operation": "render",
        "profile": "cinematic",  # explicit — must win
        "edit_decisions": {
            "render_runtime": "ffmpeg",
            "renderer_family": "cinematic-trailer",
            "cuts": [{"id": "c1", "source": str(clip),
                      "in_seconds": 0, "out_seconds": 1}],
        },
        "asset_manifest": {"assets": [{"id": "c1", "path": str(clip),
                                       "type": "video", "source_tool": "x",
                                       "scene_id": "s1"}]},
        "output_path": str(out),
    })

    assert captured["profile"] == "cinematic", (
        f"explicit profile='cinematic' was not honored — compose "
        f"dispatched with profile={captured['profile']!r}. Auto-detect "
        f"must only fire when caller didn't specify."
    )
    assert called["autodetect"] is False, (
        "explicit profile was passed; auto-detect must not have been "
        "consulted."
    )
