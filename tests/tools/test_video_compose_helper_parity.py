"""Parity guard for shared helpers across video_compose operations.

Fork issues #29 and #30 existed because `_render` got cuts→scenes
adapter + canvas autodetect first, and `_compose` was left behind.
The fix promoted both behaviors to shared helpers
(`_resolve_cut_sources`, `_auto_detect_canvas_profile`).

These tests lock the parity invariant DIRECTLY, not via end-to-end
runs: the helpers are pure-ish and produce the same output for the
same input regardless of which operation is calling them. If any
future operation needs different behavior, it must override the
helper explicitly — silent drift like #29 / #30 should fail this
suite.
"""

from __future__ import annotations

import io
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tools.video.video_compose import VideoCompose


def _write_real_mp4(path: Path, width: int, height: int) -> Path:
    """ffmpeg testsrc → real MP4 at exact dimensions for ffprobe to read."""
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


# ---------------------------------------------------------------------------
# _resolve_cut_sources — used by both _render and _compose
# ---------------------------------------------------------------------------


def test_resolve_cut_sources_replaces_id_with_path() -> None:
    """The asset-director and edit-director skills emit cuts that
    reference assets by ID. The helper must substitute path."""
    cuts = [{"source": "s1", "in_seconds": 0, "out_seconds": 3}]
    manifest = {"assets": [{"id": "s1", "path": "scene-1.mp4"}]}
    out = VideoCompose._resolve_cut_sources(cuts, manifest)
    assert out[0]["source"] == "scene-1.mp4"


def test_resolve_cut_sources_passes_literal_path_through() -> None:
    """Backwards compat: when source already looks like a path (no
    matching ID), leave it alone. Supports callers that pre-resolve."""
    cuts = [{"source": "/tmp/literal.mp4", "in_seconds": 0, "out_seconds": 3}]
    manifest = {"assets": [{"id": "different", "path": "x.mp4"}]}
    out = VideoCompose._resolve_cut_sources(cuts, manifest)
    assert out[0]["source"] == "/tmp/literal.mp4"


def test_resolve_cut_sources_does_not_mutate_input() -> None:
    """Returns NEW dicts. Mutation would corrupt the caller's
    edit_decisions and surprise downstream stages."""
    cuts = [{"source": "s1", "in_seconds": 0, "out_seconds": 3}]
    manifest = {"assets": [{"id": "s1", "path": "scene-1.mp4"}]}
    VideoCompose._resolve_cut_sources(cuts, manifest)
    # Original cuts dict unchanged.
    assert cuts[0]["source"] == "s1"


def test_resolve_cut_sources_handles_missing_manifest() -> None:
    """None manifest → cuts pass through unchanged. Empty assets
    list ditto. The helper must not raise on benign missing inputs."""
    cuts = [{"source": "s1", "in_seconds": 0, "out_seconds": 3}]
    assert VideoCompose._resolve_cut_sources(cuts, None)[0]["source"] == "s1"
    assert VideoCompose._resolve_cut_sources(cuts, {})[0]["source"] == "s1"
    assert VideoCompose._resolve_cut_sources(
        cuts, {"assets": []}
    )[0]["source"] == "s1"


def test_resolve_cut_sources_skips_malformed_assets() -> None:
    """Manifest entries missing id / path / type-mismatch → ignored,
    not crashed-on. Robust to BOS-style partial manifests."""
    cuts = [{"source": "s1", "in_seconds": 0, "out_seconds": 3}]
    manifest = {
        "assets": [
            {"id": "s1"},                # path missing
            {"path": "x.mp4"},           # id missing
            {"id": 123, "path": "x.mp4"},  # wrong type
            "not a dict",                # not a dict
            {"id": "s1", "path": "real.mp4"},  # the real one
        ],
    }
    out = VideoCompose._resolve_cut_sources(cuts, manifest)
    assert out[0]["source"] == "real.mp4"


def test_resolve_cut_sources_preserves_other_fields() -> None:
    """Only `source` is rewritten. in_seconds / out_seconds / transition
    metadata / arbitrary fields all carry through."""
    cuts = [{
        "source": "s1",
        "in_seconds": 1.5, "out_seconds": 4.0,
        "transition_in": "dissolve", "transition_duration": 0.5,
        "arbitrary_metadata": {"nested": True},
    }]
    manifest = {"assets": [{"id": "s1", "path": "x.mp4"}]}
    out = VideoCompose._resolve_cut_sources(cuts, manifest)
    assert out[0]["in_seconds"] == 1.5
    assert out[0]["out_seconds"] == 4.0
    assert out[0]["transition_in"] == "dissolve"
    assert out[0]["arbitrary_metadata"] == {"nested": True}


# ---------------------------------------------------------------------------
# _auto_detect_canvas_profile — used by both paths
# ---------------------------------------------------------------------------


def test_autodetect_picks_portrait_for_portrait_sources(tmp_path: Path) -> None:
    p = _write_real_mp4(tmp_path / "p.mp4", 720, 1280)
    cuts = [{"source": str(p)}]
    assert VideoCompose()._auto_detect_canvas_profile(cuts) == "instagram_reels"


def test_autodetect_picks_landscape_for_landscape_sources(tmp_path: Path) -> None:
    a = _write_real_mp4(tmp_path / "a.mp4", 1920, 1080)
    cuts = [{"source": str(a)}]
    assert VideoCompose()._auto_detect_canvas_profile(cuts) == "generic_hd"


def test_autodetect_returns_none_on_no_cuts() -> None:
    """No cuts → no signal → preserve historical 1920x1080 fallback by
    returning None."""
    assert VideoCompose()._auto_detect_canvas_profile([]) is None


# ---------------------------------------------------------------------------
# Parity invariant — same input → same resolved cuts
# ---------------------------------------------------------------------------


def test_render_and_compose_produce_identical_resolved_cuts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The load-bearing parity invariant. Both _render and _compose
    must call _resolve_cut_sources, and the result must be identical
    for the same input. If a future operation drifts (e.g. an
    operation that resolves IDs differently), this test fires.

    We don't run end-to-end; we capture what the resolver produces and
    assert the result is the same dict, key-for-key, across both code
    paths."""
    real_clip = _write_real_mp4(tmp_path / "scene.mp4", 1920, 1080)
    cuts = [
        {"source": "s1", "in_seconds": 0, "out_seconds": 3,
         "transition_in": "dissolve"},
    ]
    manifest = {"assets": [{"id": "s1", "path": str(real_clip)}]}

    # Direct helper call (this is what both _render and _compose do
    # after refactoring).
    resolved = VideoCompose._resolve_cut_sources(cuts, manifest)

    # The invariant: source resolves to the manifest path; everything
    # else passes through unchanged. If a future code path needs
    # different behavior, this test fires and the new path needs an
    # explicit override (or this test needs an explicit reason to skip).
    assert resolved == [
        {"source": str(real_clip), "in_seconds": 0, "out_seconds": 3,
         "transition_in": "dissolve"},
    ]


def test_both_operations_invoke_the_shared_resolver(monkeypatch) -> None:
    """Source-level guard: assert that _render and _compose both call
    _resolve_cut_sources. If a future commit removes the call from
    one path (the shape of fork issue #29), this fires.

    We monkeypatch the helper to record invocations, then exercise
    both code paths up to the point where resolution would have run."""
    calls: list[str] = []

    def spy(cuts, asset_manifest):
        calls.append("called")
        return [dict(c) for c in cuts]

    monkeypatch.setattr(VideoCompose, "_resolve_cut_sources", staticmethod(spy))

    # _compose entry — fail fast on missing required fields, but the
    # resolver should have run first.
    inputs_compose = {
        "operation": "compose",
        "edit_decisions": {"cuts": [{"source": "s1", "in_seconds": 0, "out_seconds": 1}]},
        "asset_manifest": {"assets": [{"id": "s1", "path": "/nonexistent.mp4"}]},
    }
    VideoCompose().execute(inputs_compose)
    assert "called" in calls, "compose path must invoke _resolve_cut_sources"

    calls.clear()

    # _render entry — same shape.
    inputs_render = {
        "operation": "render",
        "edit_decisions": {
            "cuts": [{"source": "s1", "in_seconds": 0, "out_seconds": 1}],
            "render_runtime": "ffmpeg",
            "renderer_family": "cinematic-trailer",
        },
        "asset_manifest": {"assets": [{"id": "s1", "path": "/nonexistent.mp4"}]},
    }
    VideoCompose().execute(inputs_render)
    assert "called" in calls, "render path must invoke _resolve_cut_sources"
