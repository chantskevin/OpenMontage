"""End-to-end render-matrix tests.

A small grid of (renderer_family × operation × aspect_ratio) cells
that runs `video_compose` against ffmpeg-generated testsrc sources
and asserts the rendered output matches the request:

  - Output file exists and is non-empty.
  - Resolution matches the requested aspect (autodetected when
    profile isn't pinned).
  - Duration matches the edit timeline within ±0.5s.
  - Less than 10% of frames are black per ffmpeg blackdetect.

This is the suite the user-suggested upstream improvement #5
proposed. Each cell would have caught at least one of the trio
fixes (#26 / #27 / #28 / #30) at PR time instead of in production.

Two suites:
  - render_matrix_fast: 5 ffmpeg-only cells (~2 min, runs per PR)
  - render_matrix_full: 12 cells including Remotion (nightly)
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from tools.video.video_compose import VideoCompose


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class RenderCell:
    """One matrix cell: a specific render configuration."""
    name: str
    renderer_family: str
    render_runtime: str  # remotion / ffmpeg / hyperframes
    operation: str        # render / compose
    width: int
    height: int
    cuts_seconds: list[float]  # per-cut duration; sums to expected total

    @property
    def expected_aspect(self) -> str:
        if self.width > self.height:
            return "landscape"
        if self.height > self.width:
            return "portrait"
        return "square"

    @property
    def expected_duration(self) -> float:
        return sum(self.cuts_seconds)


def _write_testsrc(path: Path, width: int, height: int, duration: float) -> Path:
    """Generate a testsrc2 video at the requested dimensions and duration.

    testsrc2 is moving (color bars + clock), so a black-frame ratio
    of 100% on the rendered output is unambiguous evidence that the
    source frames didn't make it through."""
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
        check=True, capture_output=True, timeout=60,
    )
    return path


def _build_inputs(cell: RenderCell, sources: list[Path], output: Path) -> dict:
    """Build the video_compose inputs for a cell."""
    asset_manifest = {
        "version": "1.0",
        "assets": [
            {
                "id": f"a{i}", "type": "video", "path": str(p),
                "source_tool": "test", "scene_id": f"s{i}",
            }
            for i, p in enumerate(sources)
        ],
    }
    cuts = [
        {
            "id": f"c{i}", "source": f"a{i}",
            "in_seconds": 0, "out_seconds": dur,
        }
        for i, dur in enumerate(cell.cuts_seconds)
    ]
    edit_decisions = {
        "version": "1.0",
        "render_runtime": cell.render_runtime,
        "renderer_family": cell.renderer_family,
        "cuts": cuts,
    }
    return {
        "operation": cell.operation,
        "edit_decisions": edit_decisions,
        "asset_manifest": asset_manifest,
        "output_path": str(output),
    }


def _measure_black_ratio(path: Path) -> float:
    """Run ffmpeg blackdetect and return the ratio of black frames.

    blackdetect emits lines like:
      [blackdetect @ 0x...] black_start:0 black_end:0.033 black_duration:0.033

    Sum black durations; divide by total duration. 0.0 = no black,
    1.0 = entirely black."""
    if shutil.which("ffmpeg") is None:
        return 0.0  # Can't measure; don't fail the test on missing tool.

    proc = subprocess.run(
        [
            "ffmpeg", "-i", str(path),
            "-vf", "blackdetect=d=0.05:pix_th=0.10",
            "-an", "-f", "null", "-",
        ],
        capture_output=True, text=True, timeout=60,
    )
    # ffmpeg writes blackdetect output to stderr.
    output = proc.stderr or ""

    total_black = 0.0
    for match in re.finditer(r"black_duration:(\d+\.?\d*)", output):
        total_black += float(match.group(1))

    # Get total file duration via ffprobe.
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(path),
        ],
        capture_output=True, text=True, timeout=15,
    )
    try:
        total_duration = float(probe.stdout.strip())
    except (ValueError, AttributeError):
        return 0.0

    if total_duration <= 0:
        return 0.0
    return min(1.0, total_black / total_duration)


def _assert_cell_invariants(cell: RenderCell, output: Path) -> None:
    """The four assertions every cell must satisfy."""
    # 1. Output file exists and is non-empty.
    assert output.exists(), f"{cell.name}: output file missing at {output}"
    assert output.stat().st_size > 0, f"{cell.name}: output is 0 bytes"

    # 2. ffprobe via the existing helper for ground truth.
    actual = VideoCompose._probe_output_metadata(output)
    assert actual is not None, f"{cell.name}: ffprobe failed on output"

    # 3. Aspect matches request.
    w, h = actual["width"], actual["height"]
    if cell.expected_aspect == "portrait":
        assert h > w, (
            f"{cell.name}: expected portrait, got {w}x{h} (would have "
            f"reproduced fork issue #27)"
        )
    elif cell.expected_aspect == "landscape":
        assert w > h, (
            f"{cell.name}: expected landscape, got {w}x{h}"
        )
    else:
        assert w == h, f"{cell.name}: expected square, got {w}x{h}"

    # 4. Duration matches edit timeline ±0.5s.
    expected_dur = cell.expected_duration
    actual_dur = actual["duration_seconds"]
    assert abs(actual_dur - expected_dur) <= 0.5, (
        f"{cell.name}: duration {actual_dur}s differs from expected "
        f"{expected_dur}s by more than 0.5s (would have reproduced "
        f"fork issue #26)"
    )

    # 5. Black-frame ratio < 10%.
    black_ratio = _measure_black_ratio(output)
    assert black_ratio < 0.10, (
        f"{cell.name}: {black_ratio*100:.1f}% black frames, threshold 10% "
        f"(would have reproduced fork issue #28)"
    )


# ---------------------------------------------------------------------------
# Fast suite — 5 cells, ffmpeg-only, runs per PR
# ---------------------------------------------------------------------------


FAST_CELLS = [
    RenderCell(
        name="cinematic-trailer__compose__landscape",
        renderer_family="cinematic-trailer",
        render_runtime="ffmpeg",
        operation="compose",
        width=1920, height=1080,
        cuts_seconds=[1.0, 1.0, 1.0],
    ),
    RenderCell(
        name="cinematic-trailer__compose__portrait",
        renderer_family="cinematic-trailer",
        render_runtime="ffmpeg",
        operation="compose",
        width=720, height=1280,
        cuts_seconds=[1.0, 1.0, 1.0],
    ),
    RenderCell(
        name="cinematic-trailer__compose__square",
        renderer_family="cinematic-trailer",
        render_runtime="ffmpeg",
        operation="compose",
        width=720, height=720,
        cuts_seconds=[1.0, 1.0],
    ),
    RenderCell(
        name="explainer__compose__landscape",
        renderer_family="explainer-data",
        render_runtime="ffmpeg",
        operation="compose",
        width=1920, height=1080,
        cuts_seconds=[1.5, 1.5],
    ),
    RenderCell(
        name="cinematic-trailer__render-via-ffmpeg__portrait",
        renderer_family="cinematic-trailer",
        render_runtime="ffmpeg",
        operation="render",
        width=720, height=1280,
        cuts_seconds=[1.0, 1.0, 1.0],
    ),
]


@pytest.mark.render_matrix_fast
@pytest.mark.parametrize("cell", FAST_CELLS, ids=lambda c: c.name)
def test_render_matrix_fast(cell: RenderCell, tmp_path: Path) -> None:
    """Each fast cell: generate sources, render, assert the four
    invariants. Catches the #26/#27/#28/#30 class at PR time."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")

    sources = [
        _write_testsrc(tmp_path / f"src_{i}.mp4", cell.width, cell.height, dur)
        for i, dur in enumerate(cell.cuts_seconds)
    ]
    output = tmp_path / "out.mp4"

    inputs = _build_inputs(cell, sources, output)
    result = VideoCompose().execute(inputs)
    assert result.success, f"{cell.name}: render failed: {result.error}"

    _assert_cell_invariants(cell, output)


# ---------------------------------------------------------------------------
# Full suite — adds Remotion cells (nightly)
# ---------------------------------------------------------------------------


FULL_CELLS = FAST_CELLS + [
    RenderCell(
        name="cinematic-trailer__render-via-remotion__landscape",
        renderer_family="cinematic-trailer",
        render_runtime="remotion",
        operation="render",
        width=1920, height=1080,
        cuts_seconds=[1.0, 1.0, 1.0],
    ),
    RenderCell(
        name="cinematic-trailer__render-via-remotion__portrait",
        renderer_family="cinematic-trailer",
        render_runtime="remotion",
        operation="render",
        width=720, height=1280,
        cuts_seconds=[1.0, 1.0, 1.0],
    ),
    RenderCell(
        name="cinematic-trailer__remotion-render__landscape",
        renderer_family="cinematic-trailer",
        render_runtime="remotion",
        operation="remotion_render",
        width=1920, height=1080,
        cuts_seconds=[1.0, 1.0],
    ),
    RenderCell(
        name="explainer__render-via-remotion__landscape",
        renderer_family="explainer-data",
        render_runtime="remotion",
        operation="render",
        width=1920, height=1080,
        cuts_seconds=[1.0, 1.0],
    ),
    RenderCell(
        name="documentary-montage__compose__landscape",
        renderer_family="documentary-montage",
        render_runtime="ffmpeg",
        operation="compose",
        width=1920, height=1080,
        cuts_seconds=[1.5, 1.5],
    ),
    RenderCell(
        name="product-reveal__compose__square",
        renderer_family="product-reveal",
        render_runtime="ffmpeg",
        operation="compose",
        width=1080, height=1080,
        cuts_seconds=[1.0, 1.0, 1.0],
    ),
    RenderCell(
        name="cinematic-trailer__compose__landscape__longer",
        renderer_family="cinematic-trailer",
        render_runtime="ffmpeg",
        operation="compose",
        width=1920, height=1080,
        cuts_seconds=[2.0, 3.0, 5.0],  # 10s — exact #26 repro shape
    ),
]


@pytest.mark.render_matrix_full
@pytest.mark.parametrize("cell", FULL_CELLS, ids=lambda c: c.name)
def test_render_matrix_full(cell: RenderCell, tmp_path: Path) -> None:
    """Full matrix — same invariants as fast suite, but adds Remotion
    cells and additional renderer_families. Skipped when npx isn't
    available (see conftest)."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")

    sources = [
        _write_testsrc(tmp_path / f"src_{i}.mp4", cell.width, cell.height, dur)
        for i, dur in enumerate(cell.cuts_seconds)
    ]
    output = tmp_path / "out.mp4"

    inputs = _build_inputs(cell, sources, output)
    result = VideoCompose().execute(inputs)
    assert result.success, f"{cell.name}: render failed: {result.error}"

    _assert_cell_invariants(cell, output)
