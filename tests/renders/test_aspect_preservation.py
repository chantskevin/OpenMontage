"""Aspect-ratio preservation tests — standard ratios (9:16, 16:9, 1:1).

The render-matrix asserts orientation only (height > width means
portrait). That's a coarse check — it would pass even if the output
landed at a DIFFERENT portrait ratio than the source (e.g. 720x1280
in → 1080x2160 out is still "portrait" but 9:16 became 1:2).

This suite asserts the EXACT aspect ratio round-trips through every
render path. If a source is 9:16 (1.778... inverted = 0.5625), the
output's width/height ratio must equal that within tolerance.

Tests only standard ratios (9:16, 16:9, 1:1) — these route to a
matching named profile via _auto_detect_canvas_profile so should
preserve exactly. Non-standard ratios (2:3, 4:5, 21:9) currently
snap to the nearest standard profile and don't round-trip; that's a
separate test (and a known limitation worth filing).
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from tools.video.video_compose import VideoCompose


# Tolerance: output must match source aspect within 1% (allows for
# integer rounding when canvas dims get rescaled, e.g. 720→1080).
ASPECT_TOLERANCE = 0.01


@dataclass
class AspectCell:
    """One aspect-preservation test: source dims + render path."""
    name: str
    width: int
    height: int
    operation: str        # compose / render
    render_runtime: str   # ffmpeg / remotion

    @property
    def source_aspect(self) -> float:
        return self.width / self.height


def _write_testsrc(path: Path, width: int, height: int, duration: float = 1.0) -> Path:
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


def _build_inputs(cell: AspectCell, source: Path, output: Path) -> dict:
    cuts = [{"id": "c1", "source": "a0", "in_seconds": 0, "out_seconds": 1.0}]
    if cell.operation == "remotion_render":
        cuts[0]["source"] = str(source)

    edit_decisions = {
        "version": "1.0",
        "render_runtime": cell.render_runtime,
        "renderer_family": "cinematic-trailer",
        "cuts": cuts,
    }
    inputs = {
        "operation": cell.operation,
        "edit_decisions": edit_decisions,
        "output_path": str(output),
    }
    if cell.operation != "remotion_render":
        inputs["asset_manifest"] = {
            "version": "1.0",
            "assets": [{
                "id": "a0", "type": "video", "path": str(source),
                "source_tool": "test", "scene_id": "s1",
            }],
        }
    return inputs


def _assert_aspect_preserved(cell: AspectCell, output: Path) -> None:
    actual = VideoCompose._probe_output_metadata(output)
    assert actual is not None, f"{cell.name}: ffprobe failed"

    out_w, out_h = actual["width"], actual["height"]
    out_aspect = out_w / out_h
    drift = abs(out_aspect - cell.source_aspect) / cell.source_aspect

    assert drift <= ASPECT_TOLERANCE, (
        f"{cell.name}: aspect drift {drift*100:.2f}% > {ASPECT_TOLERANCE*100:.0f}% "
        f"tolerance.\n"
        f"  source: {cell.width}x{cell.height} (aspect {cell.source_aspect:.4f})\n"
        f"  output: {out_w}x{out_h} (aspect {out_aspect:.4f})\n"
        f"  Standard ratio source should round-trip via the matching "
        f"named profile — autodetect or canvas selection regressed."
    )


# ---------------------------------------------------------------------------
# Cells — every standard ratio × every render path
# ---------------------------------------------------------------------------


# Each (width, height, label) — label so test IDs show the aspect.
_STANDARD_RESOLUTIONS = [
    # 16:9 landscape — two resolutions to prove the assertion isn't
    # "output is always 1920x1080" coincidence.
    (1920, 1080, "16x9-1080p"),
    (1280, 720, "16x9-720p"),
    # 9:16 portrait — same: two resolutions.
    (720, 1280, "9x16-720p"),
    (1080, 1920, "9x16-1080p"),
    # 1:1 square.
    (1080, 1080, "1x1-1080p"),
]


def _build_cells_for_path(operation: str, render_runtime: str, path_label: str) -> list[AspectCell]:
    return [
        AspectCell(
            name=f"{aspect_label}__{path_label}",
            width=w, height=h,
            operation=operation, render_runtime=render_runtime,
        )
        for w, h, aspect_label in _STANDARD_RESOLUTIONS
    ]


FAST_CELLS = (
    _build_cells_for_path("compose", "ffmpeg", "compose")
    + _build_cells_for_path("render", "ffmpeg", "render-via-ffmpeg")
)

FULL_CELLS = FAST_CELLS + _build_cells_for_path(
    "render", "remotion", "render-via-remotion"
)


# ---------------------------------------------------------------------------
# Fast suite — compose + render-via-ffmpeg paths
# ---------------------------------------------------------------------------


@pytest.mark.render_matrix_fast
@pytest.mark.parametrize("cell", FAST_CELLS, ids=lambda c: c.name)
def test_aspect_preserved_fast(cell: AspectCell, tmp_path: Path) -> None:
    """Standard source aspect must round-trip exactly through every
    fast render path. If output drifts > 1%, autodetect or canvas
    selection regressed."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")

    source = _write_testsrc(tmp_path / "src.mp4", cell.width, cell.height)
    output = tmp_path / "out.mp4"

    inputs = _build_inputs(cell, source, output)
    result = VideoCompose().execute(inputs)
    assert result.success, f"{cell.name}: render failed: {result.error}"

    _assert_aspect_preserved(cell, output)


# ---------------------------------------------------------------------------
# Full suite — adds Remotion path
# ---------------------------------------------------------------------------


@pytest.mark.render_matrix_full
@pytest.mark.parametrize("cell", FULL_CELLS, ids=lambda c: c.name)
def test_aspect_preserved_full(cell: AspectCell, tmp_path: Path) -> None:
    """Full aspect-preservation matrix — same shapes plus Remotion.
    Remotion cells skip when npx isn't available (see conftest)."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")

    source = _write_testsrc(tmp_path / "src.mp4", cell.width, cell.height)
    output = tmp_path / "out.mp4"

    inputs = _build_inputs(cell, source, output)
    result = VideoCompose().execute(inputs)
    assert result.success, f"{cell.name}: render failed: {result.error}"

    _assert_aspect_preserved(cell, output)
