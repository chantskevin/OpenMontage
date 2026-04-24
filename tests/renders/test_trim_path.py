"""Trim-path render tests — sources LONGER than the cut windows.

The render matrix synthesizes sources at exactly the cut duration, so
cut.in_seconds=0 / out_seconds=cut_duration always uses the WHOLE
source. The trim path (cut window smaller than source) is the
production reality on this fork — Veo and APIYI return ~8s clips,
the edit-director trims each to 2-5s segments. Without a test here,
trim-arithmetic regressions ship.

Specific shape exercised:
  - 3 input clips, each 8s, 9:16 portrait (720x1280)
  - cuts pick non-zero in_seconds windows summing to ~10s
  - canvas autodetects to portrait (instagram_reels, 1080x1920)
  - duration matches edit timeline within ±0.5s
  - black-frame ratio < 10% (proves trim windows actually rendered)
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from tools.video.video_compose import VideoCompose


@dataclass
class TrimCell:
    """One trim configuration: source duration > cut window."""
    name: str
    operation: str        # compose / render
    render_runtime: str   # ffmpeg / remotion
    source_duration: float
    width: int
    height: int
    cuts: list[tuple[float, float]]  # [(in_s, out_s), ...]

    @property
    def expected_duration(self) -> float:
        return sum(out_s - in_s for in_s, out_s in self.cuts)


def _write_testsrc(path: Path, width: int, height: int, duration: float) -> Path:
    """testsrc2 source — moving color bars + clock so any black frame
    in the output is obviously a bug, not source content."""
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


def _measure_black_ratio(path: Path) -> float:
    """ffmpeg blackdetect → fraction of duration flagged as black."""
    proc = subprocess.run(
        [
            "ffmpeg", "-i", str(path),
            "-vf", "blackdetect=d=0.05:pix_th=0.10",
            "-an", "-f", "null", "-",
        ],
        capture_output=True, text=True, timeout=60,
    )
    output = proc.stderr or ""
    total_black = 0.0
    for match in re.finditer(r"black_duration:(\d+\.?\d*)", output):
        total_black += float(match.group(1))

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


def _build_inputs(cell: TrimCell, sources: list[Path], output: Path) -> dict:
    cuts = [
        {
            "id": f"c{i}",
            "source": f"a{i}" if cell.operation != "remotion_render" else str(sources[i]),
            "in_seconds": in_s,
            "out_seconds": out_s,
        }
        for i, (in_s, out_s) in enumerate(cell.cuts)
    ]
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
        inputs["asset_manifest"] = asset_manifest
    return inputs


def _assert_invariants(cell: TrimCell, output: Path) -> None:
    assert output.exists(), f"{cell.name}: output missing"
    assert output.stat().st_size > 0, f"{cell.name}: output is 0 bytes"

    actual = VideoCompose._probe_output_metadata(output)
    assert actual is not None, f"{cell.name}: ffprobe failed"

    w, h = actual["width"], actual["height"]
    if cell.height > cell.width:
        assert h > w, (
            f"{cell.name}: portrait sources should yield portrait canvas; "
            f"got {w}x{h}. (Canvas autodetect regression?)"
        )
    elif cell.width > cell.height:
        assert w > h, f"{cell.name}: expected landscape, got {w}x{h}"

    actual_dur = actual["duration_seconds"]
    expected = cell.expected_duration
    assert abs(actual_dur - expected) <= 0.5, (
        f"{cell.name}: rendered {actual_dur}s vs edit timeline {expected}s "
        f"(>0.5s drift). Trim-arithmetic regression — cut.in_seconds / "
        f".out_seconds windows aren't being honored."
    )

    black_ratio = _measure_black_ratio(output)
    assert black_ratio < 0.10, (
        f"{cell.name}: {black_ratio*100:.1f}% black frames (threshold 10%). "
        f"Trim window may be falling outside the source — check "
        f"cut.in_seconds vs source duration ({cell.source_duration}s)."
    )


# ---------------------------------------------------------------------------
# The shapes
# ---------------------------------------------------------------------------


# Three 8-second 9:16 sources → 10-second portrait output via three
# different trim distributions. The exact production shape: Veo/APIYI
# return 8s clips, edit-director trims each to a different window.
TRIM_CELLS_FAST = [
    TrimCell(
        name="3x8s-9x16__compose__trim-3-3-4",
        operation="compose",
        render_runtime="ffmpeg",
        source_duration=8.0,
        width=720, height=1280,
        # cut 0-3, 0-3, 0-4 from each source = 10s total
        cuts=[(0.0, 3.0), (0.0, 3.0), (0.0, 4.0)],
    ),
    TrimCell(
        name="3x8s-9x16__compose__trim-mid-windows",
        operation="compose",
        render_runtime="ffmpeg",
        source_duration=8.0,
        width=720, height=1280,
        # Mid-source windows: skip Veo's 2s static-intro opener.
        # Each cut: in_seconds=2, out_seconds=5 → 3s per cut → 9s total.
        cuts=[(2.0, 5.0), (2.0, 5.0), (2.0, 5.0)],
    ),
    TrimCell(
        name="3x8s-9x16__compose__trim-uneven",
        operation="compose",
        render_runtime="ffmpeg",
        source_duration=8.0,
        width=720, height=1280,
        # Asymmetric trim — 2s + 3s + 5s = 10s total, different
        # in/out per source.
        cuts=[(0.0, 2.0), (1.0, 4.0), (3.0, 8.0)],
    ),
]


# Same 3x8s-9x16 shape via the render-via-ffmpeg path (operation=render).
# Exercises the ID-resolution + canvas autodetect + trim chain together.
TRIM_CELLS_FULL = TRIM_CELLS_FAST + [
    TrimCell(
        name="3x8s-9x16__render-via-ffmpeg__trim-3-3-4",
        operation="render",
        render_runtime="ffmpeg",
        source_duration=8.0,
        width=720, height=1280,
        cuts=[(0.0, 3.0), (0.0, 3.0), (0.0, 4.0)],
    ),
    TrimCell(
        name="3x8s-9x16__render-via-remotion__trim-3-3-4",
        operation="render",
        render_runtime="remotion",
        source_duration=8.0,
        width=720, height=1280,
        cuts=[(0.0, 3.0), (0.0, 3.0), (0.0, 4.0)],
    ),
]


@pytest.mark.render_matrix_fast
@pytest.mark.parametrize("cell", TRIM_CELLS_FAST, ids=lambda c: c.name)
def test_trim_path_3x8s_portrait_to_10s(cell: TrimCell, tmp_path: Path) -> None:
    """3 × 8s 9:16 sources → ~10s portrait output via trim windows.

    The exact shape Veo/APIYI runs hit in production: each clip is 8s,
    each cut takes a 2-5s window. Catches:
      - cut.in_seconds / out_seconds honored (duration assertion)
      - canvas autodetects portrait from source dims (aspect assertion)
      - trim windows actually render frames (blackdetect assertion)
      - asset_manifest ID resolution (cuts use 'a0', 'a1', 'a2')
    """
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")

    sources = [
        _write_testsrc(
            tmp_path / f"src_{i}.mp4",
            cell.width, cell.height, cell.source_duration,
        )
        for i in range(len(cell.cuts))
    ]
    output = tmp_path / "out.mp4"

    inputs = _build_inputs(cell, sources, output)
    result = VideoCompose().execute(inputs)
    assert result.success, f"{cell.name}: render failed: {result.error}"

    _assert_invariants(cell, output)


@pytest.mark.render_matrix_full
@pytest.mark.parametrize("cell", TRIM_CELLS_FULL, ids=lambda c: c.name)
def test_trim_path_full_matrix(cell: TrimCell, tmp_path: Path) -> None:
    """Full trim matrix: same shapes as fast plus render/remotion paths.
    Remotion cells skip when npx isn't available (see conftest)."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")

    sources = [
        _write_testsrc(
            tmp_path / f"src_{i}.mp4",
            cell.width, cell.height, cell.source_duration,
        )
        for i in range(len(cell.cuts))
    ]
    output = tmp_path / "out.mp4"

    inputs = _build_inputs(cell, sources, output)
    result = VideoCompose().execute(inputs)
    assert result.success, f"{cell.name}: render failed: {result.error}"

    _assert_invariants(cell, output)
