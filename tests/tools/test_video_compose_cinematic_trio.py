"""Regression tests for fork issues #26/#27/#28 — the cinematic-trio.

All three were observed on the same render:
  #26: 30s output for a 10s edit_decisions timeline
  #27: 1920x1080 landscape output from 720x1280 portrait sources
  #28: 99.9% black output

Reporter conjectured a shared root cause. Confirmed:
`CinematicRenderer.tsx` reads `props.scenes[]` with a specific
shape. `_remotion_render` was passing `edit_decisions` (which has
`cuts[]`) directly. The result: `scenes.length === 0`, so:
  - `calculateCinematicMetadata` defaults `durationInFrames` to
    `30 * fps` (#26)
  - `scenes.map(...)` renders nothing → empty
    `<AbsoluteFill style={{backgroundColor: "#000000"}}>` (#28)
  - Canvas defaults to hardcoded `width: 1920, height: 1080` (#27)

Cherry-picked fixes:
  - `3e77b8c` cuts→scenes adapter (closes #26 and #28)
  - `4f9ac35` canvas autodetect from source orientation (closes #27)

These tests assert the invariants directly without invoking
Remotion, so they run in the offline test suite.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tools.base_tool import ToolResult
from tools.video.video_compose import VideoCompose


# ---------------------------------------------------------------------------
# Helpers — stub out the Remotion CLI invocation, capture the props
# ---------------------------------------------------------------------------


def _install_capture_stub(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Capture the props.json that _remotion_render writes for inspection.

    We monkeypatch run_command so the npx remotion render CLI is never
    actually invoked. Before the patched run_command "returns," we grab
    the props file the tool wrote and stash it for the test."""
    captured: dict[str, Any] = {}

    real_run = VideoCompose.run_command

    def fake_run(self, cmd, *, timeout=None, cwd=None):
        # Locate the --props arg and read the file.
        if "--props" in cmd:
            props_path = Path(cmd[cmd.index("--props") + 1])
            if props_path.exists():
                captured["props"] = json.loads(props_path.read_text())
        # Output_path is the 6th token: `npx remotion render <bundle>
        # <composition_id> <output_path> --props <file>`.
        if (
            len(cmd) >= 6
            and cmd[0] == "npx"
            and cmd[1] == "remotion"
            and cmd[2] == "render"
        ):
            output_path = Path(cmd[5])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"\x00")
        captured["cmd"] = list(cmd)
        import subprocess
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(VideoCompose, "run_command", fake_run)
    return captured


# ---------------------------------------------------------------------------
# 1. cuts → scenes adapter (fixes #26 + #28)
# ---------------------------------------------------------------------------


def test_cinematic_render_translates_cuts_to_scenes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exact #26/#28 failure shape: edit_decisions has cuts[] but
    no scenes[]. Before the adapter, props.scenes would be undefined
    and CinematicRenderer would render 30s of pure black. After the
    adapter, scenes[] is populated and durations sum correctly."""
    captured = _install_capture_stub(monkeypatch)

    asset_01 = tmp_path / "asset_01.mp4"
    asset_02 = tmp_path / "asset_02.mp4"
    asset_03 = tmp_path / "asset_03.mp4"
    for f in (asset_01, asset_02, asset_03):
        f.write_bytes(b"\x00")

    inputs = {
        "edit_decisions": {
            "renderer_family": "cinematic-trailer",
            "cuts": [
                {"id": "cut_01", "source": str(asset_01),
                 "in_seconds": 0, "out_seconds": 2},
                {"id": "cut_02", "source": str(asset_02),
                 "in_seconds": 0, "out_seconds": 3},
                {"id": "cut_03", "source": str(asset_03),
                 "in_seconds": 0, "out_seconds": 5},
            ],
        },
        "output_path": str(tmp_path / "out.mp4"),
    }
    result = VideoCompose()._remotion_render(inputs)
    assert result.success, result.error

    props = captured.get("props")
    assert props is not None
    scenes = props.get("scenes")
    assert scenes is not None and len(scenes) == 3, (
        f"cuts → scenes adapter must populate props.scenes for "
        f"CinematicRenderer; got {scenes!r}. Without it #26 (duration "
        f"defaults to 30s) and #28 (no Video components render → "
        f"black output) reproduce."
    )

    # Scene durations match cut in/out spans (2s, 3s, 5s = 10s total).
    durations = [s["durationSeconds"] for s in scenes]
    assert durations == [2.0, 3.0, 5.0], (
        f"Scene durations must equal out_seconds - in_seconds; got {durations}."
    )
    # Total scene span (start + duration of last) is 10s, not 30s.
    last = scenes[-1]
    assert last["startSeconds"] + last["durationSeconds"] == 10.0


def test_cinematic_scenes_have_required_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The adapter output must include the fields CinematicRenderer
    reads: id, kind, src, startSeconds, durationSeconds. Missing any
    of these reproduces #28 (the scene renders nothing)."""
    captured = _install_capture_stub(monkeypatch)

    src = tmp_path / "scene.mp4"
    src.write_bytes(b"\x00")

    inputs = {
        "edit_decisions": {
            "renderer_family": "cinematic-trailer",
            "cuts": [{"id": "cut_01", "source": str(src),
                      "in_seconds": 0, "out_seconds": 4}],
        },
        "output_path": str(tmp_path / "out.mp4"),
    }
    VideoCompose()._remotion_render(inputs)

    scene = captured["props"]["scenes"][0]
    for field in ("id", "kind", "src", "startSeconds", "durationSeconds"):
        assert field in scene, f"Scene missing required field {field}: {scene}"
    assert scene["kind"] == "video"
    # src is rewritten to file:// URI for Remotion's <OffthreadVideo>.
    assert scene["src"].startswith("file://"), (
        f"Scene src must be file:// URI for Remotion; got {scene['src']}"
    )


def test_cinematic_in_seconds_becomes_trim_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When a cut starts mid-source (in_seconds > 0), the adapter
    must emit trimBeforeSeconds so CinematicRenderer's SceneVideo
    skips the right amount of the source."""
    captured = _install_capture_stub(monkeypatch)

    src = tmp_path / "scene.mp4"
    src.write_bytes(b"\x00")

    inputs = {
        "edit_decisions": {
            "renderer_family": "cinematic-trailer",
            "cuts": [{"id": "c1", "source": str(src),
                      "in_seconds": 1.5, "out_seconds": 4.0}],
        },
        "output_path": str(tmp_path / "out.mp4"),
    }
    VideoCompose()._remotion_render(inputs)

    scene = captured["props"]["scenes"][0]
    assert scene["trimBeforeSeconds"] == 1.5
    # durationSeconds is the visible span (out - in), NOT the source length.
    assert scene["durationSeconds"] == 2.5


def test_cinematic_existing_scenes_not_overwritten_by_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the caller already supplied scenes[], the adapter must NOT
    clobber them with cuts-derived data. This preserves the escape
    hatch for callers that build scenes natively."""
    captured = _install_capture_stub(monkeypatch)

    src = tmp_path / "scene.mp4"
    src.write_bytes(b"\x00")

    inputs = {
        "edit_decisions": {
            "renderer_family": "cinematic-trailer",
            "cuts": [{"id": "c1", "source": str(src),
                      "in_seconds": 0, "out_seconds": 5}],
            "scenes": [
                {"id": "manual", "kind": "video", "src": str(src),
                 "startSeconds": 0, "durationSeconds": 8.0},
            ],
        },
        "output_path": str(tmp_path / "out.mp4"),
    }
    VideoCompose()._remotion_render(inputs)

    scenes = captured["props"]["scenes"]
    assert len(scenes) == 1
    assert scenes[0]["id"] == "manual"
    assert scenes[0]["durationSeconds"] == 8.0


# ---------------------------------------------------------------------------
# 2. Adapter is gated on renderer_family (only fires for CinematicRenderer)
# ---------------------------------------------------------------------------


def test_explainer_renderer_does_not_get_scenes_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Explainer composition reads props.cuts[] natively. The
    cuts→scenes adapter must NOT fire for non-Cinematic renderers,
    since they don't expect a scenes field."""
    captured = _install_capture_stub(monkeypatch)

    src = tmp_path / "scene.mp4"
    src.write_bytes(b"\x00")

    inputs = {
        "edit_decisions": {
            "renderer_family": "explainer-data",
            "cuts": [{"id": "c1", "source": str(src),
                      "in_seconds": 0, "out_seconds": 5}],
        },
        "output_path": str(tmp_path / "out.mp4"),
    }
    VideoCompose()._remotion_render(inputs)

    props = captured["props"]
    assert "scenes" not in props, (
        f"Explainer reads cuts[] natively — adapter must not run. "
        f"Got props keys: {list(props.keys())}"
    )
    # cuts[] still gets file:// URI rewriting though.
    assert props["cuts"][0]["source"].startswith("file://")
