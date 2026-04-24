"""Regression tests for fork issue #1.

The Remotion Rust compositor panics on mixed video+image cuts:

    thread '<unnamed>' panicked at rust/frame_cache.rs:257:43:
    called `Option::unwrap()` on a `None` value

killing every render that wants a still-image scene. `_render` now
preprocesses image-source cuts into short MP4 loops via ffmpeg before
handing the composition to Remotion, so the compositor only ever sees
video.

These tests stub subprocess and the Remotion dispatch so nothing real
runs. They verify:

  1. Image cuts (by file extension) get rewritten to generated MP4s,
     the ffmpeg command targets the correct canvas dims and duration,
     and the resulting cuts[].source points into the preprocess
     tempdir.
  2. Asset-manifest-declared image type also triggers preprocessing
     even when the extension would pass as video.
  3. Video cuts pass through untouched — no preprocess, no tempdir.
  4. Profile-driven canvas: named profile (tiktok) yields 1080x1920
     dims; explicit dict {width,height,fps} is honored; missing
     profile defaults to 1920x1080@30.
  5. The preprocess tempdir is cleaned up after the Remotion dispatch
     completes, on both success and failure paths.
  6. When ffmpeg is missing, the preprocess no-ops and lets Remotion
     fail with the original message (clearer than silently dropping
     the image scenes).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from tools.base_tool import ToolResult
from tools.video.video_compose import VideoCompose


def _make_file(path: Path, content: bytes = b"\x00") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _install_render_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    remotion_result: ToolResult | None = None,
) -> dict[str, Any]:
    """Stub _pre_compose_validation, _remotion_render, _needs_remotion,
    and ffmpeg subprocess.run so nothing real executes. Capture what the
    dispatch receives for assertions."""
    captured: dict[str, Any] = {"ffmpeg_cmds": []}

    def fake_pre(self, edit_decisions, resolved_cuts, scene_plan=None,
                 proposal_packet=None, asset_manifest=None, script=None):
        return None

    def fake_needs_remotion(self, cuts):
        return True  # force Remotion path so preprocessor runs

    def fake_remotion_render(self, inputs):
        captured["remotion_inputs"] = inputs
        # Write a fake output so any post-render check passes.
        out = Path(inputs["output_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\x00")
        return remotion_result or ToolResult(success=True, data={"output": str(out)})

    def fake_final_review(self, output_path, edit_decisions=None, proposal_packet=None,
                          narration_transcript_path=None, script_text=None):
        # Bypass ffprobe + full review — unrelated to the preprocess
        # contract these tests assert.
        return {
            "version": "1.0",
            "output_path": str(output_path),
            "status": "pass",
            "checks": {},
            "issues_found": [],
            "recommended_action": "ship",
        }

    monkeypatch.setattr(VideoCompose, "_pre_compose_validation", fake_pre)
    monkeypatch.setattr(VideoCompose, "_needs_remotion", fake_needs_remotion)
    monkeypatch.setattr(VideoCompose, "_remotion_render", fake_remotion_render)
    monkeypatch.setattr(VideoCompose, "_run_final_review", fake_final_review)

    real_run = subprocess.run

    def fake_subprocess_run(cmd, *args, **kwargs):
        if cmd and cmd[0].endswith("ffmpeg"):
            captured["ffmpeg_cmds"].append(list(cmd))
            # Write the expected output file so the preprocessor thinks
            # we succeeded.
            if "-y" in cmd:
                out = Path(cmd[-1])
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(b"fake-mp4")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(VideoCompose, "_which_ffmpeg", staticmethod(lambda: "/usr/bin/ffmpeg"))

    return captured


# ---------------------------------------------------------------------------
# 1. Image cut (by extension) gets preprocessed
# ---------------------------------------------------------------------------


def test_image_cut_by_extension_preprocessed_before_remotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _install_render_stubs(monkeypatch)

    ws = tmp_path / "ws"
    video_clip = _make_file(ws / "c1.mp4")
    image = _make_file(ws / "hero.png", b"fake-png")
    output = ws / "out.mp4"

    inputs = {
        "operation": "render",
        "edit_decisions": {
            "render_runtime": "remotion",
            "renderer_family": "cinematic-trailer",
            "cuts": [
                {"id": "c1", "source": str(video_clip), "in_seconds": 0, "out_seconds": 3},
                {"id": "c2", "source": str(image), "in_seconds": 3, "out_seconds": 6},
            ],
        },
        "asset_manifest": {
            "assets": [
                {"id": "c1", "path": str(video_clip)},
                {"id": "c2", "path": str(image)},
            ],
        },
        "output_path": str(output),
    }

    result = VideoCompose()._render(inputs)
    assert result.success, result.error

    # One ffmpeg invocation ran (for the image cut only).
    assert len(captured["ffmpeg_cmds"]) == 1
    cmd = captured["ffmpeg_cmds"][0]
    assert "-loop" in cmd and "1" in cmd
    assert str(image) in cmd
    # Duration = 6 - 3 = 3s → passed to ffmpeg via -t.
    t_idx = cmd.index("-t")
    assert float(cmd[t_idx + 1]) == pytest.approx(3.0)
    # Output is a path inside a preprocess tempdir next to output_path.
    out_cut_path = Path(cmd[-1])
    assert out_cut_path.suffix == ".mp4"
    assert out_cut_path.parent.name.startswith("remotion_image_preprocess_")

    # The cut handed to Remotion was rewritten to the generated MP4.
    passed_cuts = captured["remotion_inputs"]["edit_decisions"]["cuts"]
    assert passed_cuts[0]["source"] == str(video_clip)  # untouched
    assert passed_cuts[1]["source"] == str(out_cut_path)  # rewritten


# ---------------------------------------------------------------------------
# 2. Asset-manifest type="image" triggers preprocessing
# ---------------------------------------------------------------------------


def test_image_cut_by_manifest_type_preprocessed_even_without_image_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Director sometimes writes an image to a .dat / .bin file or uses
    an ID in cuts[].source. The fallback is the asset_manifest's
    declared `type: "image"`."""
    captured = _install_render_stubs(monkeypatch)

    ws = tmp_path / "ws"
    # No image extension — test the manifest-type fallback.
    weird_image = _make_file(ws / "hero.blob")
    output = ws / "out.mp4"

    inputs = {
        "operation": "render",
        "edit_decisions": {
            "render_runtime": "remotion",
            "renderer_family": "cinematic-trailer",
            "cuts": [
                {"id": "c1", "source": str(weird_image), "in_seconds": 0, "out_seconds": 2},
            ],
        },
        "asset_manifest": {
            "assets": [{"id": "c1", "path": str(weird_image), "type": "image"}],
        },
        "output_path": str(output),
    }

    result = VideoCompose()._render(inputs)
    assert result.success, result.error
    assert len(captured["ffmpeg_cmds"]) == 1


# ---------------------------------------------------------------------------
# 3. Video-only cuts pass through untouched
# ---------------------------------------------------------------------------


def test_video_only_cuts_do_not_trigger_preprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _install_render_stubs(monkeypatch)

    ws = tmp_path / "ws"
    clip_a = _make_file(ws / "a.mp4")
    clip_b = _make_file(ws / "b.mp4")
    output = ws / "out.mp4"

    inputs = {
        "operation": "render",
        "edit_decisions": {
            "render_runtime": "remotion",
            "renderer_family": "cinematic-trailer",
            "cuts": [
                {"id": "c1", "source": str(clip_a), "in_seconds": 0, "out_seconds": 3},
                {"id": "c2", "source": str(clip_b), "in_seconds": 3, "out_seconds": 6},
            ],
        },
        "asset_manifest": {
            "assets": [
                {"id": "c1", "path": str(clip_a)},
                {"id": "c2", "path": str(clip_b)},
            ],
        },
        "output_path": str(output),
    }

    result = VideoCompose()._render(inputs)
    assert result.success, result.error

    # No ffmpeg calls, no tempdir — fast path.
    assert captured["ffmpeg_cmds"] == []


# ---------------------------------------------------------------------------
# 4. Profile-driven canvas dims
# ---------------------------------------------------------------------------


def test_image_preprocess_honors_named_profile_dims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _install_render_stubs(monkeypatch)

    ws = tmp_path / "ws"
    image = _make_file(ws / "hero.png")
    output = ws / "out.mp4"

    inputs = {
        "operation": "render",
        "profile": "tiktok",  # 1080x1920 @ 30
        "edit_decisions": {
            "render_runtime": "remotion",
            "renderer_family": "cinematic-trailer",
            "cuts": [
                {"id": "c1", "source": str(image), "in_seconds": 0, "out_seconds": 2},
            ],
        },
        "asset_manifest": {"assets": [{"id": "c1", "path": str(image)}]},
        "output_path": str(output),
    }

    VideoCompose()._render(inputs)
    cmd = captured["ffmpeg_cmds"][0]
    vf_idx = cmd.index("-vf")
    vf = cmd[vf_idx + 1]
    assert "1080:1920" in vf


def test_image_preprocess_honors_dict_profile_dims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _install_render_stubs(monkeypatch)

    ws = tmp_path / "ws"
    image = _make_file(ws / "hero.png")
    output = ws / "out.mp4"

    inputs = {
        "operation": "render",
        "profile": {"width": 1280, "height": 720, "fps": 24},
        "edit_decisions": {
            "render_runtime": "remotion",
            "renderer_family": "cinematic-trailer",
            "cuts": [
                {"id": "c1", "source": str(image), "in_seconds": 0, "out_seconds": 2},
            ],
        },
        "asset_manifest": {"assets": [{"id": "c1", "path": str(image)}]},
        "output_path": str(output),
    }

    VideoCompose()._render(inputs)
    cmd = captured["ffmpeg_cmds"][0]
    assert "1280:720" in cmd[cmd.index("-vf") + 1]
    assert cmd[cmd.index("-r") + 1] == "24"


def test_image_preprocess_defaults_to_1920x1080_at_30(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _install_render_stubs(monkeypatch)

    ws = tmp_path / "ws"
    image = _make_file(ws / "hero.png")
    output = ws / "out.mp4"

    inputs = {
        "operation": "render",
        "edit_decisions": {
            "render_runtime": "remotion",
            "renderer_family": "cinematic-trailer",
            "cuts": [
                {"id": "c1", "source": str(image), "in_seconds": 0, "out_seconds": 2},
            ],
        },
        "asset_manifest": {"assets": [{"id": "c1", "path": str(image)}]},
        "output_path": str(output),
    }

    VideoCompose()._render(inputs)
    cmd = captured["ffmpeg_cmds"][0]
    assert "1920:1080" in cmd[cmd.index("-vf") + 1]
    assert cmd[cmd.index("-r") + 1] == "30"


# ---------------------------------------------------------------------------
# 5. Tempdir cleanup on success AND failure
# ---------------------------------------------------------------------------


def test_preprocess_tempdir_cleaned_up_on_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _install_render_stubs(monkeypatch)

    ws = tmp_path / "ws"
    image = _make_file(ws / "hero.png")
    output = ws / "out.mp4"

    VideoCompose()._render({
        "operation": "render",
        "edit_decisions": {
            "render_runtime": "remotion",
            "renderer_family": "cinematic-trailer",
            "cuts": [
                {"id": "c1", "source": str(image), "in_seconds": 0, "out_seconds": 2},
            ],
        },
        "asset_manifest": {"assets": [{"id": "c1", "path": str(image)}]},
        "output_path": str(output),
    })

    tempdirs = [p for p in ws.iterdir() if p.name.startswith("remotion_image_preprocess_")]
    assert tempdirs == [], (
        f"image-preprocess tempdir not cleaned up: {tempdirs}"
    )


def test_preprocess_tempdir_cleaned_up_on_remotion_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Remotion failure still cleans up the preprocess tempdir. The
    try/finally around _remotion_render must run cleanup regardless."""
    captured = _install_render_stubs(
        monkeypatch,
        remotion_result=ToolResult(success=False, error="Remotion boom"),
    )

    ws = tmp_path / "ws"
    image = _make_file(ws / "hero.png")
    output = ws / "out.mp4"

    result = VideoCompose()._render({
        "operation": "render",
        "edit_decisions": {
            "render_runtime": "remotion",
            "renderer_family": "cinematic-trailer",
            "cuts": [
                {"id": "c1", "source": str(image), "in_seconds": 0, "out_seconds": 2},
            ],
        },
        "asset_manifest": {"assets": [{"id": "c1", "path": str(image)}]},
        "output_path": str(output),
    })

    assert not result.success

    tempdirs = [p for p in ws.iterdir() if p.name.startswith("remotion_image_preprocess_")]
    assert tempdirs == [], (
        f"tempdir leaked on Remotion failure: {tempdirs}"
    )


# ---------------------------------------------------------------------------
# 6. Missing ffmpeg → no-op, passes raw to Remotion
# ---------------------------------------------------------------------------


def test_missing_ffmpeg_bypasses_preprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If ffmpeg isn't on PATH, the preprocess logs a warning and
    returns the cuts unchanged rather than silently dropping the
    image scenes. Remotion then fails with its original compositor
    panic — which at least points at the real problem instead of
    producing a black render."""
    captured = _install_render_stubs(monkeypatch)
    monkeypatch.setattr(VideoCompose, "_which_ffmpeg", staticmethod(lambda: None))

    ws = tmp_path / "ws"
    image = _make_file(ws / "hero.png")
    output = ws / "out.mp4"

    VideoCompose()._render({
        "operation": "render",
        "edit_decisions": {
            "render_runtime": "remotion",
            "renderer_family": "cinematic-trailer",
            "cuts": [
                {"id": "c1", "source": str(image), "in_seconds": 0, "out_seconds": 2},
            ],
        },
        "asset_manifest": {"assets": [{"id": "c1", "path": str(image)}]},
        "output_path": str(output),
    })

    # ffmpeg wasn't available → no ffmpeg runs, cut passed raw.
    assert captured["ffmpeg_cmds"] == []
    assert captured["remotion_inputs"]["edit_decisions"]["cuts"][0]["source"] == str(image)
