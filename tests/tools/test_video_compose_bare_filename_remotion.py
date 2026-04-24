"""Regression tests for fork issue #31.

`_remotion_render._collect_local_asset` used to skip relative paths
with the comment "already filename-only — will resolve under
public-dir". That assumption was wrong: callers passing a bare
filename (e.g. "scene-1-urban.mp4") expected the file to be served
from the workspace cwd, NOT from Remotion's bundle public/ subdir
(which is `/tmp/remotion-webpack-bundle-*/public/` and contains
nothing the caller put there).

Result: `local_asset_paths` ended up empty, `--public-dir` was
never set on the Remotion CLI, Remotion served its bundle's
default empty public/ folder, and every asset 404'd from
`http://localhost:3000/public/<filename>`.

Fix: resolve bare filenames against `Path.cwd()` before deciding
whether to collect.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tools.video.video_compose import VideoCompose


def _write_testsrc(path: Path, width: int, height: int, duration: float = 1.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i",
            f"testsrc2=size={width}x{height}:duration={duration}:rate=30",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-t", str(duration), str(path),
        ],
        check=True, capture_output=True, timeout=60,
    )
    return path


def _capture_remotion_cmd(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stub run_command so the npx call doesn't actually run; capture
    the cmd + the props.json contents the tool wrote."""
    captured: dict[str, Any] = {"cmd": None, "props": None}

    def fake_run(self, cmd, *, timeout=None, cwd=None):
        captured["cmd"] = list(cmd)
        if "--props" in cmd:
            props_path = Path(cmd[cmd.index("--props") + 1])
            if props_path.exists():
                captured["props"] = json.loads(props_path.read_text())
        # write a stub output so the post-render existence check passes
        if (
            len(cmd) >= 6
            and cmd[0] == "npx"
            and cmd[1] == "remotion"
            and cmd[2] == "render"
        ):
            output_path = Path(cmd[5])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"\x00")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(VideoCompose, "run_command", fake_run)
    return captured


def test_bare_filename_in_cwd_sets_public_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exact #31 repro: cuts[].source is a bare filename, the
    file lives in cwd. Without the fix, --public-dir is never
    passed → Remotion serves an empty bundle dir → 404. With the
    fix, --public-dir points at cwd."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")

    # Workspace = tmp_path. Source lives there as a bare filename.
    monkeypatch.chdir(tmp_path)
    _write_testsrc(tmp_path / "scene-1-urban.mp4", 1280, 720)
    output = tmp_path / "out.mp4"

    captured = _capture_remotion_cmd(monkeypatch)

    inputs = {
        "operation": "remotion_render",
        "edit_decisions": {
            "render_runtime": "remotion",
            "renderer_family": "cinematic-trailer",
            "cuts": [
                {"id": "c1", "source": "scene-1-urban.mp4",
                 "in_seconds": 0, "out_seconds": 1},
            ],
        },
        "output_path": str(output),
    }
    result = VideoCompose().execute(inputs)
    assert result.success, f"render failed: {result.error}"

    cmd = captured["cmd"]
    assert cmd is not None, "remotion CLI was not invoked"
    assert "--public-dir" in cmd, (
        f"--public-dir missing from cmd. The bare-filename collector "
        f"silently skipped the asset, so Remotion served its bundle's "
        f"empty public/ subdir → 404 for every cut. cmd: {cmd}"
    )

    public_dir = Path(cmd[cmd.index("--public-dir") + 1])
    assert public_dir.resolve() == tmp_path.resolve(), (
        f"--public-dir should point at the workspace cwd "
        f"({tmp_path.resolve()}); got {public_dir.resolve()}"
    )


def test_bare_filename_rewritten_to_relative_under_public_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The collector pairs --public-dir with rewriting each cut's
    source to its relative position under that dir. Verify the
    bare-filename case round-trips: source name stays the same
    because public-dir IS the cwd containing the file."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")

    monkeypatch.chdir(tmp_path)
    _write_testsrc(tmp_path / "scene-2-park.mp4", 720, 1280)
    output = tmp_path / "out.mp4"

    captured = _capture_remotion_cmd(monkeypatch)

    inputs = {
        "operation": "remotion_render",
        "edit_decisions": {
            "render_runtime": "remotion",
            "renderer_family": "cinematic-trailer",
            "cuts": [
                {"id": "c1", "source": "scene-2-park.mp4",
                 "in_seconds": 0, "out_seconds": 1},
            ],
        },
        "output_path": str(output),
    }
    VideoCompose().execute(inputs)

    props = captured["props"]
    assert props is not None, "props.json wasn't written"
    # The cuts→scenes adapter fires for cinematic-trailer; check the
    # rewritten src on the resulting scene.
    scene = props["scenes"][0]
    assert scene["src"] == "scene-2-park.mp4", (
        f"After --public-dir is set to the workspace, scene src should "
        f"stay the bare filename (it's already relative to public-dir). "
        f"Got: {scene['src']!r}"
    )


def test_bare_filename_not_in_cwd_falls_through_silently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the file doesn't exist at cwd, the collector falls through
    (no public-dir set). Doesn't crash — Remotion's own 404 will
    surface a clear stderr message in the real path. This isolates
    the fix to "resolve cwd, then check exists" — the existence
    check still gates collection."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")

    monkeypatch.chdir(tmp_path)  # cwd is tmp_path, but file isn't there
    output = tmp_path / "out.mp4"

    captured = _capture_remotion_cmd(monkeypatch)

    inputs = {
        "operation": "remotion_render",
        "edit_decisions": {
            "render_runtime": "remotion",
            "renderer_family": "cinematic-trailer",
            "cuts": [
                {"id": "c1", "source": "missing-asset.mp4",
                 "in_seconds": 0, "out_seconds": 1},
            ],
        },
        "output_path": str(output),
    }
    # Stubbed run_command always returns 0, so result.success will be
    # True regardless. The behavioral assertion is "no public-dir set
    # because nothing collected" — Remotion would 404 in real life;
    # here we just verify the resolver didn't synthesize a wrong path.
    VideoCompose().execute(inputs)

    cmd = captured["cmd"]
    assert cmd is not None
    assert "--public-dir" not in cmd, (
        f"file doesn't exist at cwd → collector should fall through, "
        f"NOT set --public-dir to a phantom path. cmd: {cmd}"
    )


def test_absolute_path_still_works_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Backwards-compat: absolute paths still resolve via the
    original code path. The bare-filename fix is additive — doesn't
    break the existing absolute-path callers."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")

    src = _write_testsrc(tmp_path / "absolute_clip.mp4", 1920, 1080)
    output = tmp_path / "out.mp4"

    captured = _capture_remotion_cmd(monkeypatch)

    inputs = {
        "operation": "remotion_render",
        "edit_decisions": {
            "render_runtime": "remotion",
            "renderer_family": "cinematic-trailer",
            "cuts": [
                {"id": "c1", "source": str(src),  # absolute
                 "in_seconds": 0, "out_seconds": 1},
            ],
        },
        "output_path": str(output),
    }
    result = VideoCompose().execute(inputs)
    assert result.success

    cmd = captured["cmd"]
    assert "--public-dir" in cmd
    public_dir = Path(cmd[cmd.index("--public-dir") + 1])
    # public-dir should be tmp_path (the parent of the absolute clip).
    assert public_dir.resolve() == tmp_path.resolve()


def test_relative_subpath_resolves_against_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Beyond bare filenames, relative paths with subdirs (e.g.
    `assets/scene-1.mp4`) should also resolve against cwd. Same
    fix covers both. The single-file case picks the file's parent
    as public-dir, so the rewritten src ends up as the basename
    relative to that minimum-spanning dir."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")

    monkeypatch.chdir(tmp_path)
    asset_dir = tmp_path / "assets"
    _write_testsrc(asset_dir / "scene-1.mp4", 1280, 720)
    output = tmp_path / "out.mp4"

    captured = _capture_remotion_cmd(monkeypatch)

    inputs = {
        "operation": "remotion_render",
        "edit_decisions": {
            "render_runtime": "remotion",
            "renderer_family": "cinematic-trailer",
            "cuts": [
                {"id": "c1", "source": "assets/scene-1.mp4",
                 "in_seconds": 0, "out_seconds": 1},
            ],
        },
        "output_path": str(output),
    }
    result = VideoCompose().execute(inputs)
    assert result.success

    cmd = captured["cmd"]
    assert "--public-dir" in cmd, "relative subpath must still set --public-dir"
    public_dir = Path(cmd[cmd.index("--public-dir") + 1])
    # public-dir lands on the minimum-spanning dir for the resolved
    # asset paths — for a single asset that's its parent dir.
    assert public_dir.resolve() == asset_dir.resolve()
    # Rewritten src is relative to public-dir → just the basename.
    props = captured["props"]
    scene = props["scenes"][0]
    assert scene["src"] == "scene-1.mp4"
