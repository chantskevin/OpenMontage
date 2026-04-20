"""Regression tests for _remotion_render's --public-dir path rewriting.

Remotion's <OffthreadVideo>/<Img> do not load file:// URIs — the CLI serves
assets over HTTP rooted at --public-dir and resolves any non-URL prop
string against it. _remotion_render therefore:

  1. Collects every local absolute asset path in cuts[].source and
     scenes[].src.
  2. Computes os.path.commonpath across them as the --public-dir.
  3. Rewrites each prop path to a filename relative to that dir.
  4. Passes --public-dir to the CLI.

These tests stub subprocess so Remotion never actually runs. They verify
the path-rewriting, --public-dir selection, commonpath ValueError guard,
and the stderr-tail surfacing that replaced the opaque
"returned non-zero exit status 1" error.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tools.video.video_compose import VideoCompose


def _make_file(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00")
    return path


def _invoke_remotion_render(
    tool: VideoCompose,
    inputs: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    stderr: str | None = None,
    raise_called_process_error: bool = False,
) -> tuple[Any, dict[str, Any], list[str]]:
    """Call _remotion_render with npx + run_command stubbed.

    Returns (result, captured_props, captured_cmd). The stub reads the
    props file the renderer wrote before deleting it, writes a fake output
    mp4 so the post-render existence check passes, and optionally raises
    CalledProcessError to exercise the error-surfacing branch.
    """
    import shutil as _shutil

    monkeypatch.setattr(_shutil, "which", lambda name: "/usr/bin/" + name)

    captured: dict[str, Any] = {"cmd": None, "props": None}

    def fake_run_command(
        self: VideoCompose,
        cmd: list[str],
        *,
        timeout: int | None = None,
        cwd: Path | None = None,
    ) -> subprocess.CompletedProcess:
        captured["cmd"] = list(cmd)
        # Read the props file the renderer wrote so tests can assert on
        # the rewritten paths. The renderer deletes the file in a finally
        # block, so we must grab it now.
        props_path_str = cmd[cmd.index("--props") + 1]
        props_path = Path(props_path_str)
        if props_path.exists():
            captured["props"] = json.loads(props_path.read_text(encoding="utf-8"))

        if raise_called_process_error:
            raise subprocess.CalledProcessError(
                returncode=1, cmd=cmd, output="", stderr=stderr or ""
            )

        # Write a fake output so the post-render existence check passes.
        # The output path sits immediately before "--props" in the CLI
        # invocation (see _remotion_render's `cmd` assembly).
        out_path = Path(cmd[cmd.index("--props") - 1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"fake-mp4")
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout="", stderr=""
        )

    monkeypatch.setattr(VideoCompose, "run_command", fake_run_command)

    result = tool._remotion_render(inputs)
    return result, captured["props"], captured["cmd"] or []


def test_flat_workspace_assets_rewritten_to_filenames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = tmp_path / "ws"
    clip_a = _make_file(ws / "a.mp4")
    clip_b = _make_file(ws / "b.mp4")
    output = tmp_path / "out" / "final.mp4"

    inputs = {
        "edit_decisions": {
            "cuts": [
                {"id": "c1", "source": str(clip_a), "in_seconds": 0, "out_seconds": 2},
                {"id": "c2", "source": str(clip_b), "in_seconds": 0, "out_seconds": 2},
            ],
        },
        "output_path": str(output),
    }

    result, props, cmd = _invoke_remotion_render(
        VideoCompose(), inputs, tmp_path, monkeypatch
    )

    assert result.success, result.error

    # --public-dir equals the shared workspace root.
    assert "--public-dir" in cmd
    assert cmd[cmd.index("--public-dir") + 1] == str(ws)

    # cuts[].source rewritten to filename-only.
    sources = [c["source"] for c in props["cuts"]]
    assert sources == ["a.mp4", "b.mp4"]


def test_nested_asset_paths_rewritten_relative_to_common_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = tmp_path / "ws"
    clip = _make_file(ws / "clips" / "c1.mp4")
    still = _make_file(ws / "stills" / "hero.png")
    output = tmp_path / "out" / "final.mp4"

    inputs = {
        "edit_decisions": {
            "cuts": [
                {"id": "c1", "source": str(clip), "in_seconds": 0, "out_seconds": 2},
                {"id": "c2", "source": str(still), "in_seconds": 0, "out_seconds": 2},
            ],
        },
        "output_path": str(output),
    }

    _, props, cmd = _invoke_remotion_render(
        VideoCompose(), inputs, tmp_path, monkeypatch
    )

    assert cmd[cmd.index("--public-dir") + 1] == str(ws)
    sources = [c["source"] for c in props["cuts"]]
    assert sources == ["clips/c1.mp4", "stills/hero.png"]


def test_single_asset_uses_parent_as_public_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = tmp_path / "ws"
    clip = _make_file(ws / "only.mp4")
    output = tmp_path / "out" / "final.mp4"

    inputs = {
        "edit_decisions": {
            "cuts": [
                {"id": "c1", "source": str(clip), "in_seconds": 0, "out_seconds": 2},
            ],
        },
        "output_path": str(output),
    }

    _, props, cmd = _invoke_remotion_render(
        VideoCompose(), inputs, tmp_path, monkeypatch
    )

    # commonpath of a single path returns the file itself; the renderer
    # must fall back to its parent so the CLI serves a directory.
    assert cmd[cmd.index("--public-dir") + 1] == str(ws)
    assert props["cuts"][0]["source"] == "only.mp4"


def test_url_sources_are_not_rewritten_or_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "out" / "final.mp4"
    inputs = {
        "edit_decisions": {
            "cuts": [
                {
                    "id": "c1",
                    "source": "https://example.com/remote.mp4",
                    "in_seconds": 0,
                    "out_seconds": 2,
                },
            ],
        },
        "output_path": str(output),
    }

    _, props, cmd = _invoke_remotion_render(
        VideoCompose(), inputs, tmp_path, monkeypatch
    )

    # No local assets collected -> no --public-dir override.
    assert "--public-dir" not in cmd
    # URL preserved verbatim.
    assert props["cuts"][0]["source"] == "https://example.com/remote.mp4"


def test_nonexistent_local_path_is_left_raw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "out" / "final.mp4"
    ghost = str(tmp_path / "does-not-exist.mp4")
    inputs = {
        "edit_decisions": {
            "cuts": [
                {"id": "c1", "source": ghost, "in_seconds": 0, "out_seconds": 2},
            ],
        },
        "output_path": str(output),
    }

    _, props, cmd = _invoke_remotion_render(
        VideoCompose(), inputs, tmp_path, monkeypatch
    )

    # Ghost paths aren't collected, so no --public-dir is set and the
    # source is left as-is for Remotion to 404 on. Better to fail loudly
    # at the renderer than to silently rewrite a path that doesn't exist.
    assert "--public-dir" not in cmd
    assert props["cuts"][0]["source"] == ghost


def test_mixed_roots_fall_back_without_public_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """os.path.commonpath raises ValueError when paths share no common
    ancestor (real-world trigger: Windows C:\\ vs D:\\, or an
    asset_manifest referencing multiple workspaces). The renderer must
    swallow the exception rather than crash, and skip --public-dir so the
    existing fallback behavior takes over."""
    import os as _os

    real_commonpath = _os.path.commonpath
    call_count = {"n": 0}

    def raising_commonpath(paths: list[str]) -> str:
        call_count["n"] += 1
        raise ValueError("Paths don't have the same drive")

    monkeypatch.setattr(_os.path, "commonpath", raising_commonpath)

    ws = tmp_path / "ws"
    clip = _make_file(ws / "a.mp4")
    output = tmp_path / "out" / "final.mp4"

    inputs = {
        "edit_decisions": {
            "cuts": [
                {"id": "c1", "source": str(clip), "in_seconds": 0, "out_seconds": 2},
            ],
        },
        "output_path": str(output),
    }

    result, _, cmd = _invoke_remotion_render(
        VideoCompose(), inputs, tmp_path, monkeypatch
    )

    assert call_count["n"] == 1
    assert result.success, result.error
    # Renderer gracefully skipped --public-dir rather than crashing.
    assert "--public-dir" not in cmd

    # Restore in case pytest fixture ordering surprises us later.
    monkeypatch.setattr(_os.path, "commonpath", real_commonpath)


def test_called_process_error_surfaces_stderr_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The old `except Exception` branch collapsed every render failure
    into 'returned non-zero exit status 1' with no diagnostic. The new
    explicit CalledProcessError branch must surface the stderr tail so
    asset-404s and prop errors are actually visible."""
    ws = tmp_path / "ws"
    clip = _make_file(ws / "a.mp4")
    output = tmp_path / "out" / "final.mp4"
    inputs = {
        "edit_decisions": {
            "cuts": [
                {"id": "c1", "source": str(clip), "in_seconds": 0, "out_seconds": 2},
            ],
        },
        "output_path": str(output),
    }

    stderr_msg = "Error: 404 Not Found for asset clips/missing.mp4"
    result, _, _ = _invoke_remotion_render(
        VideoCompose(),
        inputs,
        tmp_path,
        monkeypatch,
        stderr=stderr_msg,
        raise_called_process_error=True,
    )

    assert not result.success
    assert "Remotion render failed" in result.error
    assert "404 Not Found for asset clips/missing.mp4" in result.error
