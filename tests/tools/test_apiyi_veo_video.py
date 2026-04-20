"""Unit tests for apiyi_veo_video.

Covers the two guardrails added on 2026-04-20 after a pipeline run silently
clobbered CWD:

1. `output_path` is required — missing/empty/non-string inputs fail loudly
   instead of defaulting to a relative path in the process CWD.
2. `APIYI_MOCK=1` mode generates a local ffmpeg test clip without calling
   the API, so end-to-end pipeline tests can exercise the full stack
   (selector → provider → compose → integrity gate) for zero cost.

These tests never touch the network — `APIYI_API_KEY` is unset in test 1
and APIYI_MOCK is set in tests 2 & 3.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tools.video.apiyi_veo_video import ApiyiVeoVideo


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def test_missing_output_path_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    # APIYI_MOCK off so we go down the real-API branch — but the required-path
    # check happens before any API interaction, so this still works offline.
    monkeypatch.delenv("APIYI_MOCK", raising=False)
    monkeypatch.setenv("APIYI_API_KEY", "fake-for-test")

    tool = ApiyiVeoVideo()
    result = tool.execute({"prompt": "a cat"})

    assert not result.success
    assert "output_path" in result.error.lower()
    assert "required" in result.error.lower()


def test_empty_string_output_path_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APIYI_MOCK", raising=False)
    monkeypatch.setenv("APIYI_API_KEY", "fake-for-test")

    tool = ApiyiVeoVideo()
    result = tool.execute({"prompt": "a cat", "output_path": ""})

    assert not result.success
    assert "output_path" in result.error.lower()


@pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg not on PATH")
def test_mock_mode_writes_landscape_clip_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("APIYI_MOCK", "1")
    out = tmp_path / "clip.mp4"

    tool = ApiyiVeoVideo()
    result = tool.execute({"prompt": "a wolf", "output_path": str(out)})

    assert result.success, result.error
    assert out.exists(), "mock should have written the output file"
    assert result.data["mock"] is True
    assert result.data["aspect_ratio"] == "16:9"
    assert result.cost_usd == 0.0

    # Verify the file is actually a landscape 1280x720 mp4.
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert probe.stdout.strip() == "1280,720"


@pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg not on PATH")
def test_mock_mode_respects_explicit_portrait_aspect(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("APIYI_MOCK", "1")
    out = tmp_path / "portrait.mp4"

    tool = ApiyiVeoVideo()
    result = tool.execute(
        {"prompt": "a cat", "output_path": str(out), "aspect_ratio": "9:16"}
    )

    assert result.success, result.error
    assert result.data["aspect_ratio"] == "9:16"
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert probe.stdout.strip() == "720,1280"
