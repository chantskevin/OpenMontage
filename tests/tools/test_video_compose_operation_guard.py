"""Regression tests for fork issue #25.

video_compose.execute() used to dereference inputs["operation"]
directly, raising a bare KeyError when the field was missing. Every
other tool in the registry returns a structured ToolResult on
missing required inputs; this was the only one that bubbled a
Python traceback through the HTTP dispatcher.
"""

from __future__ import annotations

from tools.video.video_compose import VideoCompose


def test_execute_missing_operation_returns_structured_error() -> None:
    """Missing 'operation' must return ToolResult, not raise KeyError.
    Empty inputs are the tightest repro — common when an LLM forgets
    the required arg on its first attempt before retrying."""
    result = VideoCompose().execute({})
    assert not result.success
    err = result.error or ""
    assert "'operation' is required" in err
    # Error should enumerate the valid operations so the caller knows
    # what to retry with — bare "missing field" forces another round
    # trip to discover the schema.
    for op in ("compose", "render", "remotion_render", "burn_subtitles", "overlay", "encode"):
        assert op in err


def test_execute_empty_string_operation_returns_structured_error() -> None:
    """Empty-string operation hits the same fail-fast path. Truthy
    check (not just .get() != None) catches both None and ""."""
    result = VideoCompose().execute({"operation": ""})
    assert not result.success
    assert "'operation' is required" in (result.error or "")


def test_execute_unknown_operation_still_returns_structured_error() -> None:
    """Unknown operation names should also fail cleanly — there's
    already a branch for this but it lives below the missing-field
    guard, so ensure both paths return ToolResult, not raise."""
    result = VideoCompose().execute({"operation": "bogus"})
    assert not result.success
    assert "Unknown operation" in (result.error or "")
    assert "bogus" in (result.error or "")
