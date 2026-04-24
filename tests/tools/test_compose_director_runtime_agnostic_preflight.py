"""Contract test for fork issue #35 — runtime-agnostic Remotion preflight.

The compose-director skill used to mandate a `python -c "..."` shell
snippet for the Remotion preflight check. LLMs in shell-less
runtimes (MCP, sub-loops, sandboxed hosts) couldn't execute it.
Their nearest-tool interpretation was "call video_compose," which
they fired with no args (`video_compose({})`) and doom-looped on
the resulting `'operation' is required` error for 15+ iterations.

This test asserts the skill now offers introspection paths for
both shell and shell-less runtimes, names the doom-loop anti-
pattern explicitly, and downgrades the preflight from MANDATORY to
optional (the tool's own error path is the load-bearing safety net).
"""

from __future__ import annotations

from pathlib import Path


SKILL_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "skills" / "pipelines" / "cinematic" / "compose-director.md"
)


def _read_skill() -> str:
    return SKILL_PATH.read_text(encoding="utf-8")


def test_skill_names_fork_issue_35() -> None:
    """The preflight rewrite must reference the issue number so a
    reader tracing the new shape can find the original failure."""
    body = _read_skill()
    assert "fork issue #35" in body, (
        "compose-director must reference fork issue #35 in the "
        "preflight section — that's the original failure shape."
    )


def test_skill_no_longer_marks_preflight_mandatory() -> None:
    """The original 'MANDATORY' tag was a major part of why LLMs
    doom-looped — they prioritized executing it over actually
    rendering. Preflight is best-effort now."""
    body = _read_skill()
    # Find the preflight block and verify it's not flagged MANDATORY.
    idx = body.lower().find("remotion preflight")
    assert idx >= 0
    nearby = body[idx:idx + 1500]
    assert "MANDATORY" not in nearby, (
        "The preflight should not be flagged MANDATORY. The doom-loop "
        "from #35 was driven by LLMs prioritizing the MANDATORY shell "
        "snippet over the actual render call."
    )


def test_skill_offers_shellless_runtime_path() -> None:
    """The whole point of the rewrite — runtime-agnostic
    introspection. Tool-dispatch / MCP / sub-loop runtimes need a
    non-shell path, otherwise they'll re-trigger #35."""
    body = _read_skill()
    lower = body.lower()
    # Some phrasing for the shell-less case.
    assert "tool-dispatch" in lower or "mcp" in lower or "no shell" in lower, (
        "skill must explicitly address shell-less runtimes; the bug "
        "was assuming every runtime has bash"
    )
    # The schema-inlined fallback ("read it from your prompt context").
    assert "input_schema" in body or "prompt context" in lower or "inlined" in lower


def test_skill_warns_against_empty_args_doom_loop() -> None:
    """The exact #35 anti-pattern (`video_compose({})` to "trigger
    preflight") must be named explicitly. Without naming it, the
    LLM will reinvent it from the shell snippet's surface form."""
    body = _read_skill()
    lower = body.lower()
    # Specific anti-pattern: don't call video_compose with empty args.
    assert "empty args" in lower or "video_compose({})" in body, (
        "skill should explicitly forbid the empty-args doom-loop "
        "pattern that was #35"
    )
    # Doom-loop / wedge term.
    assert "doom-loop" in lower or "doom loop" in lower


def test_skill_keeps_shell_snippet_for_cli_users() -> None:
    """Shell-equipped runtimes (dev CLI workflow) still benefit from
    the python -c snippet. The fix shouldn't drop it entirely —
    it should add the alternatives."""
    body = _read_skill()
    assert "python -c" in body
    assert "registry.get" in body


def test_skill_explains_preflight_is_belt_and_suspenders() -> None:
    """The skill should make clear that the tool's own ToolResult
    on a failed render is the load-bearing safety net. The preflight
    is optional; runtimes that can't introspect should skip it
    rather than wedge."""
    body = _read_skill()
    lower = body.lower()
    assert "belt-and-suspenders" in lower or "not load-bearing" in lower or "skip this step" in lower
