"""Tests for BaseTool.execute_validated — schema validation at the boundary.

The fork issue #25 class (KeyError on missing required field) is
catchable structurally: every tool declares input_schema with
required fields. A dispatcher that runs jsonschema.validate(inputs,
self.input_schema) BEFORE calling execute() turns those KeyErrors
into structured ToolResults without per-tool changes.

execute_validated() is the new boundary helper. It's opt-in (callers
choose execute() vs execute_validated()) so existing direct callers
aren't broken.
"""

from __future__ import annotations

from typing import Any

import pytest

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    ToolResult,
    ToolStability,
    ToolStatus,
    ToolTier,
)


# ---------------------------------------------------------------------------
# Test tools — minimal subclasses with various schema shapes
# ---------------------------------------------------------------------------


class _ToolWithRequiredField(BaseTool):
    name = "test_required_field"
    version = "0.1.0"
    tier = ToolTier.CORE
    capability = "test"
    provider = "test"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC
    resource_profile = ResourceProfile()

    input_schema = {
        "type": "object",
        "required": ["operation"],
        "properties": {
            "operation": {"type": "string"},
            "optional_count": {"type": "integer"},
        },
    }

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        return ToolResult(success=True, data={"got": inputs})

    def get_status(self) -> ToolStatus:
        return ToolStatus.AVAILABLE


class _ToolWithEnumField(BaseTool):
    name = "test_enum_field"
    version = "0.1.0"
    tier = ToolTier.CORE
    capability = "test"
    provider = "test"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC
    resource_profile = ResourceProfile()

    input_schema = {
        "type": "object",
        "required": ["mode"],
        "properties": {
            "mode": {"type": "string", "enum": ["a", "b", "c"]},
        },
    }

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        return ToolResult(success=True, data={"mode": inputs["mode"]})

    def get_status(self) -> ToolStatus:
        return ToolStatus.AVAILABLE


class _ToolNoSchema(BaseTool):
    name = "test_no_schema"
    version = "0.1.0"
    tier = ToolTier.CORE
    capability = "test"
    provider = "test"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC
    resource_profile = ResourceProfile()
    input_schema = {}  # tool with no schema — validation should skip

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        return ToolResult(success=True, data=dict(inputs))

    def get_status(self) -> ToolStatus:
        return ToolStatus.AVAILABLE


# ---------------------------------------------------------------------------
# Required-field validation
# ---------------------------------------------------------------------------


def test_execute_validated_rejects_missing_required_field() -> None:
    """The fork #25 class repro: caller forgets a required field.
    execute_validated() returns a ToolResult instead of letting the
    underlying execute() crash."""
    result = _ToolWithRequiredField().execute_validated({})
    assert not result.success
    err = result.error or ""
    assert "test_required_field" in err
    assert "operation" in err
    # Field name is structured: callers can branch on it without regex.
    assert "invalid input at" in err


def test_execute_validated_accepts_complete_inputs() -> None:
    """Happy path: required field present → validation passes →
    execute() runs normally."""
    result = _ToolWithRequiredField().execute_validated(
        {"operation": "do_thing"}
    )
    assert result.success
    assert result.data == {"got": {"operation": "do_thing"}}


def test_execute_validated_rejects_wrong_type() -> None:
    """jsonschema catches type mismatches too. operation must be
    string; passing an int gets a clean rejection."""
    result = _ToolWithRequiredField().execute_validated(
        {"operation": 123}
    )
    assert not result.success
    err = result.error or ""
    assert "operation" in err


# ---------------------------------------------------------------------------
# Enum validation
# ---------------------------------------------------------------------------


def test_execute_validated_rejects_value_outside_enum() -> None:
    """When the schema declares an enum, values outside it fail with a
    pointed error naming the field."""
    result = _ToolWithEnumField().execute_validated({"mode": "z"})
    assert not result.success
    err = result.error or ""
    assert "mode" in err


def test_execute_validated_accepts_value_in_enum() -> None:
    """Sanity: in-enum values pass."""
    result = _ToolWithEnumField().execute_validated({"mode": "a"})
    assert result.success


# ---------------------------------------------------------------------------
# No-schema tools — validation skips, doesn't break
# ---------------------------------------------------------------------------


def test_execute_validated_skips_when_no_schema() -> None:
    """Tools with empty input_schema (legacy or schemaless) skip
    validation rather than failing. The opt-in is per-tool: tools
    without a schema get the same behavior as direct execute()."""
    result = _ToolNoSchema().execute_validated({"anything": "goes"})
    assert result.success
    assert result.data == {"anything": "goes"}


# ---------------------------------------------------------------------------
# Direct execute() unchanged — no behavior change for existing callers
# ---------------------------------------------------------------------------


def test_direct_execute_still_raises_on_missing_field() -> None:
    """The contract for direct execute() is unchanged. Callers that
    invoke execute() directly (not execute_validated) get the
    underlying tool's behavior — which may include KeyError. This
    keeps existing callers and tests working without changes."""
    # _ToolWithRequiredField's execute() doesn't itself check for
    # required fields — it just stores them. Direct execute({}) should
    # still succeed (it doesn't dereference operation), proving execute
    # is unaffected by validation.
    result = _ToolWithRequiredField().execute({})
    assert result.success
    assert result.data == {"got": {}}


# ---------------------------------------------------------------------------
# Error format — structured for HTTP dispatchers
# ---------------------------------------------------------------------------


def test_validation_error_includes_tool_name() -> None:
    """HTTP dispatchers log this. Including the tool name lets ops
    grep validation failures by tool without correlating to which
    request hit which tool."""
    result = _ToolWithRequiredField().execute_validated({})
    assert "test_required_field" in (result.error or "")


def test_validation_error_includes_field_path() -> None:
    """The field path is a deterministic anchor callers can branch
    on, vs regex-matching free text. Pass the required field so the
    type violation on optional_count is what the validator reports."""
    result = _ToolWithRequiredField().execute_validated(
        {"operation": "do", "optional_count": "not_an_int"}
    )
    assert not result.success
    assert "optional_count" in (result.error or "")
