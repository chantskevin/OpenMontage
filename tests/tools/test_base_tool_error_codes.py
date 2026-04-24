"""Tests for the ErrorCode enum + ToolResult.error_code field.

Closed-vocabulary error codes let HTTP dispatchers branch on
structural categories instead of regex-matching free text. Tools
populate `error_code` to opt in; legacy tools default to
`ErrorCode.LEGACY` so callers always have something to switch on.
"""

from __future__ import annotations

import pytest

from tools.base_tool import (
    BaseTool,
    Determinism,
    ErrorCode,
    ExecutionMode,
    ResourceProfile,
    ToolResult,
    ToolStability,
    ToolStatus,
    ToolTier,
)


# ---------------------------------------------------------------------------
# ErrorCode enum shape
# ---------------------------------------------------------------------------


def test_error_code_enum_has_expected_codes() -> None:
    """The vocabulary is closed and tight (~12 codes). Locks the
    promise that callers can rely on this set being stable."""
    expected = {
        "invalid_input",
        "missing_dependency",
        "provider_unavailable",
        "provider_rate_limited",
        "provider_timeout",
        "provider_failed",
        "asset_missing",
        "render_failed",
        "output_invalid",
        "governance_violation",
        "internal_error",
        "legacy",
    }
    actual = {code.value for code in ErrorCode}
    assert actual == expected, (
        f"ErrorCode vocabulary changed unexpectedly. Added: "
        f"{actual - expected}. Removed: {expected - actual}."
    )


def test_error_code_values_are_lowercase_snake_case() -> None:
    """Convention: lowercase_snake. HTTP dispatchers serialize these
    as JSON strings; mixing case would break naive equality checks."""
    for code in ErrorCode:
        assert code.value == code.value.lower()
        assert " " not in code.value


# ---------------------------------------------------------------------------
# ToolResult.error_code field defaults
# ---------------------------------------------------------------------------


def test_failed_result_without_explicit_code_defaults_to_legacy() -> None:
    """Existing tools that return ToolResult(success=False, error=...)
    without setting error_code get LEGACY automatically. Migration
    is incremental — callers can switch on code regardless of
    whether the tool has migrated yet."""
    result = ToolResult(success=False, error="something went wrong")
    assert result.error_code == ErrorCode.LEGACY


def test_failed_result_with_explicit_code_keeps_it() -> None:
    """Explicitly-set codes are preserved through the post-init
    default. The default only fires when error_code is None."""
    result = ToolResult(
        success=False,
        error_code=ErrorCode.PROVIDER_RATE_LIMITED,
        error="429 Too Many Requests",
    )
    assert result.error_code == ErrorCode.PROVIDER_RATE_LIMITED


def test_successful_result_has_no_error_code() -> None:
    """Success results don't need a code. None means "no error";
    LEGACY would be confusing on a happy path."""
    result = ToolResult(success=True, data={"x": 1})
    assert result.error_code is None


# ---------------------------------------------------------------------------
# Schema validator populates INVALID_INPUT
# ---------------------------------------------------------------------------


class _ToolWithSchema(BaseTool):
    name = "test_error_codes"
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
        "properties": {"operation": {"type": "string"}},
    }

    def execute(self, inputs):
        return ToolResult(success=True, data={})

    def get_status(self) -> ToolStatus:
        return ToolStatus.AVAILABLE


def test_execute_validated_returns_invalid_input_code() -> None:
    """The validator was the first internal user of the new code
    enum. Verify it tags failures with INVALID_INPUT, not LEGACY."""
    result = _ToolWithSchema().execute_validated({})
    assert not result.success
    assert result.error_code == ErrorCode.INVALID_INPUT


# ---------------------------------------------------------------------------
# HTTP dispatcher integration shape (illustrative)
# ---------------------------------------------------------------------------


def test_error_code_enables_http_status_routing() -> None:
    """The whole point of the enum: HTTP layer can map codes to
    statuses without parsing error strings. Sketches the routing
    table a dispatcher might use."""
    code_to_http = {
        ErrorCode.INVALID_INPUT: 400,
        ErrorCode.ASSET_MISSING: 404,
        ErrorCode.PROVIDER_RATE_LIMITED: 429,
        ErrorCode.GOVERNANCE_VIOLATION: 422,
        ErrorCode.MISSING_DEPENDENCY: 503,
        ErrorCode.PROVIDER_UNAVAILABLE: 503,
        ErrorCode.PROVIDER_TIMEOUT: 504,
        ErrorCode.PROVIDER_FAILED: 502,
        ErrorCode.RENDER_FAILED: 500,
        ErrorCode.OUTPUT_INVALID: 500,
        ErrorCode.INTERNAL_ERROR: 500,
        ErrorCode.LEGACY: 500,
    }
    # Every defined code has a mapping. If a new code lands without
    # this table updating, the test fires.
    for code in ErrorCode:
        assert code in code_to_http, (
            f"ErrorCode.{code.name} added without HTTP routing entry"
        )
