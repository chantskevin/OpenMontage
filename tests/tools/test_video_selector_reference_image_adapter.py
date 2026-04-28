"""Selector-level tests for the reference-image input adapter.

`video_selector` accepts `reference_image_path` and `reference_image_url`
as canonical caller-facing keys, but each underlying provider names the
field differently in its own schema. The selector adapts the input to
whatever shape the chosen provider declares before dispatching.

The order of preference (per branch) is:

  path branch: reference_image_path (native) | image_path | image_url (FAL upload)
  url  branch: reference_image_url  (native) | image_url (rename)   | image_path (skip)

These tests pin the adapter to that shape using stub provider tools,
bypassing the registry. The key bug they prevent is calling
`upload_image_fal` (which requires FAL_KEY) for providers that read local
bytes themselves — apiyi_veo_video being the canonical example. The
provider-native `image_path` branch must win over the FAL fallback.
"""

from __future__ import annotations

from typing import Any

import pytest

from tools.base_tool import BaseTool, ToolResult, ToolRuntime, ToolStability, ToolStatus, ToolTier
from tools.video import video_selector as video_selector_module
from tools.video.video_selector import VideoSelector


# ---------------------------------------------------------------------------
# Stub provider — declares only the schema keys we want to exercise. The
# selector reads `input_schema.properties` to decide which key to populate,
# so this is sufficient to test the adapter without a real video backend.
# ---------------------------------------------------------------------------


class _StubProvider(BaseTool):
    name = "stub_video"
    version = "0.0.0"
    tier = ToolTier.GENERATE
    capability = "video_generation"
    provider = "stub"
    stability = ToolStability.BETA
    runtime = ToolRuntime.HYBRID
    capabilities = ["image_to_video"]
    supports = {"image_to_video": True}
    best_for = ["adapter tests"]

    def __init__(self, schema_keys: list[str]) -> None:
        super().__init__()
        props: dict[str, Any] = {"prompt": {"type": "string"}, "output_path": {"type": "string"}}
        for key in schema_keys:
            props[key] = {"type": "string"}
        self.input_schema = {
            "type": "object",
            "required": ["prompt"],
            "properties": props,
        }
        self.last_inputs: dict[str, Any] | None = None

    def get_status(self) -> ToolStatus:
        return ToolStatus.AVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        return 0.0

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        return 0.0

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        self.last_inputs = dict(inputs)
        return ToolResult(success=True, data={"output_path": inputs.get("output_path", "")})


def _selector_with(stub: _StubProvider, monkeypatch: pytest.MonkeyPatch) -> VideoSelector:
    """Bypass the registry and pin the selector to a single stub provider."""
    selector = VideoSelector()
    monkeypatch.setattr(selector, "_providers", lambda: [stub])
    return selector


# ---------------------------------------------------------------------------
# path branch
# ---------------------------------------------------------------------------


def test_path_branch_passes_through_when_provider_is_native(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provider declares `reference_image_path` natively → no adaptation.

    The selector must NOT rename to `image_path` or upload to FAL when the
    provider already speaks the canonical key.
    """
    stub = _StubProvider(schema_keys=["reference_image_path"])
    selector = _selector_with(stub, monkeypatch)

    result = selector.execute({
        "prompt": "test",
        "operation": "image_to_video",
        "reference_image_path": "/tmp/frame.png",
    })

    assert result.success
    assert stub.last_inputs is not None
    assert stub.last_inputs["reference_image_path"] == "/tmp/frame.png"
    assert "image_path" not in stub.last_inputs
    assert "image_url" not in stub.last_inputs


def test_path_branch_renames_to_image_path_when_provider_accepts_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provider declares `image_path` (e.g., apiyi_veo_video) → rename, no upload.

    This is the core regression check for #37: previously the selector
    fell straight through to FAL upload when the provider's schema also
    declared `image_url`. The provider-native `image_path` branch must
    win, even when `image_url` is also in the schema.
    """
    stub = _StubProvider(schema_keys=["image_path", "image_url"])
    selector = _selector_with(stub, monkeypatch)

    # Sentinel: if the upload path is reached, the test fails loudly
    # rather than silently calling out to fal.ai.
    def _explode(*_args: Any, **_kwargs: Any) -> str:
        raise AssertionError("upload_image_fal must not be called when provider accepts image_path")
    monkeypatch.setattr("tools.video._shared.upload_image_fal", _explode)

    result = selector.execute({
        "prompt": "test",
        "operation": "image_to_video",
        "reference_image_path": "/tmp/frame.png",
    })

    assert result.success
    assert stub.last_inputs is not None
    assert stub.last_inputs["image_path"] == "/tmp/frame.png"
    assert "image_url" not in stub.last_inputs


def test_path_branch_uploads_via_fal_only_when_provider_is_url_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provider declares only `image_url` (e.g., kling, runway) → FAL upload.

    FAL is the genuine last-resort path for url-only providers; this
    behavior is preserved exactly as it was before the fix.
    """
    stub = _StubProvider(schema_keys=["image_url"])
    selector = _selector_with(stub, monkeypatch)

    monkeypatch.setattr("tools.video._shared.upload_image_fal", lambda _p: "https://fal.example/abc.png")

    result = selector.execute({
        "prompt": "test",
        "operation": "image_to_video",
        "reference_image_path": "/tmp/frame.png",
    })

    assert result.success
    assert stub.last_inputs is not None
    assert stub.last_inputs["image_url"] == "https://fal.example/abc.png"
    assert "image_path" not in stub.last_inputs


def test_path_branch_surfaces_fal_error_when_url_only_and_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Url-only provider + FAL upload failure → structured error returned.

    Preserves the existing surface area: callers see the same error
    shape they did before #37, just only when FAL is genuinely the
    last resort.
    """
    stub = _StubProvider(schema_keys=["image_url"])
    selector = _selector_with(stub, monkeypatch)

    def _no_key(_p: str) -> str:
        raise RuntimeError("FAL_KEY or FAL_AI_API_KEY required for image upload")
    monkeypatch.setattr("tools.video._shared.upload_image_fal", _no_key)

    result = selector.execute({
        "prompt": "test",
        "operation": "image_to_video",
        "reference_image_path": "/tmp/frame.png",
    })

    assert not result.success
    assert "Failed to upload reference image" in (result.error or "")
    assert "FAL_KEY" in (result.error or "")


def test_path_branch_does_not_overwrite_explicit_image_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Caller-provided `image_path` wins; selector does not stomp it.

    A caller who knows the provider's native shape can populate the
    field directly and the selector must respect that.
    """
    stub = _StubProvider(schema_keys=["image_path", "image_url"])
    selector = _selector_with(stub, monkeypatch)

    result = selector.execute({
        "prompt": "test",
        "operation": "image_to_video",
        "reference_image_path": "/tmp/from_ref.png",
        "image_path": "/tmp/explicit.png",
    })

    assert result.success
    assert stub.last_inputs is not None
    assert stub.last_inputs["image_path"] == "/tmp/explicit.png"


# ---------------------------------------------------------------------------
# url branch
# ---------------------------------------------------------------------------


def test_url_branch_passes_through_when_provider_is_native(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubProvider(schema_keys=["reference_image_url"])
    selector = _selector_with(stub, monkeypatch)

    result = selector.execute({
        "prompt": "test",
        "operation": "image_to_video",
        "reference_image_url": "https://example.com/ref.png",
    })

    assert result.success
    assert stub.last_inputs is not None
    assert stub.last_inputs["reference_image_url"] == "https://example.com/ref.png"
    assert "image_url" not in stub.last_inputs


def test_url_branch_renames_to_image_url_for_apiyi_veo_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provider declares `image_url` but not `reference_image_url` → rename.

    apiyi_veo_video accepts the canonical hosted-URL key under the name
    `image_url`; the selector must adapt rather than dropping the URL
    silently.
    """
    stub = _StubProvider(schema_keys=["image_url", "image_path"])
    selector = _selector_with(stub, monkeypatch)

    result = selector.execute({
        "prompt": "test",
        "operation": "image_to_video",
        "reference_image_url": "https://example.com/ref.png",
    })

    assert result.success
    assert stub.last_inputs is not None
    assert stub.last_inputs["image_url"] == "https://example.com/ref.png"


def test_url_branch_does_not_overwrite_explicit_image_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit `image_url` wins over `reference_image_url` rename."""
    stub = _StubProvider(schema_keys=["image_url"])
    selector = _selector_with(stub, monkeypatch)

    result = selector.execute({
        "prompt": "test",
        "operation": "image_to_video",
        "reference_image_url": "https://example.com/ref.png",
        "image_url": "https://example.com/explicit.png",
    })

    assert result.success
    assert stub.last_inputs is not None
    assert stub.last_inputs["image_url"] == "https://example.com/explicit.png"


def test_url_branch_skips_when_provider_only_accepts_image_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """URL→path download is out of scope; provider surfaces its own error."""
    stub = _StubProvider(schema_keys=["image_path"])
    selector = _selector_with(stub, monkeypatch)

    result = selector.execute({
        "prompt": "test",
        "operation": "image_to_video",
        "reference_image_url": "https://example.com/ref.png",
    })

    assert result.success
    assert stub.last_inputs is not None
    assert "image_url" not in stub.last_inputs
    assert "image_path" not in stub.last_inputs


# ---------------------------------------------------------------------------
# both branches active
# ---------------------------------------------------------------------------


def test_path_and_url_branches_coexist_in_one_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """A caller may send both path and URL — both must adapt cleanly."""
    stub = _StubProvider(schema_keys=["image_path", "reference_image_url"])
    selector = _selector_with(stub, monkeypatch)

    result = selector.execute({
        "prompt": "test",
        "operation": "image_to_video",
        "reference_image_path": "/tmp/frame.png",
        "reference_image_url": "https://example.com/ref.png",
    })

    assert result.success
    assert stub.last_inputs is not None
    assert stub.last_inputs["image_path"] == "/tmp/frame.png"
    assert stub.last_inputs["reference_image_url"] == "https://example.com/ref.png"
