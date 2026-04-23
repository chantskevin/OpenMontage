"""Unit tests for apiyi_gpt_image — the prompt-building, URL-extraction,
and aspect-crop helpers that don't need the network.

Live-API behavior is covered by manual exercise; these tests lock the
behavior the model never touches: how we encode aspect-ratio hints,
how we pull URLs out of the OpenAI-shaped response, and how we
center-crop on aspect drift.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from tools.graphics.apiyi_gpt_image import (
    ApiyiGptImage,
    ASPECT_TOLERANCE,
    _build_prompt,
    _ensure_aspect_ratio,
    _extract_image_url,
    _measure_aspect_drift,
    _parse_aspect,
)


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def test_tool_metadata() -> None:
    t = ApiyiGptImage()
    assert t.name == "apiyi_gpt_image"
    # Distinct provider key disambiguates from apiyi_flash_image (which
    # also uses the APIYI gateway). The selector's tool_by_provider dict
    # keeps exactly one entry per provider key, so colliding keys make
    # IMAGE_GEN_PROVIDER=apiyi pick whichever tool was discovered first
    # (nondeterministic). Mirrors the apiyi_seedream_image convention.
    assert t.provider == "apiyi_gpt"
    assert t.capability == "image_generation"
    schema = t.input_schema["properties"]
    # Aspect-ratio enum matches the hint table — adding a ratio without
    # a hint would produce a generic prompt the model frequently
    # ignores.
    enum = schema["aspect_ratio"]["enum"]
    assert set(enum) == {"1:1", "3:2", "4:3", "16:9", "9:16", "21:9"}


# ---------------------------------------------------------------------------
# _build_prompt — light + strong hint encoding
# ---------------------------------------------------------------------------


def test_build_prompt_known_ratio_light() -> None:
    out = _build_prompt("a forest", "16:9", strong=False)
    # Light hint prepends a short descriptor.
    assert out.startswith("16:9 cinematic widescreen")
    assert "a forest" in out
    # Light hint has no STRICT shouting.
    assert "STRICT" not in out


def test_build_prompt_known_ratio_strong() -> None:
    out = _build_prompt("a forest", "16:9", strong=True)
    # Strong hint shouts STRICT and names the wrong outcome explicitly.
    assert "STRICT" in out
    assert "NOT square" in out
    assert "a forest" in out


def test_build_prompt_unknown_ratio_landscape() -> None:
    out = _build_prompt("a beach", "5:3", strong=False)
    assert "5:3" in out
    assert "horizontal landscape" in out
    assert "a beach" in out


def test_build_prompt_unknown_ratio_portrait_strong() -> None:
    out = _build_prompt("a totem", "2:5", strong=True)
    assert "STRICT" in out
    assert "vertical portrait" in out
    assert "MUCH TALLER" in out
    assert "a totem" in out


def test_build_prompt_no_aspect_returns_prompt_unchanged() -> None:
    assert _build_prompt("just a thing", None, strong=False) == "just a thing"
    assert _build_prompt("just a thing", "", strong=True) == "just a thing"


def test_build_prompt_invalid_ratio_returns_unchanged() -> None:
    # An unparseable ratio shouldn't prepend garbage — the model will
    # do its best with the bare prompt.
    assert _build_prompt("scene", "not-a-ratio", strong=False) == "scene"


# ---------------------------------------------------------------------------
# _extract_image_url — markdown link, bare URL, failure
# ---------------------------------------------------------------------------


def test_extract_image_url_markdown() -> None:
    body = "Here is your image:\n\n![generated](https://cdn.example.com/abc.png)\n\nEnjoy."
    assert _extract_image_url(body) == "https://cdn.example.com/abc.png"


def test_extract_image_url_bare_url() -> None:
    body = "Image ready: https://cdn.example.com/abc.jpeg please download."
    assert _extract_image_url(body) == "https://cdn.example.com/abc.jpeg"


def test_extract_image_url_prefers_markdown_over_bare() -> None:
    """Markdown link wins — it's the structured form the model emits
    when it's behaving."""
    body = (
        "Background context with https://example.com/decoy.png url.\n"
        "![real](https://cdn.example.com/real.png)"
    )
    assert _extract_image_url(body) == "https://cdn.example.com/decoy.png" or \
           _extract_image_url(body) == "https://cdn.example.com/real.png"
    # The markdown matcher runs first; verify it returned the markdown URL
    # if present.
    assert _extract_image_url(body) == "https://cdn.example.com/real.png"


def test_extract_image_url_no_match_raises() -> None:
    with pytest.raises(RuntimeError) as exc:
        _extract_image_url("Sorry, the model says no.")
    assert "No image URL" in str(exc.value)


# ---------------------------------------------------------------------------
# _parse_aspect + _measure_aspect_drift
# ---------------------------------------------------------------------------


def test_parse_aspect_valid() -> None:
    assert _parse_aspect("16:9") == pytest.approx(16 / 9)
    assert _parse_aspect("1:1") == 1.0


def test_parse_aspect_invalid() -> None:
    assert _parse_aspect("not-a-ratio") is None
    assert _parse_aspect("0:9") is None
    assert _parse_aspect("16:0") is None


def _png_bytes(width: int, height: int, color=(128, 128, 128)) -> bytes:
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_measure_aspect_drift_zero_when_exact() -> None:
    # 1920x1080 is exactly 16:9.
    drift = _measure_aspect_drift(_png_bytes(1920, 1080), "16:9")
    assert drift is not None
    assert drift < 1e-6


def test_measure_aspect_drift_high_when_square_returned_for_widescreen() -> None:
    # The exact failure mode the strong-hint retry exists for: model
    # returned a square image when we asked for 16:9.
    drift = _measure_aspect_drift(_png_bytes(1024, 1024), "16:9")
    assert drift is not None
    # Square returned for 16:9 → drift ~= |1.0 - 1.778| / 1.778 ≈ 0.4375.
    assert drift > ASPECT_TOLERANCE
    assert drift > 0.4


def test_measure_aspect_drift_returns_none_on_invalid_ratio() -> None:
    assert _measure_aspect_drift(_png_bytes(100, 100), "not-a-ratio") is None


def test_measure_aspect_drift_returns_none_on_unreadable_bytes() -> None:
    assert _measure_aspect_drift(b"not an image", "16:9") is None


# ---------------------------------------------------------------------------
# _ensure_aspect_ratio — center-crop fallback
# ---------------------------------------------------------------------------


def test_ensure_aspect_ratio_passthrough_when_within_tolerance() -> None:
    # Already 16:9 — no crop, identical bytes.
    src = _png_bytes(1920, 1080)
    out = _ensure_aspect_ratio(src, "16:9")
    assert out == src


def test_ensure_aspect_ratio_crops_too_wide() -> None:
    # 2000x500 is 4:1 — much wider than 16:9 (1.778). Crop horizontally.
    src = _png_bytes(2000, 500)
    out = _ensure_aspect_ratio(src, "16:9")
    with Image.open(io.BytesIO(out)) as img:
        w, h = img.size
    # Height preserved, width cropped to ~16/9 * 500 ≈ 889.
    assert h == 500
    assert abs(w / h - 16 / 9) < 0.01


def test_ensure_aspect_ratio_crops_too_tall() -> None:
    # 500x2000 — much taller than 9:16 (0.5625). Crop vertically.
    src = _png_bytes(500, 2000)
    out = _ensure_aspect_ratio(src, "9:16")
    with Image.open(io.BytesIO(out)) as img:
        w, h = img.size
    assert w == 500
    assert abs(w / h - 9 / 16) < 0.01


def test_ensure_aspect_ratio_crops_square_to_widescreen() -> None:
    # The model-ignored-hint case: 1024x1024 returned for 16:9 request.
    # Square is "too tall" relative to 16:9, so width is preserved and
    # height is cropped down — 1024 / (16/9) ≈ 576.
    src = _png_bytes(1024, 1024)
    out = _ensure_aspect_ratio(src, "16:9")
    with Image.open(io.BytesIO(out)) as img:
        w, h = img.size
    assert w == 1024
    assert abs(h - 576) <= 2
    assert abs(w / h - 16 / 9) < 0.01


def test_ensure_aspect_ratio_no_aspect_returns_input() -> None:
    src = _png_bytes(123, 456)
    assert _ensure_aspect_ratio(src, None) == src


def test_ensure_aspect_ratio_invalid_aspect_returns_input() -> None:
    src = _png_bytes(123, 456)
    assert _ensure_aspect_ratio(src, "garbage") == src
