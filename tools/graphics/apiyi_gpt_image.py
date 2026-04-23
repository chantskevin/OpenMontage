"""GPT-Image-2-All via APIYI gateway.

APIYI exposes OpenAI's gpt-image-2-all model through an OpenAI-compatible
chat-completions endpoint. The model has no native aspect_ratio / size
parameter — dimensions must be embedded in the prompt. This tool uses a
two-pass strategy: the first attempt uses a short hint, and if the
returned image's aspect drifts outside tolerance the second attempt
uses a much stronger hint. A final center-crop guarantees the output
matches the requested ratio.

Multimodal image editing is supported via OpenAI-style `image_url`
content parts carrying base64 data URIs.
"""

from __future__ import annotations

import base64
import io
import mimetypes
import os
import re
import time
from pathlib import Path
from typing import Any

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)

DEFAULT_BASE_URL = "https://api.apiyi.com"
ENDPOINT = "/v1/chat/completions"
MODEL = "gpt-image-2-all"
TIMEOUT_S = 300
ASPECT_TOLERANCE = 0.05

_MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*\]\((https?://[^)\s]+)\)")
_BARE_URL = re.compile(r"https?://[^\s)]+\.(?:png|jpe?g|webp)", re.IGNORECASE)

_ASPECT_HINTS: dict[str, str] = {
    "1:1": "1:1 square format",
    "16:9": "16:9 cinematic widescreen",
    "9:16": "9:16 vertical portrait",
    "21:9": "21:9 ultra-wide cinematic banner",
    "4:3": "4:3 standard photo format",
    "3:2": "3:2 classic photo format",
}

_STRONG_ASPECT_HINTS: dict[str, str] = {
    "1:1": "STRICT 1:1 square image, equal width and height",
    "16:9": "STRICT 16:9 cinematic widescreen image, landscape orientation, MUCH WIDER than tall, NOT square",
    "9:16": "STRICT 9:16 vertical portrait image, portrait orientation, MUCH TALLER than wide, NOT square",
    "21:9": "STRICT 21:9 ultra-wide cinematic banner, landscape orientation, MUCH WIDER than tall, NOT square",
    "4:3": "STRICT 4:3 horizontal landscape photo, landscape orientation, WIDER than tall, NOT square",
    "3:2": "STRICT 3:2 horizontal landscape photo, landscape orientation, WIDER than tall, NOT square",
}


class ApiyiGptImage(BaseTool):
    name = "apiyi_gpt_image"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "image_generation"
    provider = "apiyi"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API

    dependencies = []
    install_instructions = (
        "Set APIYI_API_KEY to your APIYI API key.\n"
        "  Optionally set APIYI_BASE_URL (defaults to https://api.apiyi.com)."
    )
    agent_skills = ["flux-best-practices"]

    capabilities = [
        "text_to_image",
        "image_to_image",
        "image_editing",
        "multi_reference",
    ]
    supports = {
        "negative_prompt": False,
        "seed": False,
        "aspect_ratio": True,
        "reference_image": True,
        "multiple_reference_images": True,
    }
    best_for = [
        "OpenAI gpt-image-2 quality via APIYI gateway",
        "multimodal image editing with reference images",
        "prompt-driven aspect ratios with center-crop guarantee",
    ]
    not_good_for = [
        "negative prompt control (not supported)",
        "exact dimension control (model picks its own ~1-2 MP budget)",
        "offline generation",
    ]
    fallback_tools = ["apiyi_flash_image", "apiyi_seedream_image", "flux_image"]

    input_schema = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {"type": "string", "description": "Image generation / editing prompt"},
            "aspect_ratio": {
                "type": "string",
                "enum": ["1:1", "3:2", "4:3", "16:9", "9:16", "21:9"],
                "default": "1:1",
                "description": (
                    "Desired aspect ratio. Embedded in the prompt as a hint; "
                    "drifted outputs are auto-cropped to match."
                ),
            },
            "image_url": {
                "type": "string",
                "description": "Single source image URL for edit mode",
            },
            "image_path": {
                "type": "string",
                "description": "Single local source image path for edit mode",
            },
            "image_urls": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Multiple reference image URLs for multi-image editing",
            },
            "image_paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Multiple local reference image paths for multi-image editing",
            },
            "output_path": {"type": "string"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=512, vram_mb=0, disk_mb=100, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=2, retryable_errors=["rate_limit", "timeout"])
    idempotency_key_fields = ["prompt", "aspect_ratio"]
    side_effects = ["writes image file to output_path", "calls APIYI API"]
    user_visible_verification = ["Inspect generated image for relevance and quality"]

    def _get_api_key(self) -> str | None:
        return os.environ.get("APIYI_API_KEY")

    def _get_base_url(self) -> str:
        return os.environ.get("APIYI_BASE_URL", DEFAULT_BASE_URL)

    def get_status(self) -> ToolStatus:
        if self._get_api_key():
            return ToolStatus.AVAILABLE
        return ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        return 0.04

    @staticmethod
    def _read_image(path_value: str | None, url_value: str | None) -> tuple[bytes, str] | None:
        if path_value:
            path = Path(path_value)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
            mime, _ = mimetypes.guess_type(path.name)
            return path.read_bytes(), mime or "image/png"

        if url_value:
            import requests

            resp = requests.get(url_value, timeout=60)
            resp.raise_for_status()
            mime = resp.headers.get("Content-Type", "image/png").split(";")[0]
            return resp.content, mime

        return None

    @classmethod
    def _collect_input_images(cls, inputs: dict[str, Any]) -> list[tuple[bytes, str]]:
        images: list[tuple[bytes, str]] = []

        primary = cls._read_image(inputs.get("image_path"), inputs.get("image_url"))
        if primary is not None:
            images.append(primary)

        for path in inputs.get("image_paths") or []:
            img = cls._read_image(path, None)
            if img is not None:
                images.append(img)

        for url in inputs.get("image_urls") or []:
            img = cls._read_image(None, url)
            if img is not None:
                images.append(img)

        return images

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        api_key = self._get_api_key()
        if not api_key:
            return ToolResult(
                success=False,
                error="APIYI_API_KEY not set. " + self.install_instructions,
            )

        start = time.time()
        prompt = inputs["prompt"]
        aspect_ratio = inputs.get("aspect_ratio", "1:1")

        try:
            input_images = self._collect_input_images(inputs)
        except FileNotFoundError as e:
            return ToolResult(success=False, error=str(e))
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to load input image: {e}")

        try:
            image_bytes, total_input_tokens, total_output_tokens, retried = (
                self._generate_with_drift_retry(
                    prompt, aspect_ratio, input_images, api_key
                )
            )
        except Exception as e:
            return ToolResult(success=False, error=f"APIYI gpt-image-2 failed: {e}")

        try:
            image_bytes = _ensure_aspect_ratio(image_bytes, aspect_ratio)
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"APIYI gpt-image-2 produced output but center-crop to {aspect_ratio} failed: {e}",
            )

        output_path = Path(inputs.get("output_path", "apiyi_gpt_image.png"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(image_bytes)

        return ToolResult(
            success=True,
            data={
                "provider": self.provider,
                "model": MODEL,
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "output": str(output_path),
                "reference_image_count": len(input_images),
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "aspect_drift_retried": retried,
            },
            artifacts=[str(output_path)],
            cost_usd=self.estimate_cost(inputs),
            duration_seconds=round(time.time() - start, 2),
            model=MODEL,
        )

    def _generate_with_drift_retry(
        self,
        prompt: str,
        aspect_ratio: str,
        input_images: list[tuple[bytes, str]],
        api_key: str,
    ) -> tuple[bytes, int, int, bool]:
        """Two-pass: light aspect hint, retry with strong hint if drift > tolerance."""
        first_bytes, in_t, out_t = self._call_api(
            _build_prompt(prompt, aspect_ratio, strong=False),
            input_images,
            api_key,
        )

        drift = _measure_aspect_drift(first_bytes, aspect_ratio)
        if drift is None or drift <= ASPECT_TOLERANCE:
            return first_bytes, in_t, out_t, False

        try:
            retry_bytes, retry_in_t, retry_out_t = self._call_api(
                _build_prompt(prompt, aspect_ratio, strong=True),
                input_images,
                api_key,
            )
            return retry_bytes, in_t + retry_in_t, out_t + retry_out_t, True
        except Exception:
            # Fall back to first attempt; ensure_aspect_ratio will crop it.
            return first_bytes, in_t, out_t, False

    def _call_api(
        self,
        prompt_text: str,
        input_images: list[tuple[bytes, str]],
        api_key: str,
    ) -> tuple[bytes, int, int]:
        """One API call. Returns (image_bytes, input_tokens, output_tokens)."""
        import requests

        base_url = self._get_base_url()

        if input_images:
            content: Any = [{"type": "text", "text": prompt_text}]
            for img_bytes, mime in input_images:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{base64.b64encode(img_bytes).decode('ascii')}"
                    },
                })
        else:
            content = prompt_text

        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": content}],
            "stream": False,
        }

        response = requests.post(
            f"{base_url}{ENDPOINT}",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=TIMEOUT_S,
        )

        if not response.ok:
            raise RuntimeError(
                f"APIYI gpt-image-2 {response.status_code}: {response.text[:500]}"
            )

        body = response.json()
        message = (body.get("choices") or [{}])[0].get("message", {}).get("content")
        if not isinstance(message, str):
            raise RuntimeError(
                f"No message content in APIYI gpt-image-2 response: {str(body)[:500]}"
            )

        image_url = _extract_image_url(message)
        img_resp = requests.get(image_url, timeout=TIMEOUT_S)
        if not img_resp.ok:
            raise RuntimeError(
                f"Failed to download gpt-image-2 image from {image_url}: {img_resp.status_code}"
            )

        usage = body.get("usage") or {}
        return (
            img_resp.content,
            int(usage.get("prompt_tokens") or 0),
            int(usage.get("completion_tokens") or 0),
        )


def _build_prompt(prompt: str, aspect_ratio: str | None, *, strong: bool) -> str:
    if not aspect_ratio:
        return prompt
    table = _STRONG_ASPECT_HINTS if strong else _ASPECT_HINTS
    known = table.get(aspect_ratio)
    if known:
        return f"{known}, {prompt}"
    parts = aspect_ratio.split(":")
    try:
        w, h = int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        return prompt
    if w == h:
        prefix = "STRICT " if strong else ""
        return f"{prefix}{aspect_ratio} square format, {prompt}"
    suffix = (
        " (MUCH WIDER than tall, NOT square)"
        if strong and w > h
        else " (MUCH TALLER than wide, NOT square)" if strong else ""
    )
    orientation = "horizontal landscape" if w > h else "vertical portrait"
    prefix = "STRICT " if strong else ""
    return f"{prefix}{aspect_ratio} {orientation}{suffix}, {prompt}"


def _extract_image_url(content: str) -> str:
    match = _MARKDOWN_IMAGE.search(content)
    if match:
        return match.group(1)
    bare = _BARE_URL.search(content)
    if bare:
        return bare.group(0)
    raise RuntimeError(f"No image URL in gpt-image-2 response: {content[:200]}")


def _parse_aspect(aspect_ratio: str) -> float | None:
    parts = aspect_ratio.split(":")
    try:
        w, h = float(parts[0]), float(parts[1])
    except (ValueError, IndexError):
        return None
    if w <= 0 or h <= 0:
        return None
    return w / h


def _measure_aspect_drift(image_bytes: bytes, aspect_ratio: str) -> float | None:
    target = _parse_aspect(aspect_ratio)
    if target is None:
        return None
    try:
        from PIL import Image
        with Image.open(io.BytesIO(image_bytes)) as img:
            w, h = img.size
    except Exception:
        return None
    if not w or not h:
        return None
    return abs(w / h - target) / target


def _ensure_aspect_ratio(image_bytes: bytes, aspect_ratio: str | None) -> bytes:
    """Center-crop image to target aspect ratio if drift exceeds tolerance."""
    if not aspect_ratio:
        return image_bytes
    target = _parse_aspect(aspect_ratio)
    if target is None:
        return image_bytes

    from PIL import Image

    with Image.open(io.BytesIO(image_bytes)) as img:
        w, h = img.size
        if not w or not h:
            return image_bytes

        actual = w / h
        drift = abs(actual - target) / target
        if drift <= ASPECT_TOLERANCE:
            return image_bytes

        if actual > target:
            crop_h = h
            crop_w = max(1, round(h * target))
        else:
            crop_w = w
            crop_h = max(1, round(w / target))
        left = (w - crop_w) // 2
        top = (h - crop_h) // 2

        cropped = img.crop((left, top, left + crop_w, top + crop_h))
        buf = io.BytesIO()
        # Preserve format when sensible; default to PNG.
        out_format = (img.format or "PNG").upper()
        if out_format not in ("PNG", "JPEG", "WEBP"):
            out_format = "PNG"
        cropped.save(buf, format=out_format)
        return buf.getvalue()
