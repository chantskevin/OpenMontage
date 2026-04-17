"""Gemini 3.1 Flash Image via APIYI gateway.

APIYI exposes Google's Gemini Flash Image model for text-to-image and
multi-reference image editing through the native Gemini API shape.
"""

from __future__ import annotations

import base64
import mimetypes
import os
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
MODEL = "gemini-3.1-flash-image-preview"
TIMEOUT_S = 300


class ApiyiFlashImage(BaseTool):
    name = "apiyi_flash_image"
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
        "style_transfer",
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
        "multimodal image editing with multiple reference images",
        "style transfer and iterative refinement",
        "Gemini 3.1 Flash Image via APIYI gateway",
    ]
    not_good_for = [
        "negative prompt control (not supported)",
        "offline generation",
    ]
    fallback_tools = ["apiyi_seedream_image", "google_imagen", "flux_image"]

    input_schema = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {"type": "string", "description": "Image generation / editing prompt"},
            "aspect_ratio": {
                "type": "string",
                "enum": ["1:1", "3:4", "4:3", "9:16", "16:9"],
                "default": "1:1",
                "description": "Desired aspect ratio. Gemini picks exact dimensions.",
            },
            "image_size": {
                "type": "string",
                "enum": ["1K", "2K", "4K"],
                "description": "Resolution tier. Omit to use provider default.",
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
    idempotency_key_fields = ["prompt", "aspect_ratio", "image_size"]
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
        """Return (bytes, mime_type) from path or URL, or None."""
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
        """Collect (bytes, mime) pairs from image_url/path + image_urls/paths."""
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

        import requests

        start = time.time()
        base_url = self._get_base_url()
        prompt = inputs["prompt"]
        aspect_ratio = inputs.get("aspect_ratio", "1:1")

        try:
            input_images = self._collect_input_images(inputs)
        except FileNotFoundError as e:
            return ToolResult(success=False, error=str(e))
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to load input image: {e}")

        # Build multimodal parts: prompt text + optional reference images
        parts: list[dict[str, Any]] = [{"text": prompt}]
        for img_bytes, mime in input_images:
            parts.append({
                "inline_data": {
                    "mime_type": mime,
                    "data": base64.b64encode(img_bytes).decode("ascii"),
                }
            })

        image_config: dict[str, str] = {"aspectRatio": aspect_ratio}
        if inputs.get("image_size"):
            image_config["image_size"] = inputs["image_size"]

        payload: dict[str, Any] = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "responseModalities": ["IMAGE"],
                "imageConfig": image_config,
            },
        }

        try:
            response = requests.post(
                f"{base_url}/v1beta/models/{MODEL}:generateContent",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=TIMEOUT_S,
            )

            if not response.ok:
                return ToolResult(
                    success=False,
                    error=f"APIYI Flash Image failed ({response.status_code}): {response.text[:500]}",
                )

            body = response.json()
            image_b64 = _extract_image_b64(body)
            if image_b64 is None:
                return ToolResult(
                    success=False,
                    error=f"No image in APIYI Flash Image response: {str(body)[:500]}",
                )

            image_bytes = base64.b64decode(image_b64)
            if not image_bytes:
                return ToolResult(success=False, error="APIYI Flash Image returned empty image data")

            output_path = Path(inputs.get("output_path", "apiyi_flash_image.png"))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(image_bytes)

        except Exception as e:
            return ToolResult(success=False, error=f"APIYI Flash Image failed: {e}")

        usage = body.get("usageMetadata") or {}

        return ToolResult(
            success=True,
            data={
                "provider": self.provider,
                "model": MODEL,
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "output": str(output_path),
                "reference_image_count": len(input_images),
                "input_tokens": usage.get("promptTokenCount", 0),
                "output_tokens": usage.get("candidatesTokenCount", 0),
            },
            artifacts=[str(output_path)],
            cost_usd=self.estimate_cost(inputs),
            duration_seconds=round(time.time() - start, 2),
            model=MODEL,
        )


def _extract_image_b64(body: dict[str, Any]) -> str | None:
    """Walk candidates → parts, returning first base64 image data.

    APIYI may return either camelCase (inlineData) or snake_case (inline_data).
    """
    for candidate in body.get("candidates") or []:
        for part in (candidate.get("content") or {}).get("parts") or []:
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                return inline["data"]
    return None
