"""Seedream 5.0 image generation via APIYI gateway.

APIYI exposes ByteDance's Seedream 5.0 through an OpenAI-compatible
images endpoint. Strong for high-resolution (2K/3K) with single-image
reference conditioning.
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
MODEL = "seedream-5-0-260128"
TIMEOUT_S = 300

# Seedream pixel-dimension table.
# Seedream constraints:
#   Total pixels: [3,686,400 .. 10,404,496]
#   Aspect ratio: [1/16 .. 16]
# Tier 2K / 3K maps to a sensible WxH for the requested aspect ratio.
PIXEL_DIMS: dict[str, dict[str, str]] = {
    "2K": {
        "1:1":  "2048x2048",
        "16:9": "2560x1440",
        "9:16": "1440x2560",
        "4:3":  "2240x1680",
        "3:4":  "1680x2240",
    },
    "3K": {
        "1:1":  "3072x3072",
        "16:9": "3072x1728",
        "9:16": "1728x3072",
        "4:3":  "3072x2304",
        "3:4":  "2304x3072",
    },
}


def _resolve_size(aspect_ratio: str | None, image_size: str | None) -> str:
    tier = (image_size or "2K").upper()
    ratio = aspect_ratio or "1:1"
    return PIXEL_DIMS.get(tier, {}).get(ratio, tier)


class ApiyiSeedreamImage(BaseTool):
    name = "apiyi_seedream_image"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "image_generation"
    provider = "apiyi_seedream"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API

    dependencies = []
    install_instructions = (
        "Set APIYI_API_KEY to your APIYI API key.\n"
        "  Optionally set APIYI_BASE_URL (defaults to https://api.apiyi.com)."
    )
    agent_skills = []

    capabilities = ["text_to_image", "image_to_image"]
    supports = {
        "negative_prompt": False,
        "seed": False,
        "aspect_ratio": True,
        "reference_image": True,
        "multiple_reference_images": False,
        "high_resolution": True,
    }
    best_for = [
        "high-resolution 2K/3K photorealistic images",
        "single-image reference conditioning",
        "Seedream 5.0 via APIYI gateway",
    ]
    not_good_for = [
        "multi-reference editing (use apiyi_flash_image instead)",
        "negative prompts",
        "offline generation",
    ]
    fallback_tools = ["apiyi_flash_image", "flux_image", "google_imagen"]

    input_schema = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {"type": "string", "description": "Image generation prompt"},
            "aspect_ratio": {
                "type": "string",
                "enum": ["1:1", "16:9", "9:16", "4:3", "3:4"],
                "default": "1:1",
                "description": "Aspect ratio (used with image_size tier to derive pixel dimensions)",
            },
            "image_size": {
                "type": "string",
                "enum": ["2K", "3K"],
                "default": "2K",
                "description": "Resolution tier. 2K = ~4MP, 3K = ~9MP.",
            },
            "size": {
                "type": "string",
                "description": (
                    "Explicit 'WIDTHxHEIGHT' override (e.g. '2048x1152'). "
                    "Bypasses aspect_ratio/image_size derivation."
                ),
            },
            "image_url": {
                "type": "string",
                "description": "Reference image URL for image-to-image conditioning",
            },
            "image_path": {
                "type": "string",
                "description": "Local reference image path for image-to-image conditioning",
            },
            "output_path": {"type": "string"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=512, vram_mb=0, disk_mb=100, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=2, retryable_errors=["rate_limit", "timeout"])
    idempotency_key_fields = ["prompt", "aspect_ratio", "image_size", "size"]
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
        tier = (inputs.get("image_size") or "2K").upper()
        return 0.06 if tier == "3K" else 0.04

    @staticmethod
    def _resolve_reference_data_uri(inputs: dict[str, Any]) -> str | None:
        """Resolve reference image as a data URI or return the URL directly."""
        image_url = inputs.get("image_url")
        image_path = inputs.get("image_path")

        if image_path:
            path = Path(image_path)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
            mime, _ = mimetypes.guess_type(path.name)
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            return f"data:{mime or 'image/png'};base64,{encoded}"

        if image_url:
            return image_url

        return None

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
        image_size = inputs.get("image_size", "2K")

        size = inputs.get("size") or _resolve_size(aspect_ratio, image_size)

        try:
            reference = self._resolve_reference_data_uri(inputs)
        except FileNotFoundError as e:
            return ToolResult(success=False, error=str(e))
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to load reference image: {e}")

        payload: dict[str, Any] = {
            "model": MODEL,
            "prompt": prompt,
            "response_format": "url",
            "size": size,
        }
        if reference:
            payload["image"] = reference

        try:
            response = requests.post(
                f"{base_url}/v1/images/generations",
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
                    error=f"APIYI Seedream failed ({response.status_code}): {response.text[:500]}",
                )

            body = response.json()
            data = body.get("data") or []
            if not data:
                return ToolResult(
                    success=False,
                    error=f"No image in APIYI Seedream response: {str(body)[:500]}",
                )

            item = data[0]
            image_bytes = _decode_or_download(item, timeout=60)
            if not image_bytes:
                return ToolResult(success=False, error="APIYI Seedream returned empty image data")

            output_path = Path(inputs.get("output_path", "apiyi_seedream.png"))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(image_bytes)

        except Exception as e:
            return ToolResult(success=False, error=f"APIYI Seedream failed: {e}")

        usage = body.get("usage") or {}

        return ToolResult(
            success=True,
            data={
                "provider": self.provider,
                "model": MODEL,
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "image_size": image_size,
                "size": size,
                "output": str(output_path),
                "has_reference_image": reference is not None,
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            },
            artifacts=[str(output_path)],
            cost_usd=self.estimate_cost(inputs),
            duration_seconds=round(time.time() - start, 2),
            model=MODEL,
        )


def _decode_or_download(item: dict[str, Any], timeout: int) -> bytes:
    """Return raw image bytes from b64_json field or by downloading the URL."""
    b64 = item.get("b64_json")
    if b64:
        return base64.b64decode(b64)

    url = item.get("url")
    if url:
        import requests

        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.content

    return b""
