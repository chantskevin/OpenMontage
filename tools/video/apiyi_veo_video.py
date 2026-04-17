"""Veo 3.1 video generation via APIYI gateway.

APIYI provides access to Google Veo 3.1 models for text-to-video and
image-to-video generation through a simple REST API with async polling.
"""

from __future__ import annotations

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
POLL_INTERVAL_S = 5
POLL_TIMEOUT_S = 600
RATE_LIMIT_WAIT_S = 10
MAX_RETRIES = 3
RETRY_DELAY_S = 5

RETRYABLE_PATTERNS = ["服务内部异常", "task_failed", "AUDIO_FILTERED"]

# APIYI encodes aspect ratio, speed, HD/4K, and frame-lock into the model name:
#   veo-3.1[-landscape][-fast|-relaxed][-fl][-hd|-4k]
#
# Suffix semantics:
#   (none)      — portrait 720x1280 (default)
#   landscape   — landscape 1280x720
#   fast        — faster/cheaper
#   relaxed     — relaxed speed (HD/4K only)
#   fl          — first/last-frame mode (image-to-video)
#   hd          — HD (landscape only)
#   4k          — 4K resolution
MODELS = [
    # Portrait (720x1280) — standard
    "veo-3.1",
    "veo-3.1-fl",
    "veo-3.1-fast",
    "veo-3.1-fast-fl",
    # Landscape (1280x720) — standard
    "veo-3.1-landscape",
    "veo-3.1-landscape-fl",
    "veo-3.1-landscape-fast",
    "veo-3.1-landscape-fast-fl",
    # Landscape HD
    "veo-3.1-landscape-hd",
    "veo-3.1-landscape-fast-hd",
    "veo-3.1-landscape-relaxed-hd",
    "veo-3.1-landscape-fl-hd",
    "veo-3.1-landscape-fast-fl-hd",
    "veo-3.1-landscape-relaxed-fl-hd",
    # 4K
    "veo-3.1-4k",
    "veo-3.1-fl-4k",
    "veo-3.1-fast-4k",
    "veo-3.1-fast-fl-4k",
    "veo-3.1-relaxed-4k",
    "veo-3.1-landscape-4k",
    "veo-3.1-landscape-fl-4k",
    "veo-3.1-landscape-fast-4k",
    "veo-3.1-landscape-fast-fl-4k",
    "veo-3.1-landscape-relaxed-4k",
]


def _model_pricing(model: str) -> float:
    """Estimate per-video cost from APIYI model name."""
    is_fast_or_relaxed = "fast" in model or "relaxed" in model
    if "-4k" in model:
        return 0.45 if is_fast_or_relaxed else 0.55
    if "-hd" in model:
        return 0.25 if is_fast_or_relaxed else 0.35
    return 0.15 if "fast" in model else 0.25


def _model_aspect_ratio(model: str) -> str:
    return "16:9" if "landscape" in model else "9:16"


def _model_resolution(model: str) -> str:
    if "-4k" in model:
        return "4k"
    if "-hd" in model:
        return "1080p"
    return "720p"


def _derive_model(
    explicit_model: str | None,
    aspect_ratio: str | None,
    resolution: str | None,
    fast: bool,
    has_image: bool,
) -> str:
    """Pick a model name from canonical selector params when no explicit model is given."""
    if explicit_model:
        return explicit_model

    parts = ["veo-3.1"]

    landscape = aspect_ratio in ("16:9", "landscape", "horizontal")
    if landscape:
        parts.append("landscape")

    if fast:
        parts.append("fast")

    if has_image:
        parts.append("fl")

    res = (resolution or "").lower()
    if res == "4k":
        parts.append("4k")
    elif res in ("hd", "1080p") and landscape:
        # HD is landscape-only per APIYI
        parts.append("hd")

    return "-".join(parts)


class ApiyiVeoVideo(BaseTool):
    name = "apiyi_veo_video"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "video_generation"
    provider = "apiyi"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.ASYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API

    dependencies = []
    install_instructions = (
        "Set APIYI_API_KEY to your APIYI API key.\n"
        "  Optionally set APIYI_BASE_URL (defaults to https://api.apiyi.com)."
    )
    agent_skills = ["ai-video-gen"]

    capabilities = ["text_to_video", "image_to_video", "first_last_frame_to_video"]
    supports = {
        "text_to_video": True,
        "image_to_video": True,
        "reference_to_video": False,
        "first_last_frame_to_video": True,
        "native_audio": False,
        "portrait": True,
        "landscape": True,
        "hd": True,
        "4k": True,
    }
    best_for = [
        "Veo 3.1 video generation via APIYI gateway",
        "portrait and landscape with 720p / HD / 4K variants",
        "first/last-frame locked image-to-video",
        "alternative Veo access when fal.ai is unavailable",
    ]
    not_good_for = [
        "offline generation",
        "native synced audio",
        "quick iteration (async polling required)",
    ]
    fallback_tools = ["veo_video", "kling_video", "minimax_video"]

    input_schema = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {"type": "string", "description": "Video generation prompt"},
            "model": {
                "type": "string",
                "enum": MODELS,
                "description": (
                    "Explicit APIYI model name. If omitted, derived from aspect_ratio, "
                    "resolution, fast, and image inputs. "
                    "Pattern: veo-3.1[-landscape][-fast|-relaxed][-fl][-hd|-4k]"
                ),
            },
            "aspect_ratio": {
                "type": "string",
                "enum": ["16:9", "9:16"],
                "default": "9:16",
                "description": "Used to derive model when 'model' not provided. 16:9 → landscape, 9:16 → portrait.",
            },
            "resolution": {
                "type": "string",
                "enum": ["720p", "1080p", "hd", "4k"],
                "default": "720p",
                "description": "Used to derive model. HD requires landscape. 4K available both orientations.",
            },
            "fast": {
                "type": "boolean",
                "default": True,
                "description": "Used to derive model. true → cheaper '-fast' variant (~40% cost reduction).",
            },
            "image_path": {
                "type": "string",
                "description": "Local image path (first frame for -fl models).",
            },
            "image_url": {
                "type": "string",
                "description": "Image URL (first frame for -fl models).",
            },
            "last_frame_path": {
                "type": "string",
                "description": "Local image path for last frame (enables first-last-frame mode with -fl models).",
            },
            "last_frame_url": {
                "type": "string",
                "description": "Image URL for last frame (enables first-last-frame mode with -fl models).",
            },
            "output_path": {"type": "string"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=512, vram_mb=0, disk_mb=500, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=3, retryable_errors=["rate_limit", "timeout", "task_failed"])
    idempotency_key_fields = ["prompt", "model"]
    side_effects = ["writes video file to output_path", "calls APIYI API"]
    user_visible_verification = ["Watch generated clip for visual quality and motion"]

    def _get_api_key(self) -> str | None:
        return os.environ.get("APIYI_API_KEY")

    def _get_base_url(self) -> str:
        return os.environ.get("APIYI_BASE_URL", DEFAULT_BASE_URL)

    def get_status(self) -> ToolStatus:
        if self._get_api_key():
            return ToolStatus.AVAILABLE
        return ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        model = self._resolve_model(inputs)
        return _model_pricing(model)

    @staticmethod
    def _resolve_model(inputs: dict[str, Any]) -> str:
        """Pick the model name from explicit or derived inputs."""
        has_image = bool(
            inputs.get("image_path") or inputs.get("image_url")
            or inputs.get("last_frame_path") or inputs.get("last_frame_url")
        )
        return _derive_model(
            explicit_model=inputs.get("model"),
            aspect_ratio=inputs.get("aspect_ratio"),
            resolution=inputs.get("resolution"),
            fast=bool(inputs.get("fast", True)),
            has_image=has_image,
        )

    @staticmethod
    def _resolve_frame(path_value: str | None, url_value: str | None) -> bytes | None:
        """Resolve a single frame from path or URL, return raw bytes or None."""
        if path_value:
            path = Path(path_value)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
            return path.read_bytes()

        if url_value:
            import requests

            resp = requests.get(url_value, timeout=30)
            resp.raise_for_status()
            return resp.content

        return None

    @classmethod
    def _resolve_frames(cls, inputs: dict[str, Any]) -> list[bytes]:
        """Resolve first and (optionally) last frame bytes for -fl models."""
        frames: list[bytes] = []
        first = cls._resolve_frame(inputs.get("image_path"), inputs.get("image_url"))
        if first is not None:
            frames.append(first)
        last = cls._resolve_frame(inputs.get("last_frame_path"), inputs.get("last_frame_url"))
        if last is not None:
            frames.append(last)
        return frames

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
        auth_header = {"Authorization": api_key}  # APIYI uses raw token, no Bearer prefix

        # Resolve frames (first + optional last for -fl models)
        try:
            frames = self._resolve_frames(inputs)
        except FileNotFoundError as e:
            return ToolResult(success=False, error=str(e))
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to resolve image input: {e}")

        has_image = len(frames) > 0

        # Resolve model (explicit or derived from aspect_ratio/resolution/fast/has_image)
        model = self._resolve_model(inputs)
        if model not in MODELS:
            return ToolResult(
                success=False,
                error=f"Unknown APIYI model '{model}'. Valid models: {MODELS}",
            )
        if frames and "-fl" not in model:
            return ToolResult(
                success=False,
                error=(
                    f"Image input provided but model '{model}' is not frame-locked. "
                    f"Use a -fl variant (e.g. 'veo-3.1-fast-fl', 'veo-3.1-landscape-fast-fl')."
                ),
            )

        last_error: str | None = None

        try:
            for attempt in range(1, MAX_RETRIES + 1):
                # Step 1 — Submit video generation
                if frames:
                    from io import BytesIO

                    # Multipart form with one input_reference per frame
                    files = [
                        ("input_reference", (f"frame_{i}.jpg", BytesIO(buf), "image/jpeg"))
                        for i, buf in enumerate(frames)
                    ]
                    data = {"prompt": prompt, "model": model}
                    submit_resp = requests.post(
                        f"{base_url}/v1/videos",
                        headers=auth_header,
                        data=data,
                        files=files,
                        timeout=30,
                    )
                else:
                    submit_resp = requests.post(
                        f"{base_url}/v1/videos",
                        headers={**auth_header, "Content-Type": "application/json"},
                        json={"prompt": prompt, "model": model},
                        timeout=30,
                    )

                if not submit_resp.ok:
                    return ToolResult(
                        success=False,
                        error=f"APIYI Veo submit failed ({submit_resp.status_code}): {submit_resp.text[:500]}",
                    )

                body = submit_resp.json()
                video_id = body.get("id")
                if not video_id:
                    return ToolResult(
                        success=False,
                        error=f"APIYI Veo submit returned no video ID: {body}",
                    )

                # Step 2 — Poll until completed or failed
                deadline = time.time() + POLL_TIMEOUT_S
                headers = {**auth_header, "Content-Type": "application/json"}
                completed = False
                failed = False

                while time.time() < deadline:
                    time.sleep(POLL_INTERVAL_S)

                    poll_resp = requests.get(
                        f"{base_url}/v1/videos/{video_id}",
                        headers=headers,
                        timeout=15,
                    )

                    if poll_resp.status_code == 429:
                        time.sleep(RATE_LIMIT_WAIT_S)
                        continue

                    if not poll_resp.ok:
                        return ToolResult(
                            success=False,
                            error=f"APIYI Veo poll failed ({poll_resp.status_code}): {poll_resp.text[:500]}",
                        )

                    poll_data = poll_resp.json()
                    status = poll_data.get("status", "")

                    if status == "completed":
                        completed = True
                        break

                    if status == "failed":
                        error = poll_data.get("error", "unknown error")
                        if not isinstance(error, str):
                            error = str(error)
                        last_error = error

                        if any(p in error for p in RETRYABLE_PATTERNS) and attempt < MAX_RETRIES:
                            failed = True
                            break
                        return ToolResult(
                            success=False,
                            error=f"APIYI Veo generation failed: {error}",
                        )

                if failed:
                    time.sleep(RETRY_DELAY_S)
                    continue

                if not completed:
                    return ToolResult(
                        success=False,
                        error="APIYI Veo generation timed out after 10 minutes",
                    )

                # Step 3 — Download video bytes
                content_resp = requests.get(
                    f"{base_url}/v1/videos/{video_id}/content",
                    headers=headers,
                    timeout=120,
                )
                if not content_resp.ok:
                    return ToolResult(
                        success=False,
                        error=f"APIYI Veo download failed ({content_resp.status_code}): {content_resp.text[:500]}",
                    )

                video_bytes = content_resp.content
                if not video_bytes:
                    return ToolResult(
                        success=False,
                        error="APIYI Veo returned empty video content",
                    )

                output_path = Path(inputs.get("output_path", "apiyi_veo_output.mp4"))
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(video_bytes)

                return ToolResult(
                    success=True,
                    data={
                        "provider": self.provider,
                        "model": model,
                        "prompt": prompt,
                        "output": str(output_path),
                        "aspect_ratio": _model_aspect_ratio(model),
                        "resolution": _model_resolution(model),
                        "frame_count": len(frames),
                        "has_image_input": has_image,
                        "attempts": attempt,
                    },
                    artifacts=[str(output_path)],
                    cost_usd=self.estimate_cost(inputs),
                    duration_seconds=round(time.time() - start, 2),
                    model=model,
                )

            return ToolResult(
                success=False,
                error=f"APIYI Veo max retries exceeded. Last error: {last_error}",
            )

        except Exception as e:
            return ToolResult(success=False, error=f"APIYI Veo failed: {e}")
