"""Veo 3.1 video generation via APIYI gateway.

APIYI provides access to Google Veo 3.1 models for text-to-video and
image-to-video generation through a simple REST API with async polling.
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
POLL_INTERVAL_S = 5
POLL_TIMEOUT_S = 600
RATE_LIMIT_WAIT_S = 10
MAX_RETRIES = 3
RETRY_DELAY_S = 5

RETRYABLE_PATTERNS = ["服务内部异常", "task_failed", "AUDIO_FILTERED"]

MODELS = [
    "veo-3.1-fast",
    "veo-3.1-landscape-fast",
    "veo-3.1-fast-fl",
    "veo-3.1-landscape-fast-fl",
]


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

    capabilities = ["text_to_video", "image_to_video"]
    supports = {
        "text_to_video": True,
        "image_to_video": True,
        "reference_to_video": False,
        "first_last_frame_to_video": False,
        "native_audio": False,
    }
    best_for = [
        "Veo 3.1 video generation via APIYI gateway",
        "text-to-video and image-to-video with frame locking",
        "alternative Veo access when fal.ai is unavailable",
    ]
    not_good_for = [
        "offline generation",
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
                "default": "veo-3.1-fast",
                "enum": MODELS,
                "description": (
                    "Veo model variant. '-fl' suffixes are frame-locked for image-to-video."
                ),
            },
            "image_path": {
                "type": "string",
                "description": "Local image path for image-to-video (uses frame-locked model)",
            },
            "image_url": {
                "type": "string",
                "description": "Image URL for image-to-video (uses frame-locked model)",
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
        # Approximate — APIYI pricing varies
        return 0.50

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        api_key = self._get_api_key()
        if not api_key:
            return ToolResult(
                success=False,
                error="APIYI_API_KEY not set. " + self.install_instructions,
            )

        start = time.time()
        try:
            result = self._generate(inputs, api_key)
        except Exception as exc:
            return ToolResult(success=False, error=f"APIYI Veo failed: {exc}")

        result.duration_seconds = round(time.time() - start, 2)
        result.cost_usd = self.estimate_cost(inputs)
        return result

    def _generate(self, inputs: dict[str, Any], api_key: str) -> ToolResult:
        import requests

        base_url = self._get_base_url()
        prompt = inputs["prompt"]
        model = inputs.get("model", "veo-3.1-fast")
        auth_header = {"Authorization": api_key}  # APIYI uses raw token, no Bearer prefix

        # Resolve image input for image-to-video
        image_bytes = self._resolve_image(inputs)
        has_image = image_bytes is not None

        last_error: str | None = None

        for attempt in range(1, MAX_RETRIES + 1):
            # Step 1 — Submit
            video_id = self._submit(
                base_url, auth_header, prompt, model, image_bytes, requests
            )

            # Step 2 — Poll
            status, error_msg = self._poll(base_url, auth_header, video_id, requests)

            if status == "completed":
                # Step 3 — Download
                video_bytes = self._download(base_url, auth_header, video_id, requests)

                if not video_bytes:
                    raise RuntimeError("APIYI returned empty video content.")

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
                        "has_image_input": has_image,
                        "attempts": attempt,
                    },
                    artifacts=[str(output_path)],
                    model=model,
                )

            if status == "failed":
                last_error = error_msg or "unknown error"
                if any(p in last_error for p in RETRYABLE_PATTERNS) and attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_S)
                    continue
                raise RuntimeError(f"APIYI Veo generation failed: {last_error}")

            if status == "timeout":
                raise RuntimeError("APIYI Veo generation timed out after 10 minutes.")

        raise RuntimeError(f"APIYI Veo max retries exceeded. Last error: {last_error}")

    def _submit(
        self,
        base_url: str,
        auth_header: dict[str, str],
        prompt: str,
        model: str,
        image_bytes: bytes | None,
        requests: Any,
    ) -> str:
        """Submit a video generation request, return the video ID."""
        if image_bytes is not None:
            from io import BytesIO

            files = {"input_reference": ("frame.jpg", BytesIO(image_bytes), "image/jpeg")}
            data = {"prompt": prompt, "model": model}
            resp = requests.post(
                f"{base_url}/v1/videos",
                headers=auth_header,
                data=data,
                files=files,
                timeout=30,
            )
        else:
            resp = requests.post(
                f"{base_url}/v1/videos",
                headers={**auth_header, "Content-Type": "application/json"},
                json={"prompt": prompt, "model": model},
                timeout=30,
            )

        if not resp.ok:
            raise RuntimeError(f"Submit failed (HTTP {resp.status_code}): {resp.text[:500]}")

        body = resp.json()
        video_id = body.get("id")
        if not video_id:
            raise RuntimeError(f"Submit returned no video ID: {body}")

        return video_id

    def _poll(
        self,
        base_url: str,
        auth_header: dict[str, str],
        video_id: str,
        requests: Any,
    ) -> tuple[str, str | None]:
        """Poll until completed, failed, or timeout. Returns (status, error_msg)."""
        deadline = time.time() + POLL_TIMEOUT_S
        headers = {**auth_header, "Content-Type": "application/json"}

        while time.time() < deadline:
            time.sleep(POLL_INTERVAL_S)

            resp = requests.get(
                f"{base_url}/v1/videos/{video_id}",
                headers=headers,
                timeout=15,
            )

            if resp.status_code == 429:
                time.sleep(RATE_LIMIT_WAIT_S)
                continue

            if not resp.ok:
                raise RuntimeError(f"Poll failed (HTTP {resp.status_code}): {resp.text[:500]}")

            data = resp.json()
            status = data.get("status", "")

            if status == "completed":
                return "completed", None

            if status == "failed":
                error = data.get("error", "unknown error")
                if not isinstance(error, str):
                    error = str(error)
                return "failed", error

        return "timeout", None

    def _download(
        self,
        base_url: str,
        auth_header: dict[str, str],
        video_id: str,
        requests: Any,
    ) -> bytes:
        """Download completed video bytes."""
        headers = {**auth_header, "Content-Type": "application/json"}
        resp = requests.get(
            f"{base_url}/v1/videos/{video_id}/content",
            headers=headers,
            timeout=120,
        )
        if not resp.ok:
            raise RuntimeError(f"Download failed (HTTP {resp.status_code}): {resp.text[:500]}")
        return resp.content

    @staticmethod
    def _resolve_image(inputs: dict[str, Any]) -> bytes | None:
        """Resolve image from path or URL, return raw bytes or None."""
        image_path = inputs.get("image_path")
        image_url = inputs.get("image_url")

        if image_path:
            path = Path(image_path)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
            return path.read_bytes()

        if image_url:
            import requests

            resp = requests.get(image_url, timeout=30)
            resp.raise_for_status()
            return resp.content

        return None
