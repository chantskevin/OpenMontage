"""Veo 3.1 video generation via APIYI gateway (Official API Version).

APIYI provides access to Google Veo 3.1 models for text-to-video and
image-to-video generation through a simple REST API with async polling.
This version implements the official API specification.
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

# The new official APIYi Veo 3.1 has only two models
MODELS = [
    "veo-3.1-fast-generate-preview",
    "veo-3.1-generate-preview",
]


def _model_pricing(model: str) -> float:
    """Official Veo 3.1 pricing: $0.30 for fast, $1.20 for standard."""
    if model == "veo-3.1-fast-generate-preview":
        return 0.30
    return 1.20


class ApiyiVeoVideo(BaseTool):
    name = "apiyi_veo_video"
    version = "0.2.0"
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
        "portrait": True,
        "landscape": True,
        "hd": True,
        "4k": True,
    }
    best_for = [
        "Veo 3.1 video generation via official APIYI endpoints",
        "portrait and landscape with 720p / 1080p / 4K resolutions",
        "high-quality image-to-video and text-to-video",
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
            "operation": {
                "type": "string",
                "enum": ["text_to_video", "image_to_video"],
                "default": "text_to_video",
                "description": "Generation mode. image_to_video requires image_url/image_path.",
            },
            "model": {
                "type": "string",
                "enum": MODELS,
                "default": "veo-3.1-fast-generate-preview",
                "description": "APIYI model name. veo-3.1-fast-generate-preview ($0.30) or veo-3.1-generate-preview ($1.20).",
            },
            "aspect_ratio": {
                "type": "string",
                "enum": ["16:9", "9:16"],
                "default": "16:9",
                "description": "Aspect ratio of the generated video.",
            },
            "resolution": {
                "type": "string",
                "enum": ["720p", "1080p", "4k"],
                "default": "720p",
                "description": "Target resolution.",
            },
            "duration": {
                "type": "string",
                "enum": ["8"],
                "default": "8",
                "description": "Duration in seconds (only '8' is supported)",
            },
            "image_url": {"type": "string", "description": "Reference image URL for image_to_video"},
            "image_path": {"type": "string", "description": "Local reference image path for image_to_video"},
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
        model = inputs.get("model", "veo-3.1-fast-generate-preview")
        if model not in MODELS:
            model = "veo-3.1-fast-generate-preview"
        return _model_pricing(model)

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

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        # output_path is required.
        output_path_raw = inputs.get("output_path")
        if not output_path_raw or not isinstance(output_path_raw, str):
            return ToolResult(
                success=False,
                error=(
                    "apiyi_veo_video: 'output_path' is required and must be a "
                    "non-empty string. Pass an absolute path (or a path "
                    "relative to the current workspace) where the generated "
                    "mp4 should be written."
                ),
            )

        # Mock mode
        if os.environ.get("APIYI_MOCK") == "1":
            return self._execute_mock(inputs, output_path_raw)

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
        operation = inputs.get("operation", "text_to_video")
        if operation == "first_last_frame_to_video":
            operation = "image_to_video"
        auth_header = {"Authorization": api_key}

        # Resolve model
        model = inputs.get("model", "veo-3.1-fast-generate-preview")
        if model not in MODELS:
            if "fast" in model:
                model = "veo-3.1-fast-generate-preview"
            else:
                model = "veo-3.1-generate-preview"

        # Normalize Aspect Ratio
        aspect_ratio_input = inputs.get("aspect_ratio", "16:9")
        aspectRatio = "16:9" if aspect_ratio_input in ("16:9", "landscape", "horizontal") else "9:16"

        # Normalize Resolution & Size
        resolution_input = inputs.get("resolution", "720p")
        if resolution_input == "4k":
            size = "3840x2160" if aspectRatio == "16:9" else "2160x3840"
            resolution = "4k"
        elif resolution_input in ("hd", "1080p"):
            size = "1920x1080" if aspectRatio == "16:9" else "1080x1920"
            resolution = "1080p"
        else:
            size = "1280x720" if aspectRatio == "16:9" else "720x1280"
            resolution = "720p"

        # Resolve image frame for image_to_video
        frame_bytes: bytes | None = None
        if operation == "image_to_video":
            try:
                frame_path = inputs.get("image_path") or inputs.get("first_frame_path")
                frame_url = inputs.get("image_url") or inputs.get("first_frame_url")
                frame_bytes = self._resolve_frame(frame_path, frame_url)
            except Exception as e:
                return ToolResult(success=False, error=f"Failed to resolve image input: {e}")

            if frame_bytes is None:
                return ToolResult(
                    success=False,
                    error="image_to_video requires image_url/image_path or first_frame_url/first_frame_path",
                )

        last_error: str | None = None

        try:
            for attempt in range(1, MAX_RETRIES + 1):
                # Step 1 — Submit video generation
                if operation == "image_to_video" and frame_bytes is not None:
                    from io import BytesIO

                    # Multipart form-data payload with exactly one file field
                    files = {
                        "input_reference": ("image.jpg", BytesIO(frame_bytes), "image/jpeg")
                    }
                    data = {
                        "model": model,
                        "prompt": prompt,
                        "duration": "8",  # strictly string "8"
                        "resolution": resolution,
                        "aspectRatio": aspectRatio,
                        "size": size,
                    }
                    submit_resp = requests.post(
                        f"{base_url}/v1/videos",
                        headers=auth_header,
                        data=data,
                        files=files,
                        timeout=30,
                    )
                else:
                    # Text-to-Video JSON payload
                    payload = {
                        "model": model,
                        "prompt": prompt,
                        "duration": "8",  # strictly string "8"
                        "size": size,
                        "metadata": {
                            "resolution": resolution,
                            "aspectRatio": aspectRatio,
                        },
                    }
                    submit_resp = requests.post(
                        f"{base_url}/v1/videos",
                        headers={**auth_header, "Content-Type": "application/json"},
                        json=payload,
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

                # Step 3 — Download video bytes. Add mitigation for the completed-but-in-progress download race.
                content_resp = requests.get(
                    f"{base_url}/v1/videos/{video_id}/content",
                    headers=headers,
                    timeout=120,
                )
                _IN_PROGRESS_TEXT = "task status is IN_PROGRESS"
                for delay in (4, 8, 16, 24, 32):
                    if content_resp.status_code != 400:
                        break
                    if _IN_PROGRESS_TEXT not in (content_resp.text or ""):
                        break
                    time.sleep(delay)
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

                output_path = Path(output_path_raw)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(video_bytes)

                return ToolResult(
                    success=True,
                    data={
                        "provider": self.provider,
                        "model": model,
                        "prompt": prompt,
                        "operation": operation,
                        "output": str(output_path),
                        "aspect_ratio": aspectRatio,
                        "resolution": resolution,
                        "frame_count": 1 if operation == "image_to_video" else 0,
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

    def _execute_mock(
        self, inputs: dict[str, Any], output_path_raw: str
    ) -> ToolResult:
        """APIYI_MOCK=1 path: generate a landscape 1280x720 8-second test MP4
        locally via ffmpeg and return a success envelope shaped like the real
        path.
        """
        import shutil
        import subprocess

        start = time.time()
        operation = inputs.get("operation", "text_to_video")
        if operation == "first_last_frame_to_video":
            operation = "image_to_video"
        model = inputs.get("model", "veo-3.1-fast-generate-preview")
        if model not in MODELS:
            if "fast" in model:
                model = "veo-3.1-fast-generate-preview"
            else:
                model = "veo-3.1-generate-preview"

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            return ToolResult(
                success=False,
                error="APIYI_MOCK=1 requires ffmpeg on PATH to generate a test clip.",
            )

        output_path = Path(output_path_raw)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        aspect_ratio_input = inputs.get("aspect_ratio", "16:9")
        aspectRatio = "16:9" if aspect_ratio_input in ("16:9", "landscape", "horizontal") else "9:16"

        resolution_input = inputs.get("resolution", "720p")
        if resolution_input == "4k":
            width, height = (3840, 2160) if aspectRatio == "16:9" else (2160, 3840)
            resolution = "4k"
        elif resolution_input in ("hd", "1080p"):
            width, height = (1920, 1080) if aspectRatio == "16:9" else (1080, 1920)
            resolution = "1080p"
        else:
            width, height = (1280, 720) if aspectRatio == "16:9" else (720, 1280)
            resolution = "720p"

        # generate an 8-second video
        cmd = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size={width}x{height}:rate=24:duration=8",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=48000:cl=stereo",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return ToolResult(
                success=False,
                error=f"APIYI_MOCK ffmpeg failed: {result.stderr[:500]}",
            )

        return ToolResult(
            success=True,
            data={
                "provider": self.provider,
                "model": model,
                "prompt": inputs["prompt"],
                "operation": operation,
                "output": str(output_path),
                "aspect_ratio": aspectRatio,
                "resolution": resolution,
                "frame_count": 1 if operation == "image_to_video" else 0,
                "attempts": 1,
                "mock": True,
            },
            artifacts=[str(output_path)],
            cost_usd=0.0,
            duration_seconds=round(time.time() - start, 2),
            model=model,
        )
