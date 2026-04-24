"""Video composition tool — FFmpeg + Remotion + HyperFrames (runtime-aware).

Pipeline-facing orchestration surface for composition. Takes `edit_decisions`,
`asset_manifest`, and audio, and delegates to the technical runtime chosen
at proposal stage.

Routing is driven by `edit_decisions.render_runtime` (locked at proposal):

- `remotion`   → React-based frame-accurate render via `npx remotion render`.
                 Handles the existing scene-component stack, word-level captions,
                 TalkingHead/CinematicRenderer. Current default.
- `hyperframes` → HTML/CSS/GSAP render via `hyperframes_compose`.
                 Handles kinetic typography, product promos, website-to-video,
                 registry blocks. Added in the parallel-runtime initiative.
- `ffmpeg`     → FFmpeg concat/trim. Used only for simple video cuts without
                 composition, or when the approved path explicitly names FFmpeg.

Silent runtime swaps are forbidden by governance. If the chosen runtime is
unavailable or fails, this tool surfaces a structured blocker and waits for
the agent to re-ask the user rather than substituting a different engine.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ResumeSupport,
    ToolResult,
    ToolStability,
    ToolTier,
)


class VideoCompose(BaseTool):
    name = "video_compose"
    version = "0.1.0"
    tier = ToolTier.CORE
    capability = "video_post"
    provider = "ffmpeg"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC

    dependencies = ["cmd:ffmpeg"]
    install_instructions = "Install FFmpeg: https://ffmpeg.org/download.html"
    agent_skills = ["remotion-best-practices", "remotion", "ffmpeg"]

    capabilities = [
        "compose_cuts",
        "burn_subtitles",
        "overlay_assets",
        "encode_profile",
        "remotion_render",
    ]

    input_schema = {
        "type": "object",
        "required": ["operation"],
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["compose", "render", "remotion_render", "burn_subtitles", "overlay", "encode"],
                "description": (
                    "compose: low-level concat cuts + audio + subtitles. "
                    "render: high-level — resolves asset IDs, auto-routes to Remotion "
                    "for images/animations or FFmpeg for video-only. Preferred for compose-director. "
                    "remotion_render: render via Remotion (Node.js). "
                    "burn_subtitles: burn subtitle file into existing video. "
                    "overlay: composite overlays onto base video. "
                    "encode: re-encode to a target profile/codec."
                ),
            },
            "input_path": {"type": "string"},
            "output_path": {"type": "string"},
            "edit_decisions": {
                "type": "object",
                "description": "Full edit_decisions artifact (required for compose/render)",
            },
            "asset_manifest": {
                "type": "object",
                "description": (
                    "Full asset_manifest artifact (required for render). "
                    "Used to resolve asset IDs in cuts[].source to file paths."
                ),
            },
            "proposal_packet": {
                "type": "object",
                "description": (
                    "Full proposal_packet artifact. Optional but STRONGLY "
                    "recommended — when present, final_review compares "
                    "proposal_packet.production_plan.render_runtime against "
                    "edit_decisions.render_runtime and flags runtime_swap_detected. "
                    "Also enables the spoken-audio narration coverage check "
                    "(forks #22 / #23): voice_led / dialogue_led runs without "
                    "matching narration assets get blocked here instead of "
                    "shipping a silent video."
                ),
            },
            "script": {
                "type": "object",
                "description": (
                    "Full script artifact. Strongly recommended when "
                    "audio_treatment.mode is voice_led / dialogue_led — "
                    "enables per-section narration coverage (every non-empty "
                    "section.id must appear as a scene_id on a narration "
                    "asset). Without it, falls back to the weaker '≥1 "
                    "narration asset' check, which lets partial-narration "
                    "runs through (fork issue #23 weak form)."
                ),
            },
            "narration_transcript_path": {
                "type": "string",
                "description": (
                    "Path to a word-level transcript JSON (from `transcriber` "
                    "tool output). Optional but STRONGLY recommended: when "
                    "combined with script_path/script_text, final_review "
                    "runs transcript_comparison and catches TTS failures "
                    "like 'Chirp3-HD reads ... as the word dot'. Without "
                    "it, content-level audio bugs ship silently."
                ),
            },
            "script_path": {
                "type": "string",
                "description": (
                    "Path to the source narration script (plain text). "
                    "Used by transcript_comparison to diff against the "
                    "transcribed audio. Provide this OR script_text."
                ),
            },
            "script_text": {
                "type": "string",
                "description": (
                    "Inline source narration script. Used by "
                    "transcript_comparison when a file path is unavailable."
                ),
            },
            "subtitle_path": {"type": "string"},
            "subtitle_style": {
                "type": "object",
                "description": "ASS subtitle styling. Also extracted from edit_decisions.subtitles if not provided.",
                "properties": {
                    "font": {"type": "string", "default": "Arial"},
                    "font_size": {"type": "integer", "default": 24},
                    "primary_color": {"type": "string", "default": "&HFFFFFF"},
                    "outline_color": {"type": "string", "default": "&H000000"},
                    "outline_width": {"type": "number", "default": 2},
                    "margin_v": {"type": "integer", "default": 40},
                    "alignment": {"type": "integer", "default": 2},
                },
            },
            "overlays": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "asset_path": {"type": "string"},
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "width": {"type": "number"},
                        "height": {"type": "number"},
                        "start_seconds": {"type": "number"},
                        "end_seconds": {"type": "number"},
                        "opacity": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                },
            },
            "audio_path": {"type": "string", "description": "Mixed audio to mux into output"},
            "profile": {
                "type": "string",
                "description": (
                    "Media profile name from media_profiles.py "
                    "(e.g. youtube_landscape, tiktok, instagram_reels). "
                    "Applied in render and encode operations."
                ),
            },
            "options": {
                "type": "object",
                "description": "Render options (used by the render operation)",
                "properties": {
                    "subtitle_burn": {"type": "boolean", "default": True},
                    "two_pass_encode": {"type": "boolean", "default": False},
                },
            },
            "codec": {"type": "string", "default": "libx264"},
            "crf": {"type": "integer", "default": 23},
            "preset": {"type": "string", "default": "medium"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=4, ram_mb=2048, vram_mb=0, disk_mb=5000, network_required=False
    )

    # Remotion scene types that trigger React-based rendering
    _REMOTION_COMPONENTS = [
        "text_card", "stat_card", "callout", "comparison",
        "progress", "chart", "bar_chart", "line_chart", "pie_chart", "kpi_grid",
    ]

    best_for = [
        "Final render for explainer and animation pipelines",
        "Image-to-video with spring animations (Remotion)",
        "Animated text cards, stat cards, charts (Remotion)",
        "Complex transitions between scenes (Remotion)",
        "Pure video concat and trim (FFmpeg)",
    ]
    retry_policy = RetryPolicy(max_retries=1, retryable_errors=["Conversion failed"])
    resume_support = ResumeSupport.FROM_START
    idempotency_key_fields = ["operation", "input_path", "edit_decisions"]
    side_effects = ["writes video file to output_path"]
    user_visible_verification = [
        "Play the composed output and verify cuts, subtitles, and overlays",
    ]

    def _remotion_available(self) -> bool:
        """Check if Remotion rendering is available (requires npx + composer project + node_modules)."""
        import shutil as _shutil

        if not _shutil.which("npx"):
            return False
        composer_dir = Path(__file__).resolve().parent.parent.parent / "remotion-composer"
        if not composer_dir.exists() or not (composer_dir / "package.json").exists():
            return False
        # Check that node_modules are actually installed — without this,
        # npx remotion render will fail even though the project exists.
        if not (composer_dir / "node_modules").exists():
            return False
        return True

    def _hyperframes_available(self) -> bool:
        """Check if HyperFrames rendering is available.

        Delegates to the dedicated tool so the availability check stays in
        one place (node 22 floor, ffmpeg + npx on PATH).
        """
        try:
            from tools.video.hyperframes_compose import HyperFramesCompose
            return bool(HyperFramesCompose()._runtime_check()["runtime_available"])
        except Exception:
            return False

    def get_info(self) -> dict[str, Any]:
        """Extend base get_info to surface all available render runtimes.

        Preflight reports each runtime's availability separately so the agent
        can choose an appropriate `render_runtime` at proposal stage. Silent
        fallback between runtimes is forbidden.
        """
        info = super().get_info()
        remotion_ok = self._remotion_available()
        hyperframes_ok = self._hyperframes_available()
        info["render_engines"] = {
            "ffmpeg": True,
            "remotion": remotion_ok,
            "hyperframes": hyperframes_ok,
        }
        # Backwards-compat alias — some proposal skills inspect this name.
        info["render_runtimes"] = info["render_engines"]

        if remotion_ok:
            info["remotion_components"] = self._REMOTION_COMPONENTS
            info["remotion_note"] = (
                "Remotion is available for React-based rendering. Use it for "
                "image-to-video with spring animations, animated text/stat cards, "
                "charts, callouts, comparisons, and word-level caption burn. "
                "Prefer Remotion over Ken Burns pan-and-zoom for explainer "
                "and motion-graphics pipelines that already use the scene-component stack."
            )
        else:
            composer_dir = Path(__file__).resolve().parent.parent.parent / "remotion-composer"
            if composer_dir.exists() and (composer_dir / "package.json").exists() and not (composer_dir / "node_modules").exists():
                info["remotion_note"] = (
                    "Remotion project exists but node_modules are NOT installed. "
                    "Run 'cd remotion-composer && npm install' to enable Remotion rendering."
                )
            else:
                info["remotion_note"] = (
                    "Remotion is NOT available (needs Node.js/npx + remotion-composer + node_modules)."
                )

        if hyperframes_ok:
            info["hyperframes_note"] = (
                "HyperFrames is available for HTML/CSS/GSAP composition. Use it "
                "for kinetic typography, product promos, launch reels, "
                "website-to-video, and registry-block-driven scenes. Consumed via "
                "'npx hyperframes' (npm package: 'hyperframes'). "
                "Before locking render_runtime='hyperframes' at the proposal stage, "
                "verify the runtime with `hyperframes_compose` operation='doctor' "
                "or `make hyperframes-doctor`. An 'available' flag from the runtime "
                "check means node + ffmpeg + the npm package all resolve; it does "
                "not guarantee a render will succeed on the first specific "
                "composition."
            )
        else:
            info["hyperframes_note"] = (
                "HyperFrames is NOT available. Requires Node.js >= 22, FFmpeg, "
                "npx on PATH, and the 'hyperframes' npm package to be resolvable. "
                "Run `make hyperframes-doctor` to see the specific missing piece, "
                "or call `hyperframes_compose` operation='doctor' directly."
            )

        # Governance note — agents and reviewers consume this.
        info["runtime_governance"] = (
            "render_runtime is locked at proposal stage and carried unchanged "
            "through edit_decisions. Silent swaps are forbidden. If the "
            "chosen runtime fails, surface a structured blocker and wait for "
            "user approval before switching."
        )
        return info

    _VALID_OPERATIONS = (
        "compose", "render", "remotion_render",
        "burn_subtitles", "overlay", "encode",
    )

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        operation = inputs.get("operation")
        if not operation:
            return ToolResult(
                success=False,
                error=(
                    "video_compose: 'operation' is required. Pass one of: "
                    + ", ".join(self._VALID_OPERATIONS) + "."
                ),
            )
        start = time.time()

        try:
            if operation == "compose":
                result = self._compose(inputs)
            elif operation == "render":
                result = self._render(inputs)
            elif operation == "remotion_render":
                result = self._remotion_render(inputs)
            elif operation == "burn_subtitles":
                result = self._burn_subtitles(inputs)
            elif operation == "overlay":
                result = self._overlay(inputs)
            elif operation == "encode":
                result = self._encode(inputs)
            else:
                return ToolResult(success=False, error=f"Unknown operation: {operation}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))

        result.duration_seconds = round(time.time() - start, 2)
        return result

    _IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

    @staticmethod
    def _is_image(path: Path) -> bool:
        """Check if a file is a still image (routes to Remotion, not FFmpeg)."""
        return path.suffix.lower() in VideoCompose._IMAGE_EXTENSIONS

    @staticmethod
    def _has_audio_stream(path: Path) -> bool:
        """Return True iff ffprobe reports at least one audio stream.

        Many stock video clips (especially from Pexels) ship with no audio
        stream at all. If we blindly tell ffmpeg to transcode the 0:a stream
        on such a file it errors out. This helper lets the segment builder
        branch on stream presence so it can synthesize a silent track when
        needed, keeping the concat segment layout consistent.
        """
        try:
            out = subprocess.check_output(
                [
                    "ffprobe", "-v", "error",
                    "-select_streams", "a",
                    "-show_entries", "stream=codec_type",
                    "-of", "default=nw=1:nk=1",
                    str(path),
                ],
                stderr=subprocess.STDOUT,
                text=True,
            )
            return "audio" in out
        except Exception:
            return False

    def _compose(self, inputs: dict[str, Any]) -> ToolResult:
        """FFmpeg composition: concat video cuts, add audio, burn subtitles.

        Handles video sources only. Still images and animated scene types
        are routed to Remotion via the render operation — call compose
        directly only for pure video pipelines (e.g. talking-head).
        """
        edit_decisions = inputs.get("edit_decisions")
        if not edit_decisions:
            return ToolResult(success=False, error="edit_decisions required for compose")

        output_path = Path(inputs.get("output_path", "composed_output.mp4"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path = inputs.get("audio_path")
        subtitle_path = inputs.get("subtitle_path")
        codec = inputs.get("codec", "libx264")
        crf = inputs.get("crf", 23)
        preset = inputs.get("preset", "medium")
        profile_name = inputs.get("profile")

        cuts = edit_decisions.get("cuts", [])
        if not cuts:
            return ToolResult(success=False, error="No cuts in edit_decisions")

        # Resolve cuts[].source against asset_manifest IDs via the
        # shared helper used by both _render and _compose (see fork
        # issue #29 — compose used to skip this and fail on every
        # skill-conformant edit_decisions).
        asset_manifest = inputs.get("asset_manifest") or {}
        cuts = self._resolve_cut_sources(cuts, asset_manifest)

        # Resolve target resolution. Priority:
        #   1. explicit profile passed by caller
        #   2. autodetect from source-cut orientation (matches #27 fix
        #      that landed on the render path; #30 ports it to compose)
        #   3. fallback to 1920x1080 landscape default
        resolution = "1920x1080"
        if not profile_name:
            profile_name = self._auto_detect_canvas_profile(cuts)
        if profile_name:
            try:
                from lib.media_profiles import get_profile
                p = get_profile(profile_name)
                resolution = f"{p.width}x{p.height}"
            except (ImportError, ValueError):
                pass

        # Resolve subtitle style using the layered priority resolver
        # (explicit > edit_decisions > playbook > defaults)
        playbook_data = inputs.get("playbook")
        resolved_sub_style = self._resolve_subtitle_style(
            inputs.get("subtitle_style"),
            edit_decisions,
            playbook_data,
        )
        inputs = dict(inputs)
        inputs["subtitle_style"] = resolved_sub_style

        ed_subs = edit_decisions.get("subtitles", {})
        if ed_subs.get("source") and not subtitle_path:
            subtitle_path = ed_subs["source"]

        temp_dir = output_path.parent / ".compose_tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_segments: list[Path] = []
        concat_path: Path | None = None
        concat_out: Path | None = None

        try:
            for i, cut in enumerate(cuts):
                source = Path(cut["source"])
                if not source.exists():
                    # If the resolved source still doesn't exist, the
                    # caller probably passed an ID that isn't in the
                    # asset_manifest — surface both the (possibly
                    # ID-shaped) original and the manifest size so they
                    # can tell whether the lookup ran.
                    manifest_size = len(
                        (asset_manifest or {}).get("assets") or []
                    )
                    return ToolResult(
                        success=False,
                        error=(
                            f"Cut source not found: {source} "
                            f"(cut[{i}].source={cut.get('source')!r}, "
                            f"asset_manifest has {manifest_size} entries)"
                        ),
                    )

                seg_path = temp_dir / f"seg_{i:04d}.mp4"
                in_s = cut["in_seconds"]
                out_s = cut["out_seconds"]
                duration = out_s - in_s
                speed = cut.get("speed", 1.0)

                if self._is_image(source):
                    return ToolResult(
                        success=False,
                        error=(
                            f"Still image '{source.name}' in cuts. "
                            "Use operation='render' (auto-routes to Remotion) "
                            "or operation='remotion_render' for compositions "
                            "with images, animations, or component scenes."
                        ),
                    )
                else:
                    # Video source: trim to segment.
                    #
                    # Semantics:
                    #   -ss BEFORE -i   → fast input-level seek to in_s
                    #   -t  AFTER  -i   → "play for `duration` seconds"
                    #                     (unambiguous regardless of seek mode)
                    #
                    # We MUST re-encode here — `-c copy` cannot do frame-accurate
                    # cuts because it snaps to keyframes. With sparse GOPs (common
                    # in Pexels / AI-generated clips), stream-copy can produce
                    # segments significantly longer than `duration`, breaking the
                    # target timeline. Re-encoding with libx264/AAC is slower but
                    # gives exact cut boundaries. Same resolution in → same
                    # resolution out, so same-res inputs concat cleanly.
                    cmd = [
                        "ffmpeg", "-y",
                        "-ss", str(in_s),
                        "-t", str(duration),
                        "-i", str(source),
                    ]

                    # Normalize every segment to a consistent container so the
                    # concat-copy step is always safe. The concat demuxer with
                    # `-c copy` requires identical codec / resolution / fps /
                    # pix_fmt / sar across ALL segments — otherwise it throws
                    # "Non-monotonous DTS" or silently produces corrupt output.
                    #
                    # Default target is 1920x1080 @ 30fps, yuv420p, sar=1. If the
                    # source is smaller it letterboxes; if larger it downscales.
                    # Callers can override via edit_decisions.metadata.compose_target
                    # (future extension) but the defaults match the most common
                    # delivery profile (YouTube landscape).
                    vf_parts: list[str] = [
                        "scale=1920:1080:force_original_aspect_ratio=decrease",
                        "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black",
                        "setsar=1",
                        "fps=30",
                    ]
                    af_parts: list[str] = []
                    if speed != 1.0:
                        vf_parts.append(f"setpts={1.0/speed}*PTS")
                        af_parts.append(self._build_atempo(speed))

                    cmd.extend(["-filter:v", ",".join(vf_parts)])
                    if af_parts:
                        cmd.extend(["-filter:a", ",".join(af_parts)])

                    cmd.extend([
                        "-c:v", codec,
                        "-crf", str(crf),
                        "-preset", preset,
                        "-pix_fmt", "yuv420p",
                        "-r", "30",
                    ])

                    # Audio handling: some source clips have no audio stream
                    # (Pexels stock often ships silent). If we unconditionally
                    # ask ffmpeg to copy/encode the 0:a stream it errors out.
                    # Probe for an audio stream first — if present, transcode
                    # to AAC; if absent, synthesize a silent stereo track so
                    # concat segments have a consistent stream layout.
                    has_audio = self._has_audio_stream(source)
                    if has_audio:
                        cmd.extend(["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"])
                    else:
                        # Inject silent audio via lavfi before the output.
                        # We have to rebuild cmd to add the lavfi input
                        # before the output path and map streams explicitly.
                        cmd = [
                            "ffmpeg", "-y",
                            "-ss", str(in_s),
                            "-t", str(duration),
                            "-i", str(source),
                            "-f", "lavfi",
                            "-t", str(duration),
                            "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
                            "-filter:v", ",".join(vf_parts),
                        ]
                        if af_parts:
                            cmd.extend(["-filter:a", ",".join(af_parts)])
                        cmd.extend([
                            "-map", "0:v:0",
                            "-map", "1:a:0",
                            "-c:v", codec,
                            "-crf", str(crf),
                            "-preset", preset,
                            "-pix_fmt", "yuv420p",
                            "-r", "30",
                            "-c:a", "aac",
                            "-b:a", "192k",
                            "-ar", "48000",
                            "-ac", "2",
                        ])

                    cmd.append(str(seg_path))
                    self.run_command(cmd)

                temp_segments.append(seg_path)

            # Step 2: Concat segments
            concat_path = temp_dir / "concat_list.txt"
            with open(concat_path, "w", encoding="utf-8") as f:
                for seg in temp_segments:
                    safe = str(seg.resolve()).replace("\\", "/")
                    f.write(f"file '{safe}'\n")

            concat_out = temp_dir / "concat.mp4"
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", str(concat_path),
                "-c", "copy",
                str(concat_out),
            ]
            self.run_command(cmd)

            # Step 3: Apply subtitles and/or replace audio
            final_input = concat_out
            vfilters = []

            if subtitle_path and Path(subtitle_path).exists():
                style = inputs.get("subtitle_style", {})
                ass_style = self._build_subtitle_style(style)
                sub_escaped = str(Path(subtitle_path).resolve()).replace("\\", "/").replace(":", "\\:")
                vfilters.append(f"subtitles='{sub_escaped}':force_style='{ass_style}'")

            cmd = ["ffmpeg", "-y", "-i", str(final_input)]

            if audio_path and Path(audio_path).exists():
                cmd.extend(["-i", audio_path])

            # Determine if profile requires re-encoding (resize/fps change)
            # This must be checked BEFORE choosing copy vs encode, because
            # -s and -r are incompatible with -c:v copy.
            profile_flags: list[str] = []
            if profile_name:
                try:
                    from lib.media_profiles import get_profile
                    p = get_profile(profile_name)
                    profile_flags = ["-s", f"{p.width}x{p.height}", "-r", str(p.fps)]
                except (ImportError, ValueError):
                    pass

            needs_reencode = bool(vfilters) or bool(profile_flags)

            if needs_reencode:
                if vfilters:
                    cmd.extend(["-vf", ",".join(vfilters)])
                cmd.extend(["-c:v", codec, "-crf", str(crf), "-preset", preset])
                cmd.extend(profile_flags)
            else:
                cmd.extend(["-c:v", "copy"])

            if audio_path and Path(audio_path).exists():
                # Use type-based selectors (0:v, 1:a) instead of index-based
                # (0:v:0) because source videos may have audio as stream 0
                # and video as stream 1 (e.g. Kling-generated clips).
                cmd.extend(["-map", "0:v", "-map", "1:a", "-c:a", "aac", "-shortest"])
            else:
                cmd.extend(["-c:a", "copy"])

            cmd.append(str(output_path))
            self.run_command(cmd)

            data = {
                "operation": "compose",
                "cut_count": len(cuts),
                "has_subtitles": subtitle_path is not None,
                "has_mixed_audio": audio_path is not None,
                "profile": profile_name,
                "output": str(output_path),
            }
            # Surface ffprobe ground-truth so downstream consumers can use
            # probe-derived values instead of fabricating render_report
            # fields. See VideoCompose._probe_output_metadata.
            actual = self._probe_output_metadata(output_path)
            if actual is not None:
                data["actual_output"] = actual

            return ToolResult(
                success=True,
                data=data,
                artifacts=[str(output_path)],
            )
        finally:
            # Cleanup temp files
            for f in temp_segments:
                if f.exists():
                    f.unlink()
            for f in [concat_path, concat_out]:
                if f is not None and f.exists():
                    f.unlink()
            if temp_dir.exists():
                try:
                    temp_dir.rmdir()
                except OSError:
                    pass

    _REMOTION_SCENE_TYPES = {
        "text_card", "stat_card", "callout", "comparison", "progress", "chart",
    }

    # Maps renderer_family (set at proposal stage) to Remotion composition ID.
    # Each family MUST map to a distinct composition — collapsing defeats visual grammar.
    # Maps renderer_family → Remotion composition ID.
    # Only compositions registered in remotion-composer/src/Root.tsx are valid.
    # Current compositions: Explainer, CinematicRenderer, TalkingHead
    RENDERER_FAMILY_MAP = {
        "explainer-data": "Explainer",
        "explainer-teacher": "Explainer",
        "cinematic-trailer": "CinematicRenderer",
        "documentary-montage": "CinematicRenderer",
        "product-reveal": "Explainer",
        "screen-demo": "Explainer",
        "presenter": "TalkingHead",
        "animation-first": "Explainer",
    }

    @classmethod
    def _get_composition_id(cls, renderer_family: str) -> str:
        """Resolve renderer_family to Remotion composition ID.

        Raises ValueError if renderer_family is not recognized — the caller
        must set it at proposal stage.
        """
        comp = cls.RENDERER_FAMILY_MAP.get(renderer_family)
        if comp is None:
            raise ValueError(
                f"Unknown renderer_family {renderer_family!r}. "
                f"Valid families: {sorted(cls.RENDERER_FAMILY_MAP)}. "
                f"Set renderer_family at proposal stage."
            )
        return comp

    @staticmethod
    def _build_theme_from_playbook(
        playbook_name: str | None,
        composition_data: dict | None,
    ) -> dict[str, Any] | None:
        """Derive a Remotion ThemeConfig from a playbook's actual color values.

        Instead of passing a playbook name and hoping Remotion has a matching
        preset, we read the playbook YAML and extract concrete colors/fonts.
        This means custom playbooks, overridden palettes, and per-project
        styles all flow through to Remotion automatically.

        Falls back to extracting colors from edit_decisions metadata if
        no playbook is loadable.
        """
        theme: dict[str, Any] = {}

        # Try to load the playbook YAML
        playbook: dict[str, Any] = {}
        if playbook_name:
            try:
                from styles.playbook_loader import load_playbook
                playbook = load_playbook(playbook_name)
            except Exception:
                pass

        if playbook:
            vl = playbook.get("visual_language", {})
            palette = vl.get("color_palette", {})
            typo = playbook.get("typography", {})

            # Extract primary/accent — may be a list (gradient stops) or string
            primary_raw = palette.get("primary", ["#2563EB"])
            accent_raw = palette.get("accent", ["#F59E0B"])
            primary = primary_raw[0] if isinstance(primary_raw, list) else primary_raw
            accent = accent_raw[0] if isinstance(accent_raw, list) else accent_raw

            bg = palette.get("background", "#FFFFFF")
            text = palette.get("text", "#1F2937")
            surface = palette.get("surface", bg)
            muted = palette.get("muted_text", "#6B7280")

            # Build chart colors from all palette entries
            chart_colors = []
            for key in ["primary", "accent", "secondary", "success", "warning", "info"]:
                val = palette.get(key)
                if val:
                    chart_colors.append(val[0] if isinstance(val, list) else val)
            if len(chart_colors) < 3:
                chart_colors = [primary, accent, "#10B981", "#8B5CF6", "#EC4899", "#06B6D4"]

            theme = {
                "primaryColor": primary,
                "accentColor": accent,
                "backgroundColor": bg,
                "surfaceColor": surface,
                "textColor": text,
                "mutedTextColor": muted,
                "headingFont": typo.get("heading", {}).get("font", "Inter"),
                "bodyFont": typo.get("body", {}).get("font", "Inter"),
                "monoFont": typo.get("code", {}).get("font", "JetBrains Mono"),
                "chartColors": chart_colors[:6],
                "springConfig": {"damping": 20, "stiffness": 120, "mass": 1},
                "transitionDuration": 0.4,
            }

            # Derive caption colors from the palette
            theme["captionHighlightColor"] = primary
            # Caption background: semi-transparent version of the bg color
            theme["captionBackgroundColor"] = (
                f"rgba(255, 255, 255, 0.85)" if bg.upper() in ("#FFFFFF", "#FAFAFA", "#F9FAFB")
                else f"rgba(15, 23, 42, 0.75)"
            )

            # Motion style from playbook
            motion = playbook.get("motion", {})
            pace = motion.get("pace", "moderate")
            if pace == "fast":
                theme["springConfig"] = {"damping": 12, "stiffness": 80, "mass": 1}
                theme["transitionDuration"] = 0.3
            elif pace == "slow":
                theme["springConfig"] = {"damping": 25, "stiffness": 150, "mass": 1}
                theme["transitionDuration"] = 0.6

        # Fallback: try to extract from edit_decisions metadata
        if not theme and composition_data:
            meta = composition_data.get("metadata", {})
            if meta.get("primary_color"):
                theme = {
                    "primaryColor": meta["primary_color"],
                    "accentColor": meta.get("accent_color", "#F59E0B"),
                    "backgroundColor": meta.get("background_color", "#FFFFFF"),
                    "surfaceColor": meta.get("surface_color", "#F9FAFB"),
                    "textColor": meta.get("text_color", "#1F2937"),
                    "mutedTextColor": "#6B7280",
                    "headingFont": meta.get("heading_font", "Inter"),
                    "bodyFont": meta.get("body_font", "Inter"),
                    "monoFont": "JetBrains Mono",
                    "chartColors": meta.get("chart_colors", ["#2563EB", "#F59E0B", "#10B981"]),
                    "springConfig": {"damping": 20, "stiffness": 120, "mass": 1},
                    "transitionDuration": 0.4,
                    "captionHighlightColor": meta["primary_color"],
                    "captionBackgroundColor": "rgba(255, 255, 255, 0.85)",
                }

        return theme if theme else None

    _IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

    def _is_image_source(self, cut: dict, asset_manifest: Optional[dict] = None) -> bool:
        """Decide whether a cut's source is a still image rather than a
        video. Checks file extension first (covers every case where the
        source is already a resolved path), then consults the asset
        manifest's declared `type` as a fallback."""
        source = cut.get("source", "")
        if isinstance(source, str) and source:
            ext = Path(source).suffix.lower()
            if ext in self._IMAGE_EXTENSIONS:
                return True
        if asset_manifest:
            src = cut.get("source", "")
            for a in asset_manifest.get("assets", []):
                if a.get("id") == src or a.get("path") == src:
                    if a.get("type") == "image":
                        return True
                    break
        return False

    def _preprocess_image_cuts_to_mp4(
        self,
        cuts: list[dict],
        *,
        asset_manifest: Optional[dict],
        profile: Any,
        workspace_dir: Path,
    ) -> tuple[list[dict], Any]:
        """Rewrite image-source cuts to short MP4 loops generated via
        FFmpeg. Remotion's Rust compositor panics on mixed video+image
        cuts (`frame_cache.rs:257 Option::unwrap on None`), so the only
        safe path is to hand it video everywhere.

        Returns (rewritten_cuts, tempdir_handle). The caller MUST call
        `tempdir_handle.cleanup()` once the render finishes — generated
        MP4s live in that directory.

        If there are no image cuts, returns the original list and None
        so the caller skips cleanup entirely.
        """
        import tempfile

        image_cut_indices = [
            i for i, cut in enumerate(cuts)
            if self._is_image_source(cut, asset_manifest)
        ]
        if not image_cut_indices:
            return cuts, None

        # Resolve target canvas (width/height/fps) from the profile. Fall
        # back to 1920x1080@30 matching video_compose's default canvas so
        # the preprocessed video matches what Remotion will render into.
        width, height, fps = self._resolve_canvas_dims(profile)

        ffmpeg = self._which_ffmpeg()
        if ffmpeg is None:
            # No ffmpeg → can't preprocess. Leave the cuts alone and let
            # Remotion fail with the original panic; that's clearer than
            # silently dropping the image scenes.
            logging.getLogger(__name__).warning(
                "image_cuts_preprocess: ffmpeg not on PATH; image cuts will "
                "be handed to Remotion unchanged and likely crash the "
                "compositor. Install ffmpeg to enable preprocessing."
            )
            return cuts, None

        tempdir = tempfile.TemporaryDirectory(
            prefix="remotion_image_preprocess_", dir=str(workspace_dir)
        )
        tempdir_path = Path(tempdir.name)

        new_cuts = list(cuts)
        for idx in image_cut_indices:
            cut = dict(cuts[idx])
            image_path = cut.get("source", "")
            duration = self._cut_duration_seconds(cut)
            if duration <= 0:
                # Skip nonsensical durations — Remotion will reject the
                # zero-length sequence with a clearer error than we could
                # produce here.
                continue
            out_path = tempdir_path / f"image_cut_{idx:03d}.mp4"
            vf = (
                f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
                f"setsar=1"
            )
            cmd = [
                ffmpeg, "-y", "-loglevel", "error",
                "-loop", "1", "-i", str(image_path),
                "-t", f"{duration:.3f}",
                "-r", str(fps),
                "-vf", vf,
                "-pix_fmt", "yuv420p",
                "-c:v", "libx264",
                "-preset", "veryfast",
                str(out_path),
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
            except subprocess.CalledProcessError as e:
                logging.getLogger(__name__).warning(
                    "image_cuts_preprocess: ffmpeg failed for cut %s (%s): %s",
                    cut.get("id"), image_path, (e.stderr or "").strip()[-500:],
                )
                continue  # leave the cut pointing at the image; Remotion will surface the error
            cut["source"] = str(out_path)
            # source_in_seconds is meaningless for a synthetic image loop — drop it.
            cut.pop("source_in_seconds", None)
            new_cuts[idx] = cut

        return new_cuts, tempdir

    @staticmethod
    def _cut_duration_seconds(cut: dict) -> float:
        in_s = float(cut.get("in_seconds", 0) or 0)
        out_s = cut.get("out_seconds")
        if out_s is None:
            dur = float(cut.get("duration") or cut.get("duration_s") or 0)
            return max(dur, 0.0)
        return max(float(out_s) - in_s, 0.0)

    def _probe_video_orientation(self, path: str) -> Optional[str]:
        """ffprobe a source file and classify as 'portrait' / 'landscape'
        / 'square'. Returns None on probe failure — the caller treats
        unknown orientation as "don't contribute to the vote."""
        import shutil as _shutil
        ffprobe = _shutil.which("ffprobe")
        if ffprobe is None:
            return None
        try:
            proc = subprocess.run(
                [
                    ffprobe, "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=width,height",
                    "-of", "csv=p=0",
                    str(path),
                ],
                capture_output=True, text=True, timeout=10, check=True,
            )
            w_str, h_str = proc.stdout.strip().split(",")
            w, h = int(w_str), int(h_str)
        except Exception:
            return None
        # ffprobe sometimes exits 0 with "0,0" on malformed inputs (e.g.
        # a file with an invalid PNG signature). Treat that as
        # unknown orientation — not "square".
        if w <= 0 or h <= 0:
            return None
        if w == h:
            return "square"
        return "portrait" if h > w else "landscape"

    def _auto_detect_canvas_profile(
        self, resolved_cuts: list[dict]
    ) -> Optional[str]:
        """When the caller didn't pass a profile, infer canvas orientation
        from the primary-layer cuts' actual source dimensions. Picks the
        matching default profile for the dominant orientation.

        The alternative (current default — hard 1920×1080 landscape)
        silently scales portrait Veo output into a landscape canvas,
        which is exactly issue #8: the agent requested aspect_ratio=9:16,
        the clips came back portrait, and compose turned the final
        render landscape anyway because no one told it otherwise.

        Returns None on no clear signal (no primary-layer cuts,
        ffprobe unavailable, or mixed orientations without a majority),
        which preserves the historical 1920×1080 fallback — the
        least-surprise outcome when intent isn't inferable.
        """
        orientations: list[str] = []
        for cut in resolved_cuts:
            if cut.get("layer", "primary") != "primary":
                continue
            src = cut.get("source", "")
            if not src or not isinstance(src, str):
                continue
            if src.startswith(("http://", "https://")):
                continue  # skip remote; probing would stall
            o = self._probe_video_orientation(src)
            if o:
                orientations.append(o)

        if not orientations:
            return None

        # Dominant orientation → matching default profile. Tie → fall
        # through to None (preserve 1920×1080 fallback rather than
        # coin-flipping).
        counts = {
            "portrait": orientations.count("portrait"),
            "landscape": orientations.count("landscape"),
            "square": orientations.count("square"),
        }
        dominant, dominant_count = max(counts.items(), key=lambda kv: kv[1])
        total = sum(counts.values())
        if dominant_count * 2 <= total:
            return None  # no majority

        return {
            "portrait": "instagram_reels",  # 1080x1920 @ 30
            "landscape": "generic_hd",       # 1920x1080 @ 30
            "square": "instagram_feed",      # 1080x1080 @ 30
        }.get(dominant)

    @staticmethod
    def _probe_output_metadata(output_path: Path) -> Optional[dict]:
        """ffprobe a rendered output to produce ground-truth metadata.

        Returns a dict shaped to drop into ToolResult.data["actual_output"]
        so downstream consumers (compose-director skill, render_report
        writer) can use probe-derived values instead of director-
        fabricated self-reports.

        The director's pre-render plan tells you what the render WAS
        SUPPOSED to produce. The probe tells you what it ACTUALLY
        produced. Every false-success render bug shipped this fork
        (#26 30s-from-10s, #27 landscape-from-portrait, #28 black
        output, #30 compose landscape default) traced to the gap
        between those two. Surface the probe result so the gap is
        visible to the next layer up.

        Returns None on probe failure (ffprobe missing, output unreadable,
        malformed file). Callers should fail loud if None — a render
        that succeeded but can't be probed is itself a red flag.
        """
        import shutil as _shutil

        ffprobe = _shutil.which("ffprobe")
        if ffprobe is None:
            return None
        if not output_path.exists():
            return None

        try:
            proc = subprocess.run(
                [
                    ffprobe, "-v", "error",
                    "-print_format", "json",
                    "-show_format",
                    "-show_streams",
                    str(output_path),
                ],
                capture_output=True, text=True, timeout=15, check=True,
            )
            data = json.loads(proc.stdout)
        except Exception:
            return None

        result: dict[str, Any] = {
            "path": str(output_path),
            "file_size_bytes": int(
                (data.get("format") or {}).get("size") or 0
            ),
        }
        try:
            result["duration_seconds"] = round(
                float((data.get("format") or {}).get("duration") or 0), 3
            )
        except (TypeError, ValueError):
            result["duration_seconds"] = 0.0

        video_stream = next(
            (s for s in data.get("streams") or []
             if s.get("codec_type") == "video"),
            None,
        )
        audio_stream = next(
            (s for s in data.get("streams") or []
             if s.get("codec_type") == "audio"),
            None,
        )

        if video_stream:
            w = int(video_stream.get("width") or 0)
            h = int(video_stream.get("height") or 0)
            if w > 0 and h > 0:
                result["resolution"] = f"{w}x{h}"
                result["width"] = w
                result["height"] = h
            result["video_codec"] = video_stream.get("codec_name")
            # Compute fps from r_frame_rate fraction (e.g. "30000/1001").
            r_fps = video_stream.get("r_frame_rate") or "0/1"
            try:
                num, den = r_fps.split("/")
                if int(den) > 0:
                    result["fps"] = round(int(num) / int(den), 3)
            except (ValueError, ZeroDivisionError):
                pass
        else:
            result["video_codec"] = None
            result["resolution"] = None

        result["has_audio"] = audio_stream is not None
        if audio_stream:
            result["audio_codec"] = audio_stream.get("codec_name")

        return result

    @staticmethod
    def _resolve_cut_sources(
        cuts: list[dict],
        asset_manifest: Optional[dict],
    ) -> list[dict]:
        """Replace cuts[].source asset-IDs with their asset_manifest paths.

        Both _render and _compose need this resolution. Centralising it
        prevents drift like fork issues #29 (compose path stayed on
        literal-only treatment after render got the ID lookup) and any
        future operation that needs the same shape.

        Returns a list of NEW dicts (input cuts are not mutated). Sources
        that don't match an ID fall through unchanged — supports the
        backwards-compat case where the caller passes a literal path.
        """
        id_to_path: dict[str, str] = {}
        for asset in (asset_manifest or {}).get("assets") or []:
            if not isinstance(asset, dict):
                continue
            aid = asset.get("id")
            apath = asset.get("path")
            if isinstance(aid, str) and isinstance(apath, str):
                id_to_path[aid] = apath

        resolved: list[dict] = []
        for cut in cuts:
            new_cut = dict(cut) if isinstance(cut, dict) else {}
            src = cut.get("source") if isinstance(cut, dict) else None
            if isinstance(src, str) and src in id_to_path:
                new_cut["source"] = id_to_path[src]
            resolved.append(new_cut)
        return resolved

    @staticmethod
    def _resolve_canvas_dims(profile: Any) -> tuple[int, int, int]:
        """Return (width, height, fps) from a profile input. Accepts
        named profiles from media_profiles.py, explicit
        {width, height, fps} dicts, or None (1920x1080 @ 30 default)."""
        if isinstance(profile, dict):
            w = int(profile.get("width") or 1920)
            h = int(profile.get("height") or 1080)
            fps = int(profile.get("fps") or 30)
            return w, h, fps
        if isinstance(profile, str) and profile:
            try:
                from lib.media_profiles import get_profile
                p = get_profile(profile)
                return p.width, p.height, p.fps
            except (ImportError, ValueError):
                pass
        return 1920, 1080, 30

    @staticmethod
    def _which_ffmpeg() -> Optional[str]:
        import shutil as _shutil
        return _shutil.which("ffmpeg")

    def _needs_remotion(self, cuts: list[dict]) -> bool:
        """Determine whether Remotion should handle this composition.

        Remotion is the DEFAULT composition engine when available.  It handles
        video clips (via <OffthreadVideo>), still images, animated scene types,
        component types, transitions, and mixed content — all in a single
        React-based render pass.

        Returns False (i.e. use FFmpeg) only when Remotion is not
        available. For `operation="render"` the governance default is
        Remotion-first: the renderer family was chosen earlier, and the
        tool should preserve that decision instead of silently
        downgrading to FFmpeg.

        This "Remotion-first" policy means mixed content (video clips +
        animated stills + text cards) is always composed in Remotion, which
        can embed <OffthreadVideo> alongside React components natively.
        """
        # If Remotion isn't installed, fall back to FFmpeg
        if not self._remotion_available():
            return False

        # Any rich content → Remotion (fast path, catches the obvious cases)
        for cut in cuts:
            source = cut.get("source", "")
            if source and Path(source).suffix.lower() in self._IMAGE_EXTENSIONS:
                return True
            if cut.get("type") in self._REMOTION_SCENE_TYPES:
                return True
            if cut.get("animation") or cut.get("transition_in") or cut.get("transition_out"):
                return True
            transform = cut.get("transform", {})
            if transform and transform.get("animation"):
                return True

        # Even for pure-video cuts, default to Remotion — it handles video
        # clips natively via <OffthreadVideo> and gives us transitions,
        # overlays, and profile scaling for free.
        return True

    def _pre_compose_validation(
        self,
        edit_decisions: dict[str, Any],
        resolved_cuts: list[dict],
        scene_plan: list[dict] | None = None,
        proposal_packet: dict[str, Any] | None = None,
        asset_manifest: dict[str, Any] | None = None,
        script: dict[str, Any] | None = None,
    ) -> ToolResult | None:
        """Pre-compose quality gate — blocks render on critical violations.

        Checks:
        1. Delivery promise violation: motion-required brief with >70% still cuts → BLOCK
        2. Slideshow risk score "fail" (average ≥ 4.0) → BLOCK
        3. Missing renderer_family → WARN (log only, don't block)
        4. voice_led / dialogue_led narration coverage — when audio_treatment
           requires spoken audio, every non-empty script.section.id must
           appear as the scene_id of at least one type:"narration" asset.
           `script` is REQUIRED for these modes; without it the validator
           can't compute coverage and falls back to the weaker "≥1 narration
           asset" check (legacy behavior, less safe).

        Returns a failed ToolResult if render should be blocked, None if OK to proceed.
        """
        log = logging.getLogger("video_compose")
        warnings: list[str] = []
        blocks: list[str] = []

        # --- 1. Delivery promise check ---
        delivery_data = edit_decisions.get("metadata", {}).get("delivery_promise")
        if not delivery_data:
            # Also check top-level (proposal_packet nests it at top level)
            delivery_data = edit_decisions.get("delivery_promise")

        if delivery_data:
            try:
                from lib.delivery_promise import DeliveryPromise
                promise = DeliveryPromise.from_dict(delivery_data)
                result = promise.validate_cuts(resolved_cuts)
                if not result["valid"]:
                    for v in result["violations"]:
                        blocks.append(f"Delivery promise violation: {v}")
            except Exception as e:
                log.warning("Could not validate delivery promise: %s", e)
        else:
            warnings.append("No delivery_promise in edit_decisions — skipping promise validation")

        # --- 2. Slideshow risk check ---
        renderer_family = edit_decisions.get("renderer_family")
        scenes = scene_plan or []

        # If no scene_plan passed, try to extract scene info from cuts
        if not scenes and resolved_cuts:
            scenes = [
                {
                    "type": c.get("type", ""),
                    "description": c.get("reason", ""),
                    "shot_language": c.get("shot_language", {}),
                    "shot_intent": c.get("shot_intent"),
                    "narrative_role": c.get("narrative_role"),
                    "information_role": c.get("information_role"),
                    "hero_moment": c.get("hero_moment", False),
                }
                for c in resolved_cuts
            ]

        if scenes:
            try:
                from lib.slideshow_risk import score_slideshow_risk
                render_runtime = edit_decisions.get("render_runtime")
                risk = score_slideshow_risk(
                    scenes, edit_decisions, renderer_family, render_runtime
                )
                if risk["verdict"] == "fail":
                    blocks.append(
                        f"Slideshow risk score {risk['average']:.1f}/5.0 (verdict: fail). "
                        f"Video plan looks like a slideshow — revise scene plan before rendering."
                    )
                elif risk["verdict"] == "revise":
                    warnings.append(
                        f"Slideshow risk score {risk['average']:.1f}/5.0 (verdict: revise). "
                        f"Consider improving scene variety before final render."
                    )
            except Exception as e:
                log.warning("Could not compute slideshow risk: %s", e)

        # --- 3. Missing renderer_family (BLOCK — must be set at proposal) ---
        if not renderer_family:
            blocks.append(
                "No renderer_family in edit_decisions. "
                "renderer_family must be set at proposal stage and locked before compose. "
                "Re-run the proposal stage with a renderer_family selection."
            )

        # --- 4. Spoken-audio narration coverage (fork issues #22 / #23) ---
        # When audio_treatment is voice_led or dialogue_led, every
        # non-empty script section must be covered by at least one
        # narration asset. Without this guard, an asset stage that
        # silently skips the TTS loop produces a video with valid
        # AAC track at -91 dB ("technically successful" silent ship).
        if proposal_packet and asset_manifest:
            plan = proposal_packet.get("production_plan") or {}
            audio_treatment = plan.get("audio_treatment") or {}
            mode = audio_treatment.get("mode")
            if mode in {"voice_led", "dialogue_led"}:
                narration_count = sum(
                    1
                    for a in (asset_manifest.get("assets") or [])
                    if isinstance(a, dict) and a.get("type") == "narration"
                )

                if isinstance(script, dict):
                    # Strong form: per-section coverage by scene_id set.
                    sections = script.get("sections") or []
                    required_section_ids = [
                        s.get("id")
                        for s in sections
                        if isinstance(s, dict)
                        and isinstance(s.get("text"), str)
                        and s["text"].strip()
                        and isinstance(s.get("id"), str)
                        and s["id"]
                    ]
                    covered_section_ids = {
                        a.get("scene_id")
                        for a in (asset_manifest.get("assets") or [])
                        if isinstance(a, dict)
                        and a.get("type") == "narration"
                        and isinstance(a.get("scene_id"), str)
                        and a["scene_id"]
                    }
                    if required_section_ids:
                        missing = [
                            sid
                            for sid in required_section_ids
                            if sid not in covered_section_ids
                        ]
                        if missing:
                            issue_tag = "#22" if not covered_section_ids else "#23"
                            blocks.append(
                                f"audio_treatment.mode={mode!r} but "
                                f"asset_manifest is missing narration "
                                f"coverage for script section(s): "
                                f"{sorted(missing)}. Required sections: "
                                f"{sorted(required_section_ids)}. Covered "
                                f"scene_ids on narration assets: "
                                f"{sorted(covered_section_ids) or '[]'}. "
                                f"Per-section coverage (by scene_id set, "
                                f"not count) is required. (fork issue "
                                f"{issue_tag})."
                            )
                elif narration_count == 0:
                    # Weak form: script not provided, fall back to >=1.
                    blocks.append(
                        f"audio_treatment.mode={mode!r} but asset_manifest "
                        f"contains zero `type: \"narration\"` assets. "
                        f"For voice_led the asset stage must call the "
                        f"resolved TTS per script.sections[]; for "
                        f"dialogue_led it must extract source audio. "
                        f"Rendering would produce a silent video. Pass "
                        f"`script` to video_compose for per-section "
                        f"coverage diagnostics. (fork issue #22)."
                    )

        # Log warnings
        for w in warnings:
            log.warning("[pre-compose] %s", w)

        # Block on critical violations
        if blocks:
            return ToolResult(
                success=False,
                error=(
                    "Pre-compose validation failed — render blocked.\n"
                    + "\n".join(f"  • {b}" for b in blocks)
                    + ("\n\nWarnings:\n" + "\n".join(f"  • {w}" for w in warnings) if warnings else "")
                ),
            )

        return None

    def _render(self, inputs: dict[str, Any]) -> ToolResult:
        """High-level render: assemble edit decisions + asset manifest into final video.

        This is the primary entry point for the compose-director skill.
        It resolves asset IDs and routes to the composition engine:

        - **Remotion (default):** Used for all compositions when available —
          video clips, images, animated scenes, component types, mixed content.
          Remotion embeds video via <OffthreadVideo> and handles transitions,
          overlays, and profile scaling natively.
        - **FFmpeg (fallback):** Used only when Remotion is unavailable, or
          when the agent explicitly calls operation='compose' for simple
          trim/concat operations.

        The agent should pass edit_decisions, asset_manifest, and optionally
        profile, subtitle_path, audio_path, and options.
        """
        edit_decisions = inputs.get("edit_decisions")
        asset_manifest = inputs.get("asset_manifest")
        if not edit_decisions:
            return ToolResult(success=False, error="edit_decisions required for render")
        if not asset_manifest:
            return ToolResult(success=False, error="asset_manifest required for render")

        output_path = Path(inputs.get("output_path", "renders/output.mp4"))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cuts = edit_decisions.get("cuts", [])
        if not cuts:
            return ToolResult(success=False, error="No cuts in edit_decisions")

        # Resolve asset IDs in cuts to file paths via the shared helper
        # used by both _render and _compose (see fork issue #29 — these
        # paths drifted apart and only render had the lookup).
        resolved_cuts = self._resolve_cut_sources(cuts, asset_manifest)

        # --- Pre-compose validation gate ---
        scene_plan = inputs.get("scene_plan")
        validation_block = self._pre_compose_validation(
            edit_decisions,
            resolved_cuts,
            scene_plan,
            proposal_packet=inputs.get("proposal_packet"),
            asset_manifest=asset_manifest,
            script=inputs.get("script"),
        )
        if validation_block is not None:
            return validation_block

        # Also accept profile as "output_profile" (skill convention) or "profile"
        profile = inputs.get("profile") or inputs.get("output_profile")

        # When no profile is passed, infer canvas orientation from the
        # primary-layer sources. This closes issue #8: the agent passes
        # aspect_ratio=9:16 to video_selector and Veo produces portrait
        # clips, but compose used to hard-default to 1920×1080 and scale
        # the portrait source into a landscape canvas. Auto-detect picks
        # the matching default profile (instagram_reels / generic_hd /
        # instagram_feed) based on the dominant source orientation.
        # Explicit `profile` still overrides — no behavior change for
        # callers that already pin the canvas.
        if not profile:
            auto = self._auto_detect_canvas_profile(resolved_cuts)
            if auto:
                profile = auto

        # --- Runtime routing: honor render_runtime locked at proposal ---
        # Silent swaps are forbidden by governance. If the chosen runtime
        # is unavailable, surface a structured blocker rather than quietly
        # picking a different engine. Missing render_runtime is itself a
        # governance violation — edit_decisions.schema.json requires it.
        render_runtime = (edit_decisions.get("render_runtime") or "").strip().lower()

        if not render_runtime:
            return ToolResult(
                success=False,
                error=(
                    "render_runtime is not set in edit_decisions. Per governance, "
                    "it MUST be locked at proposal stage (proposal_packet."
                    "production_plan.render_runtime) and carried forward through "
                    "edit_decisions.render_runtime. Valid values: 'remotion', "
                    "'hyperframes', 'ffmpeg'. Re-run the proposal stage with an "
                    "explicit runtime choice — do NOT default this field."
                ),
            )

        if render_runtime == "hyperframes":
            return self._render_via_hyperframes(
                inputs=inputs,
                edit_decisions=edit_decisions,
                asset_manifest=asset_manifest,
                resolved_cuts=resolved_cuts,
                output_path=output_path,
                profile=profile,
            )
        if render_runtime == "ffmpeg":
            # Caller explicitly asked for FFmpeg — don't auto-upgrade to Remotion.
            return self._render_via_ffmpeg(
                inputs=inputs,
                edit_decisions=edit_decisions,
                resolved_cuts=resolved_cuts,
                output_path=output_path,
                profile=profile,
            )
        if render_runtime != "remotion":
            return ToolResult(
                success=False,
                error=(
                    f"Unknown render_runtime {render_runtime!r}. "
                    f"Valid values: remotion, hyperframes, ffmpeg. "
                    f"render_runtime must be set at proposal stage."
                ),
            )

        # --- Explicit Remotion path (render_runtime == 'remotion') ---
        if self._needs_remotion(resolved_cuts):
            # Image cuts (still .png/.jpg/etc in cuts[].source) crash the
            # Remotion Rust compositor with
            #   frame_cache.rs:257 called `Option::unwrap()` on a `None` value
            # when mixed with video cuts. Preprocess each image cut into a
            # short MP4 loop at the target fps/resolution so Remotion sees
            # only video sources. Temp files are cleaned up after render.
            resolved_cuts, image_preprocess_tempdir = self._preprocess_image_cuts_to_mp4(
                resolved_cuts,
                asset_manifest=asset_manifest,
                profile=profile,
                workspace_dir=output_path.parent,
            )
            remotion_inputs: dict[str, Any] = {
                "edit_decisions": dict(edit_decisions, cuts=resolved_cuts),
                "output_path": str(output_path),
            }
            if profile:
                remotion_inputs["profile"] = profile
            try:
                render_result = self._remotion_render(remotion_inputs)
            finally:
                if image_preprocess_tempdir is not None:
                    try:
                        image_preprocess_tempdir.cleanup()
                    except Exception:
                        pass  # best-effort cleanup

            # Governance: NEVER silently fall back to FFmpeg when Remotion fails.
            # The agent must decide the fallback path, not the tool.
            if not render_result.success:
                renderer_family = edit_decisions.get("renderer_family", "unknown")
                return ToolResult(
                    success=False,
                    error=(
                        f"Remotion render failed for renderer_family={renderer_family!r}. "
                        f"Underlying error: {render_result.error}\n\n"
                        f"This composition requires Remotion (images, text cards, animations). "
                        f"Options:\n"
                        f"  1. Fix Remotion setup (cd remotion-composer && npm install)\n"
                        f"  2. Re-run with operation='compose' for FFmpeg-only (video cuts only)\n"
                        f"  3. Approve a degraded FFmpeg render (still images → Ken Burns)\n\n"
                        f"Per governance: renderer downgrade requires user approval."
                    ),
                )
        else:
            # --- FFmpeg fallback: only when Remotion is unavailable ---
            options = inputs.get("options", {})
            subtitle_burn = options.get("subtitle_burn", True)

            # Resolve subtitle_path from edit_decisions if not provided
            subtitle_path = inputs.get("subtitle_path")
            if subtitle_burn and not subtitle_path:
                ed_subs = edit_decisions.get("subtitles", {})
                if ed_subs.get("enabled") and ed_subs.get("source"):
                    subtitle_path = ed_subs["source"]

            # Build compose inputs
            compose_inputs = dict(inputs)
            compose_inputs["edit_decisions"] = dict(edit_decisions, cuts=resolved_cuts)
            compose_inputs["output_path"] = str(output_path)
            if subtitle_path:
                compose_inputs["subtitle_path"] = subtitle_path
            if profile:
                compose_inputs["profile"] = profile

            render_result = self._compose(compose_inputs)

        # --- Post-render: mandatory final self-review ---
        if render_result.success and output_path.exists():
            final_review = self._run_final_review(
                output_path,
                edit_decisions,
                inputs.get("proposal_packet"),
                narration_transcript_path=inputs.get("narration_transcript_path"),
                script_text=inputs.get("script_text") or self._read_text_file(
                    inputs.get("script_path")
                ),
            )

            # Attach final_review to the ToolResult data so the compose-director
            # skill can include it in the checkpoint alongside the render_report.
            if render_result.data is None:
                render_result.data = {}
            render_result.data["final_review"] = final_review
            render_result.data["final_review_status"] = final_review["status"]

            # If the self-review says fail, downgrade the ToolResult
            if final_review["status"] == "fail":
                return ToolResult(
                    success=False,
                    error=(
                        "Post-render self-review FAILED. The output is not presentable.\n"
                        + "\n".join(f"  • {i}" for i in final_review.get("issues_found", []))
                    ),
                    data=render_result.data,
                )

        return render_result

    def _render_via_hyperframes(
        self,
        *,
        inputs: dict[str, Any],
        edit_decisions: dict[str, Any],
        asset_manifest: dict[str, Any],
        resolved_cuts: list[dict],
        output_path: Path,
        profile: Optional[str],
    ) -> ToolResult:
        """Delegate to hyperframes_compose and run the mandatory final self-review.

        Governance: if HyperFrames is unavailable or fails, return a structured
        blocker — do NOT silently route to Remotion or FFmpeg. The agent must
        surface the blocker and get user approval before any runtime swap.
        """
        if not self._hyperframes_available():
            return ToolResult(
                success=False,
                error=(
                    "render_runtime='hyperframes' was locked at proposal, but "
                    "the HyperFrames runtime is not available on this machine. "
                    "Per governance this is a BLOCKER — surface it to the user "
                    "per AGENT_GUIDE.md > 'Escalate Blockers Explicitly' and wait "
                    "for approval before switching runtime. Requirements: "
                    "Node.js >= 22, FFmpeg, and npx on PATH. See "
                    "tools/video/hyperframes_compose.py for the specific missing piece."
                ),
            )

        try:
            from tools.video.hyperframes_compose import HyperFramesCompose
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Could not import hyperframes_compose: {e}",
            )

        workspace_path = (
            inputs.get("workspace_path")
            or str(output_path.parent.parent / "hyperframes")
        )

        # Pass the playbook through so the style bridge can emit CSS vars.
        playbook_data = inputs.get("playbook")
        if not playbook_data:
            playbook_name = (
                inputs.get("playbook_name")
                or (edit_decisions.get("metadata") or {}).get("playbook")
            )
            if playbook_name:
                try:
                    from styles.playbook_loader import load_playbook  # type: ignore
                    playbook_data = load_playbook(playbook_name)
                except Exception:
                    playbook_data = None

        hf_inputs: dict[str, Any] = {
            "operation": "render",
            "workspace_path": workspace_path,
            "output_path": str(output_path),
            "edit_decisions": dict(edit_decisions, cuts=resolved_cuts),
            "asset_manifest": asset_manifest,
        }
        if playbook_data:
            hf_inputs["playbook"] = playbook_data
        if profile:
            hf_inputs["profile"] = profile
        if "quality" in inputs:
            hf_inputs["quality"] = inputs["quality"]
        if "fps" in inputs:
            hf_inputs["fps"] = inputs["fps"]
        if "strict" in inputs:
            hf_inputs["strict"] = inputs["strict"]
        if "skip_contrast" in inputs:
            hf_inputs["skip_contrast"] = inputs["skip_contrast"]

        render_result = HyperFramesCompose().execute(hf_inputs)

        if not render_result.success:
            return ToolResult(
                success=False,
                error=(
                    f"HyperFrames render failed: {render_result.error}. "
                    "Per governance: do NOT silently fall back to Remotion or "
                    "FFmpeg. Surface the failure to the user along with the "
                    "hyperframes_compose step log before proposing a swap."
                ),
                data=render_result.data,
            )

        # Post-render: mandatory final self-review (identical contract to the Remotion path).
        if output_path.exists():
            final_review = self._run_final_review(
                output_path,
                edit_decisions,
                inputs.get("proposal_packet"),
                narration_transcript_path=inputs.get("narration_transcript_path"),
                script_text=inputs.get("script_text") or self._read_text_file(
                    inputs.get("script_path")
                ),
            )
            if render_result.data is None:
                render_result.data = {}
            render_result.data["final_review"] = final_review
            render_result.data["final_review_status"] = final_review["status"]
            if final_review["status"] == "fail":
                return ToolResult(
                    success=False,
                    error=(
                        "Post-render self-review FAILED (HyperFrames). The output is not presentable.\n"
                        + "\n".join(f"  • {i}" for i in final_review.get("issues_found", []))
                    ),
                    data=render_result.data,
                )

        return render_result

    def _render_via_ffmpeg(
        self,
        *,
        inputs: dict[str, Any],
        edit_decisions: dict[str, Any],
        resolved_cuts: list[dict],
        output_path: Path,
        profile: Optional[str],
    ) -> ToolResult:
        """Explicit FFmpeg-only render path.

        Use when the proposal locked `render_runtime="ffmpeg"` — e.g. simple
        source-footage concat/trim jobs that don't benefit from composition.
        Still runs the mandatory final self-review.
        """
        options = inputs.get("options", {})
        subtitle_burn = options.get("subtitle_burn", True)

        subtitle_path = inputs.get("subtitle_path")
        if subtitle_burn and not subtitle_path:
            ed_subs = edit_decisions.get("subtitles", {})
            if ed_subs.get("enabled") and ed_subs.get("source"):
                subtitle_path = ed_subs["source"]

        compose_inputs = dict(inputs)
        compose_inputs["edit_decisions"] = dict(edit_decisions, cuts=resolved_cuts)
        compose_inputs["output_path"] = str(output_path)
        if subtitle_path:
            compose_inputs["subtitle_path"] = subtitle_path
        if profile:
            compose_inputs["profile"] = profile

        render_result = self._compose(compose_inputs)

        if render_result.success and output_path.exists():
            final_review = self._run_final_review(
                output_path,
                edit_decisions,
                inputs.get("proposal_packet"),
                narration_transcript_path=inputs.get("narration_transcript_path"),
                script_text=inputs.get("script_text") or self._read_text_file(
                    inputs.get("script_path")
                ),
            )
            if render_result.data is None:
                render_result.data = {}
            render_result.data["final_review"] = final_review
            render_result.data["final_review_status"] = final_review["status"]
            if final_review["status"] == "fail":
                return ToolResult(
                    success=False,
                    error=(
                        "Post-render self-review FAILED (FFmpeg). The output is not presentable.\n"
                        + "\n".join(f"  • {i}" for i in final_review.get("issues_found", []))
                    ),
                    data=render_result.data,
                )

        return render_result

    def _remotion_render(self, inputs: dict[str, Any]) -> ToolResult:
        """Render via Remotion (requires Node.js + npx).

        Handles compositions with still images, animated scenes, component
        types, and transitions using React-based frame-accurate rendering.
        Accepts edit_decisions (with resolved file paths) or raw composition_data.
        """
        import shutil

        if not shutil.which("npx"):
            return ToolResult(
                success=False,
                error="npx not found. Install Node.js to use Remotion rendering.",
            )

        composition_data = inputs.get("edit_decisions") or inputs.get("composition_data")
        if not composition_data:
            return ToolResult(
                success=False,
                error="edit_decisions or composition_data required for remotion_render",
            )

        output_path = Path(inputs.get("output_path", "renders/remotion_output.mp4"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Absolutise so the CLI can resolve the output regardless of cwd.
        output_path = output_path.resolve()

        # Deep-copy props so we don't mutate the original
        props = json.loads(json.dumps(composition_data))

        # Resolve the target composition early so we can remap props to the
        # shape that composition expects. Without this, Explainer-shaped
        # `cuts[]` data handed to CinematicRenderer (which reads `scenes[]`)
        # silently renders pure black — the scenes prop defaults to [].
        renderer_family = (composition_data or {}).get("renderer_family", "explainer-data")
        composition_id = self._get_composition_id(renderer_family)

        # Adapter: the agent emits a single `cuts[]` shape across every
        # renderer_family, but CinematicRenderer reads props.scenes[] with a
        # different field layout. Translate cuts → scenes when we're
        # targeting CinematicRenderer and the caller didn't already supply
        # scenes. This keeps the edit-director's output schema uniform
        # (edit_decisions.cuts) without hard-coding composition-specific
        # shapes into the agent's skill prose.
        if composition_id == "CinematicRenderer" and "scenes" not in props:
            cursor = 0.0
            scenes: list[dict[str, Any]] = []
            for idx, cut in enumerate(props.get("cuts", [])):
                in_s = float(cut.get("in_seconds") or cut.get("source_in_seconds") or 0)
                out_s = cut.get("out_seconds")
                if out_s is None:
                    duration = float(cut.get("duration") or cut.get("duration_s") or 0)
                else:
                    duration = max(float(out_s) - in_s, 0.0)
                scene: dict[str, Any] = {
                    "id": cut.get("id") or f"scene-{idx}",
                    "kind": "video",
                    "src": cut.get("source") or cut.get("src") or "",
                    "startSeconds": round(cursor, 3),
                    "durationSeconds": round(duration, 3) if duration > 0 else 0,
                }
                if in_s > 0:
                    scene["trimBeforeSeconds"] = in_s
                for hint_src, hint_dst in (
                    ("tone", "tone"),
                    ("filter", "filter"),
                    ("fade_in_frames", "fadeInFrames"),
                    ("fade_out_frames", "fadeOutFrames"),
                ):
                    if hint_src in cut:
                        scene[hint_dst] = cut[hint_src]
                scenes.append(scene)
                cursor += float(scene["durationSeconds"] or 0)
            props["scenes"] = scenes

        # Collect every local asset path referenced by the composition so we
        # can both (a) derive a single --public-dir Remotion will serve them
        # from and (b) rewrite absolute paths in the props to filenames
        # relative to that public-dir.
        #
        # Why this dance: Remotion's <OffthreadVideo> / <Img> do NOT load
        # `file://` URIs. The CLI stands up an internal HTTP server over
        # `--public-dir` (default: the bundle's `public/` folder), and any
        # non-URL string we hand the component is resolved against it.
        # Previously we converted absolute paths to `file:///...` URIs,
        # which Remotion reinterpreted as `http://localhost:3000/public/...`
        # and returned 404 — renders silently failed every time.
        local_asset_paths: list[tuple[dict[str, Any], str, Path]] = []

        def _collect_local_asset(obj: dict[str, Any], field: str) -> None:
            source = obj.get(field, "")
            if not source or not isinstance(source, str):
                return
            if source.startswith(("http://", "https://", "file://")):
                return
            resolved = Path(source)
            if not resolved.is_absolute():
                return  # already filename-only — will resolve under public-dir
            resolved = resolved.resolve()
            if resolved.exists() and resolved.is_file():
                local_asset_paths.append((obj, field, resolved))

        for cut in props.get("cuts", []):
            _collect_local_asset(cut, "source")
        for scene in props.get("scenes", []):
            _collect_local_asset(scene, "src")

        # Pick a public-dir that covers every asset. Commonly all assets and
        # the output share a workspace root: /workspace/{clip.mp4,title.png}
        # with output at /workspace/renders/final.mp4. Use os.path.commonpath
        # across every asset path so an even deeper nesting still works.
        #
        # Assumes every collected path is within a single workspace root.
        # commonpath raises ValueError on mixed roots (Windows C:\ vs D:\,
        # or cross-workspace asset_manifest entries); we catch and skip
        # --public-dir rather than crashing the render — Remotion will then
        # serve its bundle's default public/ folder and the absent assets
        # will 404 with a clear error message surfaced via the stderr tail.
        public_dir: Path | None = None
        if local_asset_paths:
            try:
                common = os.path.commonpath([str(p) for _, _, p in local_asset_paths])
            except ValueError:
                common = ""
            if common:
                common_path = Path(common)
                # commonpath may return a file path if there's only one asset;
                # use its parent so the CLI serves a directory.
                public_dir = common_path if common_path.is_dir() else common_path.parent

                # Rewrite each collected path to its position under public-dir.
                for obj, field, resolved in local_asset_paths:
                    try:
                        rel = resolved.relative_to(public_dir)
                        obj[field] = rel.as_posix()
                    except ValueError:
                        # Shouldn't happen given commonpath, but leave the raw
                        # string if it does rather than emitting a broken URI.
                        obj[field] = str(resolved)

        # Build a custom themeConfig from the playbook's actual colors.
        # This ensures every video gets a unique visual identity derived
        # from its production decisions — not picked from a preset menu.
        if "themeConfig" not in props:
            playbook_name = (
                props.get("playbook")
                or props.get("theme")
                or props.get("metadata", {}).get("playbook")
            )
            theme_config = self._build_theme_from_playbook(playbook_name, composition_data)
            if theme_config:
                props["themeConfig"] = theme_config

        # Write props to temp file for Remotion CLI
        props_path = output_path.parent / ".remotion_props.json"
        with open(props_path, "w", encoding="utf-8") as f:
            json.dump(props, f)

        # remotion-composer lives at project root
        composer_dir = Path(__file__).resolve().parent.parent.parent / "remotion-composer"
        if not composer_dir.exists():
            return ToolResult(
                success=False,
                error=f"Remotion composer project not found at {composer_dir}",
            )

        # composition_id is already resolved above (pre-props-rewrite) so the
        # scenes/cuts adapter could target the right shape. Reuse it here.
        cmd = [
            "npx", "remotion", "render",
            str(composer_dir / "src" / "index.tsx"),
            composition_id,
            str(output_path),
            "--props", str(props_path),
        ]
        # Serve the collected-assets root via Remotion's static server so the
        # filename-rewritten props above resolve. Without this flag, Remotion
        # serves its bundle's built-in `public/` folder, which doesn't
        # contain workspace clips or images — every asset 404s and the
        # render exits non-zero.
        if public_dir is not None:
            cmd.extend(["--public-dir", str(public_dir)])

        # Apply media profile dimensions. `profile` is documented as a string
        # name (e.g. "youtube_landscape"), but LLM-driven callers sometimes
        # pass a dict like {"width": 1280, "height": 720, "fps": 30}. Accept
        # both rather than erroring with TypeError: unhashable type: 'dict'.
        profile_input = inputs.get("profile")
        if isinstance(profile_input, dict):
            w = profile_input.get("width")
            h = profile_input.get("height")
            if w and h:
                cmd.extend(["--width", str(w), "--height", str(h)])
        elif isinstance(profile_input, str) and profile_input:
            try:
                from lib.media_profiles import get_profile
                p = get_profile(profile_input)
                cmd.extend(["--width", str(p.width), "--height", str(p.height)])
            except (ImportError, ValueError):
                pass

        try:
            # Invoke from inside the composer dir so npx can resolve the
            # local remotion binary via node_modules/.bin. Without this,
            # Windows npx cannot locate the CLI and returns "could not
            # determine executable to run".
            self.run_command(cmd, timeout=600, cwd=composer_dir)
        except subprocess.CalledProcessError as e:
            # stderr carries the real error (asset 404, missing prop, etc.).
            # Surface it instead of letting str(e) reduce everything to the
            # useless "returned non-zero exit status 1" line.
            stderr_tail = (e.stderr or "").strip()[-2000:] if e.stderr else ""
            detail = stderr_tail or str(e)
            return ToolResult(success=False, error=f"Remotion render failed: {detail}")
        except Exception as e:
            return ToolResult(success=False, error=f"Remotion render failed: {e}")
        finally:
            if props_path.exists():
                props_path.unlink()

        if not output_path.exists():
            return ToolResult(
                success=False,
                error=f"Remotion render completed but output file missing: {output_path}",
            )

        data: dict[str, Any] = {
            "operation": "remotion_render",
            "output": str(output_path),
            "profile": profile_input,
        }
        actual = self._probe_output_metadata(output_path)
        if actual is not None:
            data["actual_output"] = actual

        return ToolResult(
            success=True,
            data=data,
            artifacts=[str(output_path)],
        )

    # ------------------------------------------------------------------
    # Final self-review — mandatory post-render inspection
    # ------------------------------------------------------------------

    # Punctuation/SSML-leak words that should NEVER appear in rendered audio.
    # When a TTS engine reads a literal "..." as the word "dot", or a "—" as
    # "hyphen", those leak into the transcript. Catching these in the final
    # review is the difference between catching a bad voice render in-tool
    # vs. shipping a video that says "dot dot dot" twelve times. CRITICAL.
    _TTS_PUNCTUATION_LEAK_WORDS = {
        "dot", "dots", "ellipsis", "period", "periods",
        "comma", "commas", "semicolon", "colon",
        "dash", "hyphen", "emdash", "endash",
        "parenthesis", "bracket", "brace",
        "asterisk", "slash", "backslash",
        "exclamation", "question mark",
    }

    @staticmethod
    def _read_text_file(path: str | Path | None) -> str | None:
        """Read a small text file if given a path; None-safe and exception-safe."""
        if not path:
            return None
        try:
            return Path(path).read_text(encoding="utf-8")
        except Exception:
            return None

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        """Split text into comparable word tokens (lowercased, punctuation
        stripped, numeric-word-aware). Empty tokens dropped."""
        import re

        # Preserve hyphenated words as single tokens ("many-worlds" -> "many-worlds").
        # Drop everything except letters, digits, hyphens, apostrophes.
        cleaned = re.sub(r"[^A-Za-z0-9\-' ]+", " ", text.lower())
        return [t for t in cleaned.split() if t and t != "-"]

    @classmethod
    def _compare_transcript_to_script(
        cls,
        transcript_path: Path,
        script_text: str,
    ) -> dict[str, Any]:
        """Compare a word-level transcript against the source script.

        Purpose: catch TTS failures that look fine on audio-volume/duration
        checks but produce garbage content. The canonical example is
        Chirp3-HD reading ellipses ("...") literally as the word "dot" — our
        volume check says "narration present, not clipped" and the video
        ships. This check diffs the actual transcribed audio against what
        was supposed to be said, and flags:

        - Spurious punctuation-leak words ("dot", "comma", "hyphen", etc.)
          that appear in audio but not script → CRITICAL
        - Overall word-accuracy ratio against script → SUGGESTION if < 0.9

        Returns the transcript_comparison section of final_review, or a
        placeholder with an issue describing why the check couldn't run
        (missing transcript, missing script) so the review never goes
        silently quiet on this contract.
        """
        result: dict[str, Any] = {
            "transcript_matches_script": False,
            "word_accuracy": None,
            "script_word_count": 0,
            "transcript_word_count": 0,
            "spurious_punctuation_words": [],
            "issues": [],
        }

        if not transcript_path or not Path(transcript_path).is_file():
            result["issues"].append(
                "transcript_comparison skipped: narration_transcript not provided"
            )
            return result
        if not script_text:
            result["issues"].append(
                "transcript_comparison skipped: script_text not provided"
            )
            return result

        try:
            transcript_data = json.loads(Path(transcript_path).read_text(encoding="utf-8"))
        except Exception as e:
            result["issues"].append(f"transcript_comparison could not parse transcript: {e}")
            return result

        transcript_words = [
            w.get("word", "").strip() for w in transcript_data.get("word_timestamps", [])
        ]
        transcript_tokens = cls._tokenize(" ".join(transcript_words))
        script_tokens = cls._tokenize(script_text)

        result["script_word_count"] = len(script_tokens)
        result["transcript_word_count"] = len(transcript_tokens)

        if not script_tokens or not transcript_tokens:
            result["issues"].append(
                f"transcript_comparison: empty token set "
                f"(script={len(script_tokens)}, transcript={len(transcript_tokens)})"
            )
            return result

        # --- Punctuation-leak detection (TTS reading literal punctuation) ---
        script_set = set(script_tokens)
        leak_occurrences: dict[str, int] = {}
        for token in transcript_tokens:
            if token in cls._TTS_PUNCTUATION_LEAK_WORDS and token not in script_set:
                leak_occurrences[token] = leak_occurrences.get(token, 0) + 1

        if leak_occurrences:
            formatted = ", ".join(
                f"{w!r}×{n}" for w, n in sorted(leak_occurrences.items(), key=lambda x: -x[1])
            )
            result["spurious_punctuation_words"] = [
                {"word": w, "count": n} for w, n in leak_occurrences.items()
            ]
            result["issues"].append(
                f"TTS punctuation leak: transcript contains {formatted} — "
                f"these words are NOT in the script, which means the voice "
                f"engine is reading literal punctuation aloud. Rewrite the "
                f"script to eliminate the corresponding characters (ellipses, "
                f"em-dashes, etc.) and regenerate narration."
            )

        # --- Word accuracy via set overlap (cheap & ordering-insensitive) ---
        # We don't penalize small word-order differences or minor TTS
        # hallucinations; we just want to know "did 90%+ of the script's
        # content make it into the audio." Using set overlap on the script
        # side is robust to transcription noise.
        matched = sum(1 for t in script_tokens if t in set(transcript_tokens))
        accuracy = matched / max(1, len(script_tokens))
        result["word_accuracy"] = round(accuracy, 3)
        result["transcript_matches_script"] = accuracy >= 0.9 and not leak_occurrences

        if accuracy < 0.9:
            result["issues"].append(
                f"Low transcript-to-script match: only {accuracy:.0%} of script "
                f"words appear in the transcribed audio ({matched}/"
                f"{len(script_tokens)}). Narration may be truncated, mispronounced, "
                f"or the wrong script was used."
            )

        return result

    def _run_final_review(
        self,
        output_path: Path,
        edit_decisions: dict[str, Any] | None = None,
        proposal_packet: dict[str, Any] | None = None,
        narration_transcript_path: str | Path | None = None,
        script_text: str | None = None,
    ) -> dict[str, Any]:
        """Run post-render self-review and produce a final_review artifact.

        This is the governance contract: the compose runtime MUST inspect
        the actual rendered output before marking the stage complete.
        Never claim a video is ready without a real probe + frame sample.

        When `proposal_packet` is provided, its
        `production_plan.render_runtime` is compared against
        `edit_decisions.render_runtime` so `runtime_swap_detected` can
        actually flip. Without it, we fall back to
        `edit_decisions.metadata.proposal_render_runtime` (which the edit
        director can set explicitly to opt into swap detection).

        Returns a dict conforming to final_review.schema.json.
        """
        log = logging.getLogger("video_compose.final_review")
        issues: list[str] = []

        # --- 1. Technical probe via ffprobe ---
        technical_probe: dict[str, Any] = {
            "valid_container": False,
            "issues": [],
        }
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", str(output_path),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if proc.returncode == 0:
                probe_data = json.loads(proc.stdout)
                fmt = probe_data.get("format", {})
                streams = probe_data.get("streams", [])
                video_stream = next(
                    (s for s in streams if s.get("codec_type") == "video"), {}
                )
                audio_stream = next(
                    (s for s in streams if s.get("codec_type") == "audio"), {}
                )

                duration = float(fmt.get("duration", 0))
                width = int(video_stream.get("width", 0))
                height = int(video_stream.get("height", 0))
                fps_str = video_stream.get("r_frame_rate", "0/1")
                fps = self._parse_probe_fps(fps_str)

                technical_probe = {
                    "valid_container": bool(video_stream),
                    "duration_seconds": round(duration, 2),
                    "resolution": f"{width}x{height}",
                    "fps": fps,
                    "has_audio": bool(audio_stream),
                    "codec": video_stream.get("codec_name", "unknown"),
                    "file_size_bytes": int(fmt.get("size", 0)),
                    "issues": [],
                }

                # Sanity checks
                if duration < 1.0:
                    technical_probe["issues"].append(
                        f"Output is only {duration:.1f}s — suspiciously short"
                    )

                # Check target duration from edit_decisions
                target_dur = None
                if edit_decisions:
                    target_dur = (
                        edit_decisions.get("total_duration_seconds")
                        or edit_decisions.get("metadata", {}).get("target_duration_seconds")
                    )
                if target_dur and target_dur > 0:
                    drift_pct = abs(duration - target_dur) / target_dur
                    if drift_pct > 0.25:
                        technical_probe["issues"].append(
                            f"Duration drift: rendered {duration:.1f}s vs target {target_dur}s "
                            f"({drift_pct:.0%} off). Review pacing or trim."
                        )
                    technical_probe["target_duration"] = target_dur
                    technical_probe["duration_drift_pct"] = round(drift_pct * 100, 1)
                if width < 320 or height < 240:
                    technical_probe["issues"].append(
                        f"Resolution {width}x{height} is very low"
                    )
                if not audio_stream:
                    technical_probe["issues"].append("No audio stream in output")
            else:
                technical_probe["issues"].append(
                    f"ffprobe failed with exit code {proc.returncode}"
                )
        except FileNotFoundError:
            technical_probe["issues"].append("ffprobe not found — cannot validate output")
        except Exception as e:
            technical_probe["issues"].append(f"ffprobe error: {e}")

        issues.extend(technical_probe.get("issues", []))

        # --- 2. Visual spotcheck: sample 4 frames ---
        visual_spotcheck: dict[str, Any] = {
            "frames_sampled": 0,
            "frame_paths": [],
            "black_frames_detected": False,
            "broken_overlays": False,
            "missing_assets": False,
            "unreadable_text": False,
            "issues": [],
        }
        duration = technical_probe.get("duration_seconds", 0)
        if duration > 0 and technical_probe.get("valid_container"):
            try:
                frame_dir = output_path.parent / ".final_review_frames"
                frame_dir.mkdir(parents=True, exist_ok=True)
                # Sample at 10%, 35%, 65%, 90% of duration
                sample_points = [0.10, 0.35, 0.65, 0.90]
                frame_paths = []
                for i, pct in enumerate(sample_points):
                    ts = round(duration * pct, 2)
                    frame_path = frame_dir / f"review_frame_{i}.png"
                    cmd = [
                        "ffmpeg", "-y", "-ss", str(ts),
                        "-i", str(output_path),
                        "-frames:v", "1", "-q:v", "2",
                        str(frame_path),
                    ]
                    subprocess.run(cmd, capture_output=True, timeout=15)
                    if frame_path.exists():
                        frame_paths.append(str(frame_path))

                        # Check for black frames (file size heuristic:
                        # a 1920x1080 PNG of pure black is ~5KB)
                        if frame_path.stat().st_size < 2000:
                            visual_spotcheck["black_frames_detected"] = True

                visual_spotcheck["frames_sampled"] = len(frame_paths)
                visual_spotcheck["frame_paths"] = frame_paths

                if len(frame_paths) < 4:
                    visual_spotcheck["issues"].append(
                        f"Only {len(frame_paths)}/4 frames extracted — some timestamps may be out of range"
                    )
                if visual_spotcheck["black_frames_detected"]:
                    visual_spotcheck["issues"].append(
                        "Black frame detected — possible missing asset or failed render segment"
                    )
            except Exception as e:
                visual_spotcheck["issues"].append(f"Frame sampling error: {e}")

        issues.extend(visual_spotcheck.get("issues", []))

        # --- 3. Audio spotcheck ---
        audio_spotcheck: dict[str, Any] = {
            "narration_present": False,
            "music_present": False,
            "unexpected_silence": False,
            "clipping_detected": False,
            "mix_intelligible": True,
            "issues": [],
        }
        if technical_probe.get("has_audio") and duration > 0:
            try:
                # Use ffmpeg volumedetect to check audio levels
                cmd = [
                    "ffmpeg", "-i", str(output_path),
                    "-af", "volumedetect", "-f", "null", "-",
                ]
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=60
                )
                stderr = proc.stderr or ""
                # Parse mean_volume and max_volume
                mean_vol = None
                max_vol = None
                for line in stderr.split("\n"):
                    if "mean_volume:" in line:
                        try:
                            mean_vol = float(line.split("mean_volume:")[1].strip().split()[0])
                        except (ValueError, IndexError):
                            pass
                    if "max_volume:" in line:
                        try:
                            max_vol = float(line.split("max_volume:")[1].strip().split()[0])
                        except (ValueError, IndexError):
                            pass

                if mean_vol is not None:
                    if mean_vol < -60:
                        audio_spotcheck["unexpected_silence"] = True
                        audio_spotcheck["issues"].append(
                            f"Mean volume {mean_vol:.1f} dB — effectively silent"
                        )
                    # Assume narration present if mean volume is reasonable
                    if mean_vol > -40:
                        audio_spotcheck["narration_present"] = True
                    # Assume music present if audio exists (conservative)
                    if mean_vol > -50:
                        audio_spotcheck["music_present"] = True

                if max_vol is not None and max_vol > -0.5:
                    audio_spotcheck["clipping_detected"] = True
                    audio_spotcheck["issues"].append(
                        f"Max volume {max_vol:.1f} dB — possible clipping"
                    )
            except Exception as e:
                audio_spotcheck["issues"].append(f"Audio analysis error: {e}")

        issues.extend(audio_spotcheck.get("issues", []))

        # --- 4. Promise preservation ---
        promise_preservation: dict[str, Any] = {
            "delivery_promise_honored": True,
            "silent_downgrade_detected": False,
            "runtime_swap_detected": False,
            "issues": [],
        }
        if edit_decisions:
            renderer_family = edit_decisions.get("renderer_family", "")
            promise_preservation["renderer_family_used"] = renderer_family

            # Runtime governance — record what actually ran and flag a swap.
            # Three sources of truth, in priority order:
            #   1. proposal_packet.production_plan.render_runtime (authoritative)
            #   2. edit_decisions.metadata.proposal_render_runtime (if edit stage
            #      explicitly copied it to opt into in-tool swap detection)
            #   3. edit_decisions.render_runtime itself (cannot detect a swap in
            #      this case — reviewer does cross-artifact comparison instead)
            render_runtime_edit = (edit_decisions.get("render_runtime") or "").strip().lower()
            if render_runtime_edit:
                promise_preservation["render_runtime_used"] = render_runtime_edit

                proposal_runtime: str | None = None
                runtime_source: str | None = None
                if proposal_packet:
                    pp_runtime = (
                        (proposal_packet.get("production_plan") or {}).get("render_runtime")
                        or ""
                    ).strip().lower()
                    if pp_runtime:
                        proposal_runtime = pp_runtime
                        runtime_source = "proposal_packet.production_plan.render_runtime"
                if proposal_runtime is None:
                    md_runtime = (
                        (edit_decisions.get("metadata") or {}).get("proposal_render_runtime")
                        or ""
                    ).strip().lower()
                    if md_runtime:
                        proposal_runtime = md_runtime
                        runtime_source = "edit_decisions.metadata.proposal_render_runtime"

                if proposal_runtime is None:
                    promise_preservation["runtime_swap_check"] = (
                        "skipped — no proposal_packet or proposal_render_runtime "
                        "metadata provided. Reviewer skill does cross-artifact "
                        "comparison separately."
                    )
                elif proposal_runtime != render_runtime_edit:
                    promise_preservation["runtime_swap_detected"] = True
                    promise_preservation["runtime_swap_check"] = (
                        f"detected — source: {runtime_source}"
                    )
                    promise_preservation["issues"].append(
                        f"render_runtime changed between proposal ({proposal_runtime}) "
                        f"and compose ({render_runtime_edit}) — this is a contract "
                        f"violation unless a render_runtime_selection decision was logged."
                    )
                else:
                    promise_preservation["runtime_swap_check"] = (
                        f"ok — proposal and edit agree ({runtime_source})"
                    )

            delivery_data = (
                edit_decisions.get("metadata", {}).get("delivery_promise")
                or edit_decisions.get("delivery_promise")
            )
            if delivery_data:
                try:
                    from lib.delivery_promise import DeliveryPromise
                    promise = DeliveryPromise.from_dict(delivery_data)
                    cuts = edit_decisions.get("cuts", [])
                    result = promise.validate_cuts(cuts)
                    motion_ratio = result.get("motion_ratio", 0)
                    promise_preservation["motion_ratio_actual"] = round(motion_ratio, 3)

                    if not result["valid"]:
                        promise_preservation["delivery_promise_honored"] = False
                        for v in result["violations"]:
                            promise_preservation["issues"].append(v)

                    # Detect silent downgrade: motion-led promise but <50% motion
                    if (delivery_data.get("type") == "motion_led"
                            and motion_ratio < 0.5):
                        promise_preservation["silent_downgrade_detected"] = True
                        promise_preservation["issues"].append(
                            f"Motion-led promise but only {motion_ratio:.0%} motion — "
                            f"silent downgrade to still-led"
                        )
                except Exception as e:
                    promise_preservation["issues"].append(
                        f"Could not validate delivery promise: {e}"
                    )

        issues.extend(promise_preservation.get("issues", []))

        # --- 5. Subtitle check ---
        subtitle_check: dict[str, Any] = {
            "subtitles_expected": False,
            "subtitles_present": False,
            "issues": [],
        }
        if edit_decisions:
            ed_subs = edit_decisions.get("subtitles", {})
            subtitle_check["subtitles_expected"] = bool(ed_subs.get("enabled"))

            # Check if output has subtitle stream
            if technical_probe.get("valid_container"):
                try:
                    cmd = [
                        "ffprobe", "-v", "quiet", "-print_format", "json",
                        "-show_streams", "-select_streams", "s",
                        str(output_path),
                    ]
                    proc = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=15
                    )
                    if proc.returncode == 0:
                        sub_data = json.loads(proc.stdout)
                        sub_streams = sub_data.get("streams", [])
                        subtitle_check["subtitles_present"] = len(sub_streams) > 0

                    # If subtitles were expected but not found as a stream,
                    # they may be burned in (which is fine — not a failure)
                    if (subtitle_check["subtitles_expected"]
                            and not subtitle_check["subtitles_present"]):
                        # Check if subtitle_path was used (burned in)
                        sub_source = ed_subs.get("source")
                        if sub_source and Path(sub_source).exists():
                            # Burned-in subtitles are not detectable as streams
                            subtitle_check["subtitles_present"] = True
                            subtitle_check["coverage_ratio"] = 1.0
                        else:
                            subtitle_check["issues"].append(
                                "Subtitles expected but not found in output and "
                                "no subtitle source file exists for burn-in"
                            )
                except Exception as e:
                    subtitle_check["issues"].append(f"Subtitle check error: {e}")

        issues.extend(subtitle_check.get("issues", []))

        # --- 6. Transcript-vs-script comparison ---
        # Catches content-level TTS failures (the classic "Chirp reads `...`
        # as the word 'dot'" trap) that volume-based audio checks miss.
        # Only runs when caller provides both the transcript and script; when
        # skipped, issues list records that so the silence is visible.
        transcript_comparison = self._compare_transcript_to_script(
            Path(narration_transcript_path) if narration_transcript_path else None,
            script_text,
        )
        issues.extend(transcript_comparison.get("issues", []))

        # --- 7. Determine overall status ---
        critical_issues = [
            i for i in issues
            if any(kw in i.lower() for kw in [
                "silent downgrade", "delivery promise violation",
                "effectively silent", "ffprobe failed", "suspiciously short",
                "tts punctuation leak",  # reading literal punctuation aloud
            ])
        ]

        if critical_issues:
            status = "revise"
            recommended_action = "re_render"
        elif issues:
            status = "pass"
            recommended_action = "present_to_user"
        else:
            status = "pass"
            recommended_action = "present_to_user"

        if not technical_probe.get("valid_container"):
            status = "fail"
            recommended_action = "re_render"

        final_review = {
            "version": "1.0",
            "output_path": str(output_path),
            "status": status,
            "checks": {
                "technical_probe": technical_probe,
                "visual_spotcheck": visual_spotcheck,
                "audio_spotcheck": audio_spotcheck,
                "promise_preservation": promise_preservation,
                "subtitle_check": subtitle_check,
                "transcript_comparison": transcript_comparison,
            },
            "issues_found": issues,
            "recommended_action": recommended_action,
        }

        log.info(
            "Final review: status=%s, issues=%d, action=%s",
            status, len(issues), recommended_action,
        )

        return final_review

    @staticmethod
    def _parse_probe_fps(fps_str: str) -> float:
        """Parse ffprobe fps string like '30/1' or '24000/1001'."""
        try:
            if "/" in fps_str:
                num, den = fps_str.split("/")
                return round(int(num) / max(int(den), 1), 2)
            return float(fps_str)
        except (ValueError, ZeroDivisionError):
            return 0.0

    def _burn_subtitles(self, inputs: dict[str, Any]) -> ToolResult:
        """Burn subtitle file into video."""
        input_path = Path(inputs["input_path"])
        subtitle_path = Path(inputs["subtitle_path"])
        output_path = Path(inputs.get("output_path", str(input_path.with_stem(f"{input_path.stem}_subtitled"))))

        if not input_path.exists():
            return ToolResult(success=False, error=f"Input not found: {input_path}")
        if not subtitle_path.exists():
            return ToolResult(success=False, error=f"Subtitle file not found: {subtitle_path}")

        style = inputs.get("subtitle_style", {})
        ass_style = self._build_subtitle_style(style)
        sub_escaped = str(subtitle_path.resolve()).replace("\\", "/").replace(":", "\\:")
        codec = inputs.get("codec", "libx264")
        crf = inputs.get("crf", 23)

        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vf", f"subtitles='{sub_escaped}':force_style='{ass_style}'",
            "-c:v", codec, "-crf", str(crf),
            "-c:a", "copy",
            str(output_path),
        ]

        self.run_command(cmd)

        return ToolResult(
            success=True,
            data={
                "operation": "burn_subtitles",
                "output": str(output_path),
            },
            artifacts=[str(output_path)],
        )

    def _overlay(self, inputs: dict[str, Any]) -> ToolResult:
        """Composite overlay images/videos on top of base video."""
        input_path = Path(inputs["input_path"])
        overlays = inputs.get("overlays", [])
        output_path = Path(inputs.get("output_path", str(input_path.with_stem(f"{input_path.stem}_overlay"))))
        codec = inputs.get("codec", "libx264")
        crf = inputs.get("crf", 23)

        if not input_path.exists():
            return ToolResult(success=False, error=f"Input not found: {input_path}")
        if not overlays:
            return ToolResult(success=False, error="No overlays provided")

        # Build complex filter for each overlay
        input_args = ["-i", str(input_path)]
        filter_parts = []
        prev_label = "0:v"

        for i, ov in enumerate(overlays):
            asset_path = Path(ov["asset_path"])
            if not asset_path.exists():
                return ToolResult(success=False, error=f"Overlay asset not found: {asset_path}")

            input_args.extend(["-i", str(asset_path)])

            x = int(ov.get("x", 0))
            y = int(ov.get("y", 0))
            start = ov.get("start_seconds", 0)
            end = ov.get("end_seconds")
            opacity = ov.get("opacity", 1.0)

            overlay_input = f"{i + 1}:v"

            # Scale overlay if dimensions specified
            if "width" in ov and "height" in ov:
                w = int(ov["width"])
                h = int(ov["height"])
                filter_parts.append(f"[{overlay_input}]scale={w}:{h}[ov_scaled_{i}]")
                overlay_input = f"ov_scaled_{i}"

            # Build enable expression for timed overlays
            enable = f"between(t,{start},{end})" if end else f"gte(t,{start})"
            out_label = f"v{i}"

            filter_parts.append(
                f"[{prev_label}][{overlay_input}]overlay={x}:{y}:enable='{enable}'[{out_label}]"
            )
            prev_label = out_label

        filter_complex = ";".join(filter_parts)

        cmd = ["ffmpeg", "-y"]
        cmd.extend(input_args)
        cmd.extend(["-filter_complex", filter_complex])
        cmd.extend(["-map", f"[{prev_label}]", "-map", "0:a?"])
        cmd.extend(["-c:v", codec, "-crf", str(crf), "-c:a", "copy"])
        cmd.append(str(output_path))

        self.run_command(cmd)

        return ToolResult(
            success=True,
            data={
                "operation": "overlay",
                "overlay_count": len(overlays),
                "output": str(output_path),
            },
            artifacts=[str(output_path)],
        )

    def _encode(self, inputs: dict[str, Any]) -> ToolResult:
        """Re-encode video with a specific profile/codec settings."""
        input_path = Path(inputs["input_path"])
        output_path = Path(inputs.get("output_path", str(input_path.with_stem(f"{input_path.stem}_encoded"))))
        codec = inputs.get("codec", "libx264")
        crf = inputs.get("crf", 23)
        preset = inputs.get("preset", "medium")
        profile_name = inputs.get("profile")

        if not input_path.exists():
            return ToolResult(success=False, error=f"Input not found: {input_path}")

        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-c:v", codec, "-crf", str(crf), "-preset", preset,
            "-c:a", "aac", "-b:a", "192k",
        ]

        # Apply media profile if specified
        if profile_name:
            try:
                from lib.media_profiles import get_profile, ffmpeg_output_args
                profile = get_profile(profile_name)
                cmd.extend(["-s", f"{profile.width}x{profile.height}"])
                cmd.extend(["-r", str(profile.fps)])
            except (ImportError, ValueError):
                pass  # proceed without profile

        cmd.append(str(output_path))
        self.run_command(cmd)

        return ToolResult(
            success=True,
            data={
                "operation": "encode",
                "codec": codec,
                "crf": crf,
                "profile": profile_name,
                "output": str(output_path),
            },
            artifacts=[str(output_path)],
        )

    @staticmethod
    def _resolve_subtitle_style(
        explicit_style: dict | None,
        edit_decisions: dict | None,
        playbook: dict | None,
    ) -> dict:
        """Resolve subtitle style with layered priority.

        Priority: explicit_style > edit_decisions.subtitles.style > playbook > defaults.
        This prevents every video from looking identical (Arial bold white).
        """
        # Start with minimal fallback defaults
        resolved = {
            "font": "Inter",
            "font_size": 28,
            "bold": True,
            "outline_width": 2,
            "shadow": 0,
            "margin_v": 40,
            "alignment": 2,
        }

        # Layer 1: Playbook-derived style
        if playbook:
            typo = playbook.get("typography", {})
            colors = playbook.get("visual_language", {}).get("color_palette", {})
            if typo.get("body", {}).get("family"):
                resolved["font"] = typo["body"]["family"]
            if colors.get("text"):
                resolved["primary_color"] = colors["text"]
            if colors.get("background"):
                resolved["outline_color"] = colors["background"]
                # Semi-transparent background for readability
                bg = colors["background"]
                resolved["back_color"] = bg

        # Layer 2: edit_decisions subtitle style
        if edit_decisions:
            ed_style = edit_decisions.get("subtitles", {}).get("style", {})
            for k, v in ed_style.items():
                if v is not None:
                    resolved[k] = v

        # Layer 3: Explicit override (highest priority)
        if explicit_style:
            for k, v in explicit_style.items():
                if v is not None:
                    resolved[k] = v

        return resolved

    @staticmethod
    def _build_subtitle_style(style: dict) -> str:
        """Build ASS force_style string from style dict."""
        parts = []
        parts.append(f"FontName={style.get('font', 'Inter')}")
        parts.append(f"FontSize={style.get('font_size', 28)}")
        parts.append(f"Bold={1 if style.get('bold', True) else 0}")
        if style.get("primary_color"):
            parts.append(f"PrimaryColour={style['primary_color']}")
        if style.get("outline_color"):
            parts.append(f"OutlineColour={style['outline_color']}")
        if style.get("back_color"):
            parts.append(f"BackColour={style['back_color']}")
        border_style = style.get("border_style", 1)
        parts.append(f"BorderStyle={border_style}")
        parts.append(f"Outline={style.get('outline_width', 2)}")
        parts.append(f"Shadow={style.get('shadow', 0)}")
        parts.append(f"MarginV={style.get('margin_v', 40)}")
        parts.append(f"Alignment={style.get('alignment', 2)}")
        return ",".join(parts)

    @staticmethod
    def _build_atempo(factor: float) -> str:
        """Build atempo filter chain for audio speed adjustment."""
        filters = []
        remaining = factor
        while remaining > 100.0:
            filters.append("atempo=100.0")
            remaining /= 100.0
        while remaining < 0.5:
            filters.append("atempo=0.5")
            remaining /= 0.5
        filters.append(f"atempo={remaining:.4f}")
        return ",".join(filters)
