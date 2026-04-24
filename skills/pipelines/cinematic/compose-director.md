# Compose Director - Cinematic Pipeline

## When To Use

Render the cinematic piece with careful attention to grade, audio dynamics, and frame treatment. This is not a generic export step.

## Runtime Routing (MANDATORY first step)

Read `edit_decisions.render_runtime`. Cinematic work routes to:

- **`render_runtime="remotion"`** — default for video-led trailers using `CinematicRenderer`. Keeps video clips, transitions, and ambient overlays in one React-based pass.
- **`render_runtime="hyperframes"`** — for kinetic title cards, HTML/GSAP-driven trailers, or launch-reel-style compositions where the visual grammar is HTML/CSS. See `skills/core/hyperframes.md`. `hyperframes lint` and `hyperframes validate` must both pass before render.
- **`render_runtime="ffmpeg"`** — simple source-footage concat with no composition.

`delivery_promise.motion_required=true` means the locked runtime is a commitment. Silent swap to another runtime (including FFmpeg Ken Burns) is a CRITICAL governance violation. If the locked runtime fails, escalate per AGENT_GUIDE.md > "Escalate Blockers Explicitly."

**Pass `proposal_packet` to `video_compose.execute()`** so the tool's `runtime_swap_detected` check compares directly against `proposal_packet.production_plan.render_runtime`. Without it the swap check is skipped in-tool and only the reviewer skill catches the drift.

## Prerequisites

| Layer | Resource | Purpose |
|-------|----------|---------|
| Schema | `schemas/artifacts/render_report.schema.json` | Artifact validation |
| Prior artifacts | `state.artifacts["edit"]["edit_decisions"]`, `state.artifacts["assets"]["asset_manifest"]` | Edit plan and media assets |
| Tools | `video_compose`, `audio_mixer`, `video_stitch`, `video_trimmer`, `color_grade`, `audio_enhance` | Render and finishing |
| Playbook | Active style playbook | Finish consistency |

## Process

### 0. Check Hard Requirements Before Rendering

If the approved brief or scene plan makes motion a hard requirement, verify that the render path still preserves that promise.

- If Remotion is required and unavailable or failing, stop and bubble the issue to the user immediately.
- Do not switch to an FFmpeg-only still-image fallback for a motion-led trailer, teaser, or agent video.
- Do not convert the piece into an animatic unless the user explicitly approves that downgrade.
- If the render engine changes materially, tell the user before rendering and explain why.

**Remotion preflight (runtime-agnostic — fork issue #35):**

Before calling `video_compose` with `operation="render"` or `"remotion_render"` for any scene plan that includes Remotion scene types (title cards, stat cards, anime/hero_title, end-tag, overlays), confirm Remotion is available. Use whichever introspection mechanism your runtime exposes:

- **Tool-dispatch / MCP / sub-loop runtimes (no shell):** the host has already inlined `video_compose`'s `input_schema` in your prompt context — the `render_engines` and `remotion_note` fields are surfaced via `get_info()` data attached to the tool definition. Read them there. If your runtime exposes a separate introspection tool (e.g. `tools.get_info`, `/tools/{name}` HTTP route), use that.

  **Do NOT call `video_compose` with empty args to "trigger preflight" — `operation` is required and the call will error.** That doom-loop pattern was fork issue #35; the LLM was reading the old shell snippet as "call video_compose" and firing `video_compose({})` repeatedly until iteration limits killed the run.

- **Shell-equipped runtimes (CLI, dev workflow):** run

  ```bash
  python -c "from tools.tool_registry import registry; registry.discover(); import json; print(json.dumps(registry.get('video_compose').get_info(), indent=2))"
  ```

  and grep `render_engines` / `remotion_note` from the JSON output.

If Remotion is not in the available render engines, stop and report to the user per the Decision Communication Contract. Do not substitute a reduced-fidelity render path without approval.

If your runtime can't introspect at all, skip this step — the tool's own `ToolResult(success=False, error=...)` on a Remotion render with the runtime missing will surface the same information loud-fail at the actual render call. The preflight is belt-and-suspenders, not load-bearing.

### 1. Use Frame Treatment Deliberately

Only use letterbox, 24fps intent, or heavy grading if they help the piece. Do not apply them because the pipeline name says cinematic.

### 2. Choose the right audio_mixer operation (HARD RULE — fork issue #33)

`audio_mixer` exposes several operations, and picking the wrong one introduces silent regressions like fork issue #33 (the `segmented_music` `pumping` bug). Match the operation to the creative intent:

| Intent | Operation | Notes |
|---|---|---|
| **Continuous background music** under the entire piece (the modal cinematic case) | Pass the music file directly as `audio_path` to `video_compose` | Works on both ffmpeg-`compose` and remotion-`render` paths after fork issue #32. No mixer step needed for this case. |
| Continuous narration + continuous music (ducked) | `audio_mixer operation="full_mix"` with `tracks=[{role:"speech",...},{role:"music",...}]` and `ducking.enabled=true`, then pass the result as `audio_path` to `video_compose` | The standard cinematic narration path. Music ducks under speech, no per-cut fades. |
| Music ONLY during specific time windows (e.g. music during talking-head section, silence during showcase clip) | `audio_mixer operation="segmented_music"` with the windows you want music in | Each segment fades in/out at its boundaries. |
| Multiple tracks summed without ducking | `audio_mixer operation="mix"` | Use `music_gain_db` if you need to bias tracks per role. |

**Do NOT use `segmented_music` with one segment per scene cut for continuous background music.** That was fork issue #33: `_segmented_music` applies a fade-in + fade-out to EVERY segment, so when segments meet at scene-cut boundaries the music dips to zero at every cut. The correct pattern for "music plays through all 3 scenes" is one of:

- `audio_path=<music>` to `video_compose` directly (simplest, no mixer step)
- `segmented_music` with a SINGLE segment spanning the whole timeline `[{start:0, end:total_duration}]` (fades only at the start/end)
- `full_mix` if you also need narration mixed alongside

If the brief calls for music to start/stop selectively, multi-segment `segmented_music` is the right choice — but verify the segments you pass match the "where should music play" creative intent, not "where the visual cuts are."

### 3. Verify The Final Mood

Check:

- opening frame,
- reveal beat,
- final landing,
- subtitle readability where relevant.

### 4. Use Render Metadata

Recommended metadata keys:

- `frame_treatment`
- `grade_profile`
- `mix_notes`
- `variant_outputs`

## Common Pitfalls

- Flattening the audio so the piece loses dynamics.
- Applying letterbox to footage that needs every pixel.
- Letting grading or sharpening damage faces or text.
- Silently swapping a blocked Remotion render for a lower-fidelity still-image export.
