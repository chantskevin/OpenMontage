# Script Director - Cinematic Pipeline

## When To Use

This stage builds the beat map, selected lines, title-card copy, and reveal structure for the cinematic piece. You are shaping rhythm, not writing a dense explainer.

## Prerequisites

| Layer | Resource | Purpose |
|-------|----------|---------|
| Schema | `schemas/artifacts/script.schema.json` | Artifact validation |
| Prior artifact | `state.artifacts["proposal"]["proposal_packet"]` | Emotional arc and source truth |
| Tools | `transcriber`, `scene_detect` | Optional dialogue mining and source review |

## Audio Treatment — read the lock, don't re-decide (HARD RULE)

The proposal stage locks `proposal_packet.production_plan.audio_treatment.mode` ∈ `{voice_led, dialogue_led, music_only}`. **Read it. Honor it. Never re-decide here.** Re-deciding is how issue #7 happened — scripts silently dropped narration because this stage thought it had the authority to pick.

Per-mode behavior:

| `audio_treatment.mode` | What this stage writes |
|---|---|
| `voice_led` | Original narration. Every `sections[].text` is non-empty; `speaker_directions` per section drives TTS tone; `metadata.voice_direction` carries the brand/register baseline. |
| `dialogue_led` | Lines extracted from user source footage via `transcriber`. `sections[].text` holds the selected spoken line verbatim; `source_ref` points at the timecoded source segment. |
| `music_only` | Beat map only. `sections[].text` may be empty strings (the field is schema-required but empty is valid). `metadata.silent_treatment_reason` MUST be populated from the proposal's `audio_treatment.rationale` so downstream reviewers see the explicit opt-in. |

**If `audio_treatment` is missing from proposal_packet** — don't default, **escalate**. This is a proposal-stage governance violation (see `skills/pipelines/cinematic/proposal-director.md` → Step 4d). Calling it out here is cheap; silently picking a mode here breaks every assumption downstream.

**Timings must tile the full duration regardless of mode:** `sections[]` start/end_seconds cover `[0, total_duration_seconds]` with no gaps. Compose reads the tiling as the beat plan even when there's no spoken audio to synthesize.

Canonical shape for a `voice_led` 10s spot:

```json
{
  "version": "1.0",
  "title": "ACME — ALL IN",
  "total_duration_seconds": 10,
  "sections": [
    { "id": "s1", "text": "No filter. No shortcut.",      "start_seconds": 0, "end_seconds": 3, "speaker_directions": "warm, low, confident" },
    { "id": "s2", "text": "Every ingredient, all in.",     "start_seconds": 3, "end_seconds": 7, "speaker_directions": "slower, more intimate" },
    { "id": "s3", "text": "ACME. The full soy, for real.", "start_seconds": 7, "end_seconds": 10, "speaker_directions": "declarative, payoff" }
  ],
  "metadata": {
    "voice_direction": "native-speaker product voice; warm health register; no announcer energy"
  }
}
```

For `music_only`, same tiling, empty text, `metadata.silent_treatment_reason` set:

```json
{
  "version": "1.0",
  "title": "Drift",
  "total_duration_seconds": 30,
  "sections": [
    { "id": "s1", "text": "", "start_seconds": 0, "end_seconds": 10 },
    { "id": "s2", "text": "", "start_seconds": 10, "end_seconds": 20 },
    { "id": "s3", "text": "", "start_seconds": 20, "end_seconds": 30 }
  ],
  "metadata": {
    "silent_treatment_reason": "atmospheric mood piece, user approved sound-design-led treatment at proposal stage"
  }
}
```

## Process

### 1. Build A Beat Map First

Use a simple structure:

- hook,
- escalation,
- reveal,
- landing.

If the piece is longer, add one midpoint turn. Do not let it become essay-shaped.

### 2. Use Dialogue Sparingly

If source speech exists, use `transcriber` to find:

- strong standalone lines,
- emotional phrases,
- concise declarations,
- reveal phrases.

Source-dialogue use is mode-gated by the proposal's `audio_treatment`. In `dialogue_led` mode, `transcriber` output is the primary source of `sections[].text`. In `voice_led` mode, transcribed lines may inform the narration you write but do not replace it. In `music_only` mode, skip this step entirely.

### 3. Keep Title Cards Short

Title-card copy should feel trailer-like:

- fewer words,
- more contrast,
- more whitespace,
- more timing precision.

### 4. Store Beat Truth In Metadata

Recommended metadata keys:

- `beat_map`
- `dialogue_selects`
- `title_card_copy`
- `music_turns`
- `silence_windows`

### 5. Quality Gate

- the beat map escalates cleanly,
- dialogue and title cards do not explain the same thing twice,
- the reveal lands distinctly,
- the landing gives the viewer a final feeling or action.

### Mid-Production Fact Verification

If you encounter uncertainty during script writing:
- Use `web_search` to verify factual claims before committing them to the script
- Use `web_search` to find reference images for visual accuracy
- Log verification in the decision log: `category="visual_accuracy_check"`

Every factual claim in the script should be traceable to the `research_brief`.
If you make a claim that isn't in the research, do additional research and
add the source. Do not invent statistics, dates, or attributions.

## Common Pitfalls

- Writing full explanatory paragraphs instead of beats.
- Using too many title cards.
- Revealing the best moment too early.
