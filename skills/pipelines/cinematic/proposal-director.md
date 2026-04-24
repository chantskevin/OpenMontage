# Proposal Director — Cinematic Pipeline

## When to Use

You are the **Proposal Director** for a cinematic video (trailers, brand films, montages, dramatic edits). You sit between the Research Director and the Script Director. You receive a `research_brief` full of visual references, mood research, and cinematic direction options, and transform it into a concrete, reviewable proposal that the user approves before any money is spent.

**This is the approval gate.** Nothing downstream runs until the user says "go."

## Runtime Selection (required field — `render_runtime`)

Cinematic proposals must lock **both** a `renderer_family` (creative grammar: `cinematic-trailer`, `documentary-montage`, etc.) and a `render_runtime` (technical engine). Read `skills/meta/animation-runtime-selector.md` and `skills/core/hyperframes.md` for the decision matrix, and `AGENT_GUIDE.md` → "Present Both Composition Runtimes (HARD RULE)" for the governance contract.

**MANDATORY workflow — present both runtimes, don't silently default:**

1. Query `video_compose.get_info()["render_engines"]`. If both `remotion` and `hyperframes` are `True`, proceed to step 2.
2. Present both runtimes to the user with brief-specific analysis:
   - **Remotion** — one line on fit (mention `CinematicRenderer`, `<OffthreadVideo>`, existing transition stack if applicable), one line on tradeoff.
   - **HyperFrames** — one line on fit (mention kinetic title sequences, registry shader transitions, or HTML-native typographic motion if applicable), one line on tradeoff.
3. Recommend one with rationale tied to the brief's `delivery_promise` (especially `motion_required`), `renderer_family`, and approved tone.
4. Wait for explicit user approval. Do NOT write `render_runtime` into `proposal_packet.production_plan` before approval.
5. Log a `render_runtime_selection` decision in `decision_log` with BOTH runtimes in `options_considered` plus `ffmpeg` if it was a realistic option.

Fit cheat-sheet for the recommendation (NOT an auto-decision):

- Video-led trailer with motion clips via `<OffthreadVideo>` + color-graded overlays → lean **Remotion**.
- HTML/GSAP-driven trailer: kinetic title sequence, launch reel, brand film where the visual grammar is typographic → lean **HyperFrames**.
- Shader transitions or registry grain overlays → lean **HyperFrames**.
- Simplest source-footage concat with no composition → **ffmpeg**.

**Motion-required deliverables**: if `delivery_promise.motion_required=true`, the chosen runtime is a commitment. Silent downgrade to FFmpeg Ken Burns or still-led animatic is forbidden. If the chosen runtime becomes unavailable at render time, compose must escalate, not substitute.

A `render_runtime_selection` decision with only one option considered when both were available is a CRITICAL reviewer finding.

## Prerequisites

| Layer | Resource | Purpose |
|-------|----------|---------|
| Schema | `schemas/artifacts/proposal_packet.schema.json` | Artifact validation |
| Prior artifact | `research_brief` from Research Director | Visual references, mood research, cinematic directions |
| Pipeline manifest | `pipeline_defs/cinematic.yaml` | Stage and tool definitions |
| Tool registry | `support_envelope()` output | What's actually available right now |
| Cost tracker | `tools/cost_tracker.py` | Cost estimation data |
| Style playbooks | `styles/*.yaml` | Available visual styles |
| User input | Subject, footage, preferences | Creative direction |

## Process

### Step 0: Check for Reference Video Context

Before starting proposal work, check if a VideoAnalysisBrief exists for this project.

**When a VideoAnalysisBrief is present — Reference-Aware Cinematic Concept Design:**

**HARD RULE: No carbon copies.** Each concept option MUST:
1. Name at least ONE cinematic element it keeps from the reference (mood, pacing, color palette, shot language)
2. Name at least ONE element it changes (emotional arc, visual treatment, subject matter, sound design)
3. Explain WHY the change creates a different emotional impact

**Cinematic differentiation patterns:**

| Pattern | Example |
|---------|---------|
| **Same mood, different subject** | Reference: dark sci-fi mood → Ours: same darkness applied to deep ocean |
| **Same subject, different emotional arc** | Reference: tension→reveal → Ours: wonder→scale |
| **Same pacing, different visual language** | Reference: handheld raw → Ours: locked-off geometric |
| **Same color world, different lighting** | Reference: warm golden hour → Ours: warm but tungsten/interior |

**Mandatory Sample Protocol:** After concept approval, produce a 10-15 second cinematic
sample BEFORE full production. This is critical for cinematic work — mood mismatches are
expensive to fix downstream. Present with visual + audio + music.

**When no VideoAnalysisBrief is present:** Skip this step and proceed normally.

### Step 1: Absorb the Research

Read the `research_brief` thoroughly. Extract:

- **`research_summary`** — the researcher's strongest creative direction.
- **`angles_discovered`** — these are your raw cinematic direction candidates.
- **Visual references** — the real-world precedents that inform each direction.
- **Audio direction** — music mood and sound design notes.
- **Source reality** — what footage/stills the user actually has.
- **Motion commitment** — whether motion is required.

### Step 2: Run Preflight

Before designing concepts, know what tools are available:

```bash
python -c "from tools.tool_registry import registry; import json; registry.discover(); print(json.dumps(registry.support_envelope(), indent=2))"
```

Record:
- Video generation providers — **critical for cinematic**. If motion is required, these must be available.
- Image generation providers — for support visuals and mood inserts
- TTS providers — check availability so you know what's on offer before Step 4d locks the audio treatment. A `voice_led` brief without a configured TTS provider is a blocker, not a downgrade; surface it to the user rather than silently picking `music_only`.
- Music generation — check availability honestly
- Enhancement tools — color_grade, audio_enhance are high-value for cinematic
- **Remotion render engine** — check `video_compose.get_info()["render_engines"]["remotion"]`

**Motion-required enforcement:** If the research brief indicates `motion_required: true`, verify that video generation or source footage can actually deliver motion. If neither is available, **do not silently downgrade to still-led**. Instead, present the constraint honestly and let the user decide.

### Step 2c: Mood Board (Before Concepts)

Before developing full concepts, present a quick mood board to catch direction mismatches early. Cinematic work is especially susceptible to tone misalignment, so this step is critical:

- **3-5 reference images** (from web search — real film stills, not generic stock)
- **Color palette direction** (2-3 palettes: e.g. desaturated cold vs warm golden vs high-contrast noir)
- **Tone references** ("Think: Terrence Malick meets National Geographic" or "Think: David Fincher trailer pacing")
- **1-2 music mood references** (genre + energy + emotional arc, e.g. "ambient synth building to orchestral crescendo")

Ask: **"Does this FEEL like what you're imagining? Any of these off-track?"**

This is cheaper than building 3 full cinematic directions and catches tone mismatches before they're embedded in concept design. If the user says "less moody, more energetic," you've saved a concept round.

### Step 3: Design Concept Directions

Build **at least 3 genuinely different cinematic directions.** Start from the `angles_discovered` in the research brief.

For each concept, specify all fields in `proposal_packet.concept_options`:

#### 3a: Title and Emotional Hook

Cinematic hooks are different from explainer hooks — they evoke **feeling**, not information gaps.

| Pattern | When to Use |
|---------|-------------|
| **Sensory** | "You hear it before you see it. A frequency that shouldn't exist." | When the mood is mystery/tension |
| **Scale shift** | "In the time it takes to read this sentence, 4.7 million packets crossed the Atlantic." | When the concept involves scale |
| **Intimate** | "She waters them at 6am. Before the city wakes. Before anyone watches." | When the mood is intimate/human |
| **Provocation** | "They told us the message was noise. It wasn't." | When the concept involves a reveal |
| **Contrast** | "Three blocks from Wall Street, on a rooftop covered in clover, 40,000 bees are building a city." | When the subject is surprising in context |

#### 3b: Emotional Arc

Every cinematic concept needs an explicit arc:

| Arc | Structure | Best For |
|-----|-----------|----------|
| `tension → reveal` | Build unease, then pay it off | Teasers, sci-fi, thriller |
| `wonder → scale` | Start small, expand to massive | Nature, tech, cosmos |
| `intimacy → payoff` | Close, personal, then earned moment | Documentary, human interest |
| `urgency → resolution` | Fast pace to satisfying close | Product, action, launch |
| `mystery → CTA` | Intrigue that leads to action | Brand films, campaigns |
| `stillness → eruption` | Calm before powerful climax | Music videos, art films |

#### 3c: Delivery Promise

For cinematic, explicitly classify:

```yaml
delivery_promise:
  promise_type: motion_led  # or source_led, hybrid
  motion_required: true     # false only if user approves still-led
  source_required: false    # true if user has footage
  tone_mode: cinematic      # cinematic, raw, intimate, epic
  quality_floor: presentable  # draft, presentable, broadcast
  approved_fallback: null   # animatic, still_led, or null (no fallback)
```

**Rule:** `motion_led` forbids still-led fallback unless the user explicitly approves `animatic` as the fallback.

#### 3d: Visual Treatment

For each concept, define:
- **Color palette** — specific hex references, not just "dark"
- **Lighting approach** — high_key, low_key, natural, golden_hour, etc.
- **Camera language** — dominant shot sizes, movements
- **Texture** — film grain, clean digital, anamorphic, handheld
- **Typography** — if title cards are used, their style and restraint level

#### 3e: Renderer Family Selection

Choose the renderer family and lock it in the proposal:

| Family | When | Composition |
|--------|------|-------------|
| `cinematic-trailer` | Trailers, teasers, brand films with generated/source video | CinematicRenderer |
| `presenter` | Talking head with cinematic enhancement | TalkingHead |
| `explainer-data` | Only if this is really an explainer that wants cinematic dressing | Explainer |

**Rule:** The renderer family is selected here and locked before scene planning. The compose stage cannot change it without logging a decision and surfacing the change to the user.

### Step 4: Progressive Reveal, Diversity Check, and Concept Selection

#### 4a: Progressive Reveal

Don't dump the full proposal at once. Build understanding step by step:

1. **Research summary** (2-3 sentences): "Here's what I found about the subject and its visual potential..."
   → User reacts, course-corrects if needed.
2. **Mood board** (from Step 2c — already presented)
   → User confirms feel.
3. **Concept directions** (3+ emotional/visual approaches):
   → Present each concept's emotional hook, arc, and visual treatment.
4. **Invite mixing** (see 4c below).
5. **Production plan for selected direction** (tools, cost, renderer family):
   → User approves budget and approach.

Each step is a chance for the user to course-correct before the next step builds on it.

#### 4b: Diversity Check

Before presenting concepts:
- [ ] No two concepts share the same emotional arc
- [ ] No two concepts use the same visual treatment
- [ ] At least one concept takes a creative risk
- [ ] Each concept's visual references are from different sources
- [ ] Each concept is achievable with current capabilities (or states what's missing)

#### 4c: Invite Mixing

After presenting concepts, always say something like:
> "You can also mix elements — for example, Concept A's emotional arc with Concept C's visual treatment and Concept B's music direction. What speaks to you?"

If the user mixes, create a new hybrid concept entry in the proposal_packet with clear attribution: "Emotional arc from Concept A, visual treatment from Concept C, music direction from Concept B."

Let the user select, combine, modify, or redirect entirely.

### Step 4d: Audio Treatment Lock (HARD RULE)

Cinematic covers three distinct audio treatments, and each drives a different script shape + different asset-stage toolchain. **The user picks. You surface the choice; they sign off. Defaulting silently is forbidden — silent defaults are what produced issue #7 (voice-led briefs landing as music-only videos).**

Write `proposal_packet.production_plan.audio_treatment` before the user sees the final proposal:

```json
{
  "mode": "voice_led" | "dialogue_led" | "music_only",
  "rationale": "<one-line tie to research findings and user intent>"
}
```

**Which mode fits:**

| Signal in brief / research | Mode |
|---|---|
| Brand commercial, product hero, founder story, launch film, explainer-with-voice | `voice_led` |
| User has source footage with usable speech; documentary/interview/testimonial treatment | `dialogue_led` |
| Fashion film, abstract trailer, mood piece, atmospheric / sound-design-led — AND brief explicitly says so | `music_only` |
| **Brief is ambiguous (no audio cue either way)** | **`voice_led` (default)** — see fork issue #34 |

**Mode-specific commitments:**

- `voice_led` → also lock `voice_selection` (provider, voice_id, rationale). Without a concrete voice pick the asset stage has to guess, and guessing picks the wrong register for brand work.
- `dialogue_led` → confirm `source_media_review` actually contains usable speech files (`usable_for` includes `"dialogue"` or `"source_line"`). If not, fall back to `voice_led`; don't ship an empty-audio cut.
- `music_only` → `rationale` MUST reflect an EXPLICIT user signal — phrases like "no narration", "atmospheric", "mood piece", "silent film", "visual-only", "sound-design-led" in the brief. Present as: "This is a music-only / sound-design-led piece — no narration or dialogue. Do you confirm?" Do NOT default to this mode silently to save on TTS cost.

**HARD RULE (fork issue #34): default to `voice_led` for ambiguous briefs.** A 10s cinematic for a brand with no audio cue is most often narration-led — defaulting to `music_only` ships a silent-but-for-music deliverable that the user can't easily salvage (the asset/script/compose stages will all respect the lock). If the user wanted music_only they'll redirect; if you defaulted to music_only and they wanted narration they have to re-run the entire pipeline.

Picking `music_only` requires a concrete signal from the brief. Post-hoc rationale ("the high-intensity pacing is best served by atmospheric music") is exactly the failure mode #34 calls out — narrative the director spins around an arbitrary choice. If you find yourself writing rationale like that, you're defaulting silently. Switch to `voice_led` and write narration; the user can ask for a music-only re-cut if they want.

The reviewer check: a cinematic proposal with `audio_treatment.mode = "music_only"` whose `rationale` doesn't quote a brief phrase or explicit user signal is a CRITICAL finding — same severity as a missing audio_treatment lock.

Log the choice in `decision_log` as `audio_treatment_selection` with `options_considered` covering all three modes and `rejected_because` on each unchosen one. A cinematic proposal without `audio_treatment` locked is a CRITICAL reviewer finding.

### Step 5: Music Plan (Mandatory for Cinematic)

Cinematic videos live and die by their audio. Surface the music situation before the user approves.

Check availability in this order:
1. **User music library (`music_library/`)** — list available tracks
2. **Music generation APIs** — report status, cost, and quality honestly
3. **Bring-your-own path** — user can drop a track in `music_library/`

Present explicit options:
```
MUSIC PLAN
├── Your music library: [N tracks / empty]
├── AI generation: [provider] — [AVAILABLE/UNAVAILABLE] [cost]
└── Bring your own: Drop a track in music_library/ before asset stage

Recommendation: [specific recommendation based on mood research]
```

### Step 6: Build Production Plan

For the selected concept, design the stage-by-stage plan with specific providers, costs, and honest tradeoffs.

**Cinematic-specific tool priorities:**
- **Color grade** — high priority. Cinematic output without grade looks flat.
- **Audio enhance** — high priority. Audio dynamics matter more in mood-driven work.
- **Video generation** — if motion-required, this is non-negotiable.
- **Music** — must be resolved. No cinematic piece should have silence as a surprise.

### Step 7: Cost Estimate

Itemize all costs honestly. Cinematic tends to be more expensive than explainer (more generated clips, music, grade passes).

### Step 8: Present and Approve

Present concepts clearly. Invite the user to:
- Select one as-is
- Mix elements from multiple concepts
- Request modifications
- Redirect entirely

Set `approval.status: "pending"`. Pipeline does NOT proceed without approval.

### Step 9: Before calling `complete_stage` — Proposal Lock Pre-check (HARD RULE)

**Fork issue #21:** even with Step 4d's HARD RULE prose, BOS observed cinematic proposals shipping `production_plan` with NO `audio_treatment` and NO `voice_selection` — the director either skipped Step 4d or satisfied the prose without actually writing the fields. The downstream asset-director had nothing to read, silently skipped TTS, and the final video shipped in silence. The user-visible failure was our agent telling them "voice-processing is temporarily faulty" — a lie caused by an unlocked proposal decision.

Before calling `complete_stage` on the proposal stage, assert ALL of the following. If any fail, call `fail_stage` naming the missing field — do not complete_stage:

1. **`production_plan.audio_treatment.mode`** is present and one of `{voice_led, dialogue_led, music_only}`. Missing or null is forbidden.
2. **`production_plan.audio_treatment.rationale`** is present and non-empty. The reviewer uses rationale to tell "deliberate music_only sign-off" from "accidentally defaulted to music_only" — don't skip it.
3. **If `audio_treatment.mode == "voice_led"`:** `production_plan.voice_selection` is present with `provider`, `voice_id`, and `provider_candidates` populated per Step 4d.1. A voice_led proposal without voice_selection forces the asset stage to guess and reintroduces the fork issue #17 hallucination surface.
4. **`production_plan.render_runtime`** is one of `{remotion, hyperframes, ffmpeg}`. Already schema-required; verify explicitly anyway because a missing value here causes the compose stage to fail with a cryptic routing error.
5. **`production_plan.renderer_family`** is present and matches the delivery_promise.

**Reviewer check:** `schemas/artifacts/proposal_packet.schema.json` now encodes requirement (1) via a JSON Schema `if/then` — when `pipeline == "cinematic"`, `audio_treatment` is required. Requirement (3) is encoded the same way. Requirement (2) is enforced via `audio_treatment.required = [mode, rationale]`. `lib/checkpoint.py._validate_cross_artifact_invariants` is the redundant safety net for callers that bypass jsonschema.validate.

A cinematic proposal that somehow lands `status: "completed"` without `audio_treatment` locked is a CRITICAL reviewer finding — the same severity as a voice_led run with zero narration assets (fork issue #19). Both are different symptoms of the same underlying bug: a stage director skipping a HARD RULE the downstream stages depend on.

### Step 10: Submit

Validate `proposal_packet` against schema and submit.

## Common Pitfalls

- **Calling it cinematic because of black bars**: Letterboxing is not cinematography. The treatment must include shot language, lighting, movement, and emotional arc.
- **Hiding motion downgrade**: If motion-required content will actually be still images with Ken Burns, say so explicitly.
- **Music as afterthought**: In cinematic work, music is 50% of the mood. Surface it early.
- **Three versions of "dark and moody"**: Three concepts with the same emotional register but different titles are one concept. Diversity means different arcs, different moods, different risks.
- **Ignoring source reality**: If the user has no footage and limited generation tools, the proposal must reflect that — not pretend the constraints don't exist.


## When You Do Not Know How

If you encounter a generation technique, provider behavior, or prompting pattern you are unsure about:

1. **Search the web** for current best practices — models and APIs change frequently, and the agent's training data may be stale
2. **Check `.agents/skills/`** for existing Layer 3 knowledge (provider-specific prompting guides, API patterns)
3. **If neither helps**, write a project-scoped skill at `projects/<project-name>/skills/<name>.md` documenting what you learned
4. **Reference source URLs** in the skill so the knowledge is traceable
5. **Log it** in the decision log: `category: "capability_extension"`, `subject: "learned technique: <name>"`

This is especially important for:
- **Video generation prompting** — models respond to specific vocabularies that change with each version
- **Image model parameters** — optimal settings for FLUX, DALL-E, Imagen differ and evolve
- **Audio provider quirks** — voice cloning, music generation, and TTS each have model-specific best practices
- **Remotion component patterns** — new composition techniques emerge as the framework evolves

Do not rely on stale knowledge. When in doubt, search first.
