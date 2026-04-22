"""Checkpoint writer/reader for pipeline state persistence.

Each stage writes a checkpoint after completion. The orchestrator uses
checkpoints to resume pipelines and to present state at human checkpoints.
"""

from __future__ import annotations

import json
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import jsonschema

from schemas.artifacts import ARTIFACT_NAMES, validate_artifact

# All known stages across all pipelines (used only for artifact name lookup).
ALL_KNOWN_STAGES = frozenset([
    "research", "proposal", "idea", "script", "scene_plan",
    "assets", "edit", "compose", "publish",
])

# Backward-compatible alias — existing code / tests that import STAGES still work.
# New code should use get_pipeline_stages(pipeline_type) instead.
STAGES = ["research", "proposal", "idea", "script", "scene_plan",
          "assets", "edit", "compose", "publish"]

CANONICAL_STAGE_ARTIFACTS = {
    "research": "research_brief",
    "proposal": "proposal_packet",
    "idea": "brief",
    "script": "script",
    "scene_plan": "scene_plan",
    "assets": "asset_manifest",
    "edit": "edit_decisions",
    "compose": "render_report",
    "publish": "publish_log",
}

# Additional artifacts that may be produced alongside canonical ones.
# These are not stage-defining but are required by governance contracts.
SUPPLEMENTARY_ARTIFACTS = {
    "source_media_review",  # Required before first planning stage when user media exists
    "final_review",         # Required by compose stage before presenting to user
    "video_analysis_brief", # Reference-video grounding artifact carried alongside stages
}


def get_pipeline_stages(pipeline_type: str | None) -> list[str]:
    """Return the ordered stage list for a specific pipeline.

    Falls back to STAGES (deterministic canonical order) when pipeline_type
    is not provided or the manifest cannot be loaded.

    Previous versions used a set intersection here, which produced
    nondeterministic ordering. The fallback now uses a stable list.
    """
    if pipeline_type is None:
        # Deterministic canonical fallback — sorted to ensure stable ordering
        import logging
        logging.getLogger(__name__).warning(
            "get_pipeline_stages called without pipeline_type — "
            "using canonical fallback order. Pass pipeline_type for correctness."
        )
        return list(STAGES)

    try:
        from lib.pipeline_loader import load_pipeline, get_stage_order
        manifest = load_pipeline(pipeline_type)
        return get_stage_order(manifest)
    except (FileNotFoundError, Exception):
        # Graceful fallback: return all known stages in canonical order
        return list(STAGES)

CHECKPOINT_SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent
    / "schemas"
    / "checkpoints"
    / "checkpoint.schema.json"
)


class CheckpointValidationError(ValueError):
    """Raised when a checkpoint or its canonical artifacts are invalid."""


@lru_cache(maxsize=1)
def _load_checkpoint_schema() -> dict[str, Any]:
    with open(CHECKPOINT_SCHEMA_PATH) as f:
        return json.load(f)


def _validate_artifacts_for_stage(
    stage: str,
    status: str,
    artifacts: dict[str, Any],
) -> None:
    required_artifact = CANONICAL_STAGE_ARTIFACTS[stage]
    if status in {"completed", "awaiting_human"} and required_artifact not in artifacts:
        raise CheckpointValidationError(
            f"Stage {stage!r} with status {status!r} must include "
            f"canonical artifact {required_artifact!r}"
        )

    for artifact_name, artifact_data in artifacts.items():
        if artifact_name not in ARTIFACT_NAMES:
            continue
        if not isinstance(artifact_data, dict):
            raise CheckpointValidationError(
                f"Artifact {artifact_name!r} must be a JSON object matching its schema"
            )
        try:
            validate_artifact(artifact_name, artifact_data)
        except Exception as exc:
            raise CheckpointValidationError(
                f"Artifact {artifact_name!r} failed schema validation: {exc}"
            ) from exc

    _validate_cross_artifact_invariants(stage, status, artifacts)


_NARRATION_ASSET_TYPES = frozenset({"narration"})


def _count_narration_assets(asset_manifest: dict[str, Any]) -> int:
    """Count narration assets in an asset_manifest.

    The asset_manifest schema enum names exactly one spoken-audio
    type: `narration`. Any TTS or dialogue-extract output for a
    voice_led/dialogue_led run must use that type to count.
    """
    assets = asset_manifest.get("assets") or []
    return sum(
        1
        for a in assets
        if isinstance(a, dict) and a.get("type") in _NARRATION_ASSET_TYPES
    )


def _validate_cross_stage_invariants(
    pipeline_dir: Path,
    project_id: str,
    stage: str,
    status: str,
    artifacts: dict[str, Any],
) -> None:
    """Enforce invariants that span multiple stage checkpoints on disk.

    Within-checkpoint invariants live in
    `_validate_cross_artifact_invariants`. This function loads prior
    stage checkpoints to catch cases where a later stage's output
    contradicts a decision locked earlier.

    If a prior checkpoint can't be read, the check is skipped — we
    don't want a missing-but-optional artifact to block a legitimate
    checkpoint write. The skills layer names these as reviewer
    findings too, so a skipped check isn't a silent gap.
    """
    if status not in {"completed", "awaiting_human"}:
        return

    # Fork issue #19: when proposal locked audio_treatment.mode to a
    # spoken-audio mode (voice_led or dialogue_led), the asset stage
    # MUST produce narration/voice assets. BOS runs observed zero
    # narration entries in the manifest because the asset-director
    # skipped the TTS loop even with the proposal lock in place.
    if stage == "assets":
        asset_manifest = artifacts.get("asset_manifest")
        if not isinstance(asset_manifest, dict):
            return

        prior_proposal = read_checkpoint(pipeline_dir, project_id, "proposal")
        if prior_proposal is None:
            return
        proposal_packet = (
            prior_proposal.get("artifacts", {}).get("proposal_packet")
        )
        if not isinstance(proposal_packet, dict):
            return

        audio_treatment = (
            proposal_packet.get("production_plan", {}).get("audio_treatment")
        )
        if not isinstance(audio_treatment, dict):
            return

        mode = audio_treatment.get("mode")
        if mode not in {"voice_led", "dialogue_led"}:
            return

        narration_count = _count_narration_assets(asset_manifest)
        if narration_count == 0:
            raise CheckpointValidationError(
                f"assets stage cannot be {status!r} when "
                f"proposal_packet locked audio_treatment.mode={mode!r} "
                f"but asset_manifest contains zero narration/voice "
                f"assets. For {mode}, the asset stage must iterate "
                f"script.sections[] and call the resolved TTS "
                f"(voice_led) or extract source dialogue "
                f"(dialogue_led) per section — see "
                f"skills/pipelines/cinematic/asset-director.md → "
                f"Step 0c.1. Silent skip is fork issue #19."
            )


def _validate_cross_artifact_invariants(
    stage: str,
    status: str,
    artifacts: dict[str, Any],
) -> None:
    """Enforce invariants that span multiple artifacts in the same checkpoint.

    Individual artifact schemas can't express "A and B together must agree."
    This is where those checks live. Each invariant names the fork issue it
    closes so a reviewer tracing a regression can find the history.
    """
    if status not in {"completed", "awaiting_human"}:
        return

    # Fork issue #21: a cinematic proposal without audio_treatment locked
    # cascades into a silent final video (asset-director has nothing to
    # read → skips TTS → silent mp4 → final_review flags silence but
    # pipeline already shipped). The schema now encodes this conditional
    # requirement; this check is the redundant safety net for callers
    # that bypass jsonschema validation.
    if stage == "proposal":
        proposal_packet = artifacts.get("proposal_packet")
        if isinstance(proposal_packet, dict):
            plan = proposal_packet.get("production_plan")
            if isinstance(plan, dict) and plan.get("pipeline") == "cinematic":
                if "audio_treatment" not in plan:
                    raise CheckpointValidationError(
                        f"cinematic proposal stage cannot be {status!r} "
                        f"without production_plan.audio_treatment locked. "
                        f"The proposal-director's HARD RULE names the "
                        f"three modes (voice_led, dialogue_led, "
                        f"music_only); defaulting silently is forbidden. "
                        f"(fork issue #21)."
                    )
                mode = plan["audio_treatment"].get("mode") \
                    if isinstance(plan["audio_treatment"], dict) else None
                if mode == "voice_led" and "voice_selection" not in plan:
                    raise CheckpointValidationError(
                        f"cinematic proposal with audio_treatment.mode="
                        f"'voice_led' cannot be {status!r} without "
                        f"production_plan.voice_selection locked. The "
                        f"asset stage's TTS loop reads voice_selection "
                        f"for provider+voice_id; without it the asset "
                        f"stage has to guess and reintroduces the "
                        f"fork issue #17 hallucination surface."
                    )

    # Fork issue #20: compose stage cannot mark itself completed when its
    # own final_review self-report says the render failed. The schemas
    # accept both independently; only the combination is wrong.
    if stage == "compose":
        final_review = artifacts.get("final_review")
        if isinstance(final_review, dict):
            fr_status = final_review.get("status")
            if fr_status == "fail":
                issues = final_review.get("issues_found") or []
                detail = f" issues: {issues}" if issues else ""
                raise CheckpointValidationError(
                    f"compose stage cannot be {status!r} when "
                    f"final_review.status=={fr_status!r}. The director's "
                    f"self-review already flagged render failure — "
                    f"use fail_stage instead of complete_stage. "
                    f"(fork issue #20).{detail}"
                )
            # "Render failed, audio not verified." in audio_spotcheck is
            # the exact fingerprint from fork issue #20 — the director's
            # review spotted failure but still stamped `completed`.
            checks = final_review.get("checks") or {}
            audio_checks = checks.get("audio_spotcheck") or {}
            audio_issues = audio_checks.get("issues") or []
            if any(
                isinstance(i, str) and "Render failed" in i
                for i in audio_issues
            ):
                raise CheckpointValidationError(
                    f"compose stage cannot be {status!r} when "
                    f"final_review.checks.audio_spotcheck.issues names "
                    f'"Render failed…" — the director\'s own audio '
                    f"check saw the failure but called complete_stage "
                    f"anyway. Use fail_stage instead "
                    f"(fork issue #20). issues: {audio_issues}"
                )

def validate_checkpoint(checkpoint: dict[str, Any]) -> None:
    """Validate checkpoint structure and canonical artifact payloads.

    Uses pipeline_type (if present) to resolve the valid stage list.
    Falls back to ALL_KNOWN_STAGES when pipeline_type is absent.
    """
    stage = checkpoint.get("stage")
    status = checkpoint.get("status")
    artifacts = checkpoint.get("artifacts")
    pipeline_type = checkpoint.get("pipeline_type")

    valid_stages = (
        set(get_pipeline_stages(pipeline_type)) if pipeline_type
        else ALL_KNOWN_STAGES
    )

    if not isinstance(stage, str) or stage not in valid_stages:
        raise CheckpointValidationError(
            f"Invalid stage: {stage!r} for pipeline {pipeline_type!r}. "
            f"Valid stages: {sorted(valid_stages)}"
        )
    if not isinstance(status, str):
        raise CheckpointValidationError(f"Invalid status: {status!r}")
    if not isinstance(artifacts, dict):
        raise CheckpointValidationError("Checkpoint artifacts must be a dictionary")

    _validate_artifacts_for_stage(stage, status, artifacts)

    try:
        jsonschema.validate(instance=checkpoint, schema=_load_checkpoint_schema())
    except jsonschema.ValidationError as exc:
        raise CheckpointValidationError(f"Checkpoint failed schema validation: {exc.message}") from exc


def _checkpoint_path(pipeline_dir: Path, project_id: str, stage: str) -> Path:
    return pipeline_dir / project_id / f"checkpoint_{stage}.json"


def _decision_log_path(pipeline_dir: Path, project_id: str) -> Path:
    return pipeline_dir / project_id / "decision_log.json"


def _merge_decision_log(
    pipeline_dir: Path, project_id: str, new_log: dict[str, Any]
) -> None:
    """Append new decisions to the project-level decision log.

    Each stage may produce decisions. This function merges them into a
    single cumulative file so reviewers and the bench can inspect the
    full audit trail.
    """
    path = _decision_log_path(pipeline_dir, project_id)
    if path.exists():
        with open(path) as f:
            existing = json.load(f)
    else:
        existing = {
            "version": "1.0",
            "project_id": project_id,
            "decisions": [],
        }

    existing_ids = {d["decision_id"] for d in existing.get("decisions", [])}
    for decision in new_log.get("decisions", []):
        if decision.get("decision_id") not in existing_ids:
            existing["decisions"].append(decision)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)


def write_checkpoint(
    pipeline_dir: Path,
    project_id: str,
    stage: str,
    status: str,
    artifacts: dict[str, Any],
    *,
    pipeline_type: Optional[str] = None,
    style_playbook: Optional[str] = None,
    checkpoint_policy: str = "guided",
    human_approval_required: bool = False,
    human_approved: bool = False,
    review: Optional[dict] = None,
    cost_snapshot: Optional[dict] = None,
    error: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> Path:
    """Write a checkpoint file for a pipeline stage."""
    valid_stages = (
        set(get_pipeline_stages(pipeline_type)) if pipeline_type
        else ALL_KNOWN_STAGES
    )
    if stage not in valid_stages:
        raise ValueError(
            f"Invalid stage: {stage!r} for pipeline {pipeline_type!r}. "
            f"Valid stages: {sorted(valid_stages)}"
        )

    checkpoint = {
        "version": "1.0",
        "project_id": project_id,
        "pipeline_type": pipeline_type or "unknown",
        "stage": stage,
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint_policy": checkpoint_policy,
        "human_approval_required": human_approval_required,
        "human_approved": human_approved,
        "artifacts": artifacts,
    }
    if style_playbook is not None:
        checkpoint["style_playbook"] = style_playbook
    if review is not None:
        checkpoint["review"] = review
    if cost_snapshot is not None:
        checkpoint["cost_snapshot"] = cost_snapshot
    if error is not None:
        checkpoint["error"] = error
    if metadata is not None:
        checkpoint["metadata"] = metadata

    # Merge decision_log: if this checkpoint carries new decisions,
    # append them to the project-level decision log file, then write the
    # reference back into relevant artifacts so downstream consumers can find it.
    if "decision_log" in artifacts and isinstance(artifacts["decision_log"], dict):
        _merge_decision_log(pipeline_dir, project_id, artifacts["decision_log"])
        log_ref = str(_decision_log_path(pipeline_dir, project_id))

        # Write decision_log_ref into proposal_packet and render_report
        # artifacts if they are present in this checkpoint.
        for artifact_key in ("proposal_packet", "render_report"):
            if artifact_key in artifacts and isinstance(artifacts[artifact_key], dict):
                plan_or_top = artifacts[artifact_key]
                # proposal_packet stores it under production_plan
                if artifact_key == "proposal_packet":
                    plan = plan_or_top.get("production_plan")
                    if isinstance(plan, dict):
                        plan["decision_log_ref"] = log_ref
                else:
                    plan_or_top["decision_log_ref"] = log_ref

    validate_checkpoint(checkpoint)

    path = _checkpoint_path(pipeline_dir, project_id, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(checkpoint, f, indent=2)

    return path


def read_checkpoint(
    pipeline_dir: Path, project_id: str, stage: str
) -> Optional[dict[str, Any]]:
    """Read a checkpoint file. Returns None if not found."""
    path = _checkpoint_path(pipeline_dir, project_id, stage)
    if not path.exists():
        return None
    with open(path) as f:
        checkpoint = json.load(f)
    validate_checkpoint(checkpoint)
    return checkpoint


def get_latest_checkpoint(
    pipeline_dir: Path, project_id: str
) -> Optional[dict[str, Any]]:
    """Find the most recent checkpoint for a project (by file mtime)."""
    project_dir = pipeline_dir / project_id
    if not project_dir.exists():
        return None

    checkpoints = sorted(
        project_dir.glob("checkpoint_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not checkpoints:
        return None

    with open(checkpoints[0]) as f:
        checkpoint = json.load(f)
    validate_checkpoint(checkpoint)
    return checkpoint


def get_completed_stages(
    pipeline_dir: Path, project_id: str, pipeline_type: str | None = None
) -> list[str]:
    """Return list of stages that have a completed checkpoint.

    When pipeline_type is provided, only checks stages defined in that
    pipeline's manifest — preventing false positives from leftover
    checkpoints of a different pipeline type.
    """
    stages_to_check = get_pipeline_stages(pipeline_type)
    completed = []
    for stage in stages_to_check:
        cp = read_checkpoint(pipeline_dir, project_id, stage)
        if cp and cp.get("status") == "completed":
            completed.append(stage)
    return completed


def get_next_stage(
    pipeline_dir: Path, project_id: str, pipeline_type: str | None = None
) -> Optional[str]:
    """Determine the next stage to run based on completed checkpoints.

    Uses pipeline-specific stage order so that pipelines with different
    stage sequences (e.g. cinematic vs explainer) progress correctly.
    """
    stages = get_pipeline_stages(pipeline_type) if pipeline_type else STAGES
    completed = set(get_completed_stages(pipeline_dir, project_id, pipeline_type))
    for stage in stages:
        if stage not in completed:
            return stage
    return None
