"""Base tool class implementing the expanded ToolContract.

Every tool in OpenMontage inherits from BaseTool. This enforces a uniform
interface for discovery, execution, cost estimation, and health reporting.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import platform
import subprocess
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional


def _load_dotenv() -> None:
    """Load .env into os.environ once at import time.

    This ensures API keys are available before any tool is instantiated,
    even when tools are imported directly without going through the registry.
    Only sets variables that are not already in the environment.
    """
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.is_file():
        return
    with open(env_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            # Strip inline comments: VAR=value  # comment
            # But only if the # is preceded by whitespace (avoid stripping from values like colors)
            if "  #" in value:
                value = value[:value.index("  #")].rstrip()
            elif "\t#" in value:
                value = value[:value.index("\t#")].rstrip()
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv()


class ToolTier(str, Enum):
    CORE = "core"
    VOICE = "voice"
    ENHANCE = "enhance"
    GENERATE = "generate"
    SOURCE = "source"
    ANALYZE = "analyze"
    PUBLISH = "publish"


class ToolStability(str, Enum):
    EXPERIMENTAL = "experimental"
    BETA = "beta"
    PRODUCTION = "production"


class ToolStatus(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    DEGRADED = "degraded"


class ToolRuntime(str, Enum):
    """Where and how a tool executes."""
    LOCAL = "local"            # Runs entirely on-device, free, no network
    LOCAL_GPU = "local_gpu"    # Runs on-device but needs GPU (VRAM)
    API = "api"                # Calls an external API, requires API key, costs money
    HYBRID = "hybrid"          # Can run locally OR via API (e.g., image_selector)


class ExecutionMode(str, Enum):
    SYNC = "sync"
    ASYNC = "async"


class Determinism(str, Enum):
    DETERMINISTIC = "deterministic"
    SEEDED = "seeded"
    STOCHASTIC = "stochastic"


class ResumeSupport(str, Enum):
    NONE = "none"
    FROM_START = "from_start"
    FROM_CHECKPOINT = "from_checkpoint"


@dataclass
class ResourceProfile:
    """Hardware resource envelope for a tool."""
    cpu_cores: int = 1
    ram_mb: int = 512
    vram_mb: int = 0
    disk_mb: int = 100
    network_required: bool = False


@dataclass
class RetryPolicy:
    """Safe retry behavior for a tool."""
    max_retries: int = 0
    backoff_seconds: float = 1.0
    retryable_errors: list[str] = field(default_factory=list)


class ErrorCode(str, Enum):
    """Closed vocabulary of failure categories.

    HTTP dispatchers branch on these codes instead of regex-matching
    error strings. Tools populate `ToolResult.error_code` to opt in;
    legacy tools default to `legacy` and migrate over time.

    Keep the vocabulary tight (~12 codes) — too narrow loses signal,
    too wide and callers stop branching on them. When proposing a new
    code, ask: "would an HTTP layer translate THIS code differently
    from any of the existing codes?" If no, fold into the closest
    existing one.
    """

    INVALID_INPUT = "invalid_input"
    """Caller-provided inputs failed validation (schema, type, enum).
    HTTP: 400. Caller should fix their request."""

    MISSING_DEPENDENCY = "missing_dependency"
    """Required external binary not on PATH (ffmpeg, npx, ffprobe).
    HTTP: 503. Operator-side issue, not the caller's."""

    PROVIDER_UNAVAILABLE = "provider_unavailable"
    """API key missing, provider returned 401/403, or selector found
    no available provider for the capability. HTTP: 503."""

    PROVIDER_RATE_LIMITED = "provider_rate_limited"
    """Provider returned 429 or equivalent. HTTP: 429. Caller may
    retry with backoff."""

    PROVIDER_TIMEOUT = "provider_timeout"
    """Provider didn't respond within the timeout. HTTP: 504."""

    PROVIDER_FAILED = "provider_failed"
    """Generic provider failure (5xx, unexpected response shape,
    parse error). HTTP: 502. Caller can't fix without provider
    changes."""

    ASSET_MISSING = "asset_missing"
    """A referenced asset (file path, URL, asset_manifest ID) doesn't
    exist or can't be reached. HTTP: 404 / 422."""

    RENDER_FAILED = "render_failed"
    """The renderer (ffmpeg, Remotion, HyperFrames) crashed or
    returned no output. HTTP: 500."""

    OUTPUT_INVALID = "output_invalid"
    """Renderer produced output but post-condition probe found it
    malformed (zero duration, missing video stream, all-black, etc.).
    HTTP: 500. Often pairs with actual_output diagnostic data."""

    GOVERNANCE_VIOLATION = "governance_violation"
    """A pipeline-level invariant was broken (e.g. render_runtime
    must be locked at proposal stage). HTTP: 422."""

    INTERNAL_ERROR = "internal_error"
    """Unhandled exception caught by an outer wrapper. HTTP: 500.
    Includes a Python traceback in `error` for debugging."""

    LEGACY = "legacy"
    """Default for tools that haven't been migrated to populate
    error_code. Means "I don't know what category this failure is
    in." HTTP dispatchers should treat this as INTERNAL_ERROR for
    routing purposes."""


@dataclass
class ToolResult:
    """Standard result returned by tool execution."""
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    error: Optional[str] = None
    error_code: Optional[ErrorCode] = None
    cost_usd: float = 0.0
    duration_seconds: float = 0.0
    seed: Optional[int] = None
    model: Optional[str] = None

    def __post_init__(self) -> None:
        # Failed results without an explicit code default to LEGACY so
        # downstream branching always has something to switch on.
        # Successful results don't need a code (None means "no error").
        if not self.success and self.error_code is None:
            self.error_code = ErrorCode.LEGACY


class BaseTool(ABC):
    """Abstract base class for all OpenMontage tools."""

    # --- Identity (override in subclasses) ---
    name: str = ""
    version: str = "0.1.0"
    tier: ToolTier = ToolTier.CORE
    stability: ToolStability = ToolStability.EXPERIMENTAL
    execution_mode: ExecutionMode = ExecutionMode.SYNC
    determinism: Determinism = Determinism.DETERMINISTIC
    runtime: ToolRuntime = ToolRuntime.LOCAL

    # --- Dependencies ---
    # For API tools, add "env:ENVVAR_NAME" to signal required API keys
    dependencies: list[str] = []
    install_instructions: str = ""

    # --- Capabilities ---
    capability: str = "generic"
    provider: str = "openmontage"
    capabilities: list[str] = []
    input_schema: dict = {}
    output_schema: dict = {}
    artifact_schema: dict = {}
    progress_schema: Optional[dict] = None
    supports: dict[str, Any] = {}
    best_for: list[str] = []
    not_good_for: list[str] = []
    provider_matrix: dict[str, Any] = {}

    # --- Resource & retry ---
    resource_profile: ResourceProfile = ResourceProfile()
    retry_policy: RetryPolicy = RetryPolicy()

    # --- Resume & idempotency ---
    resume_support: ResumeSupport = ResumeSupport.NONE
    idempotency_key_fields: list[str] = []

    # --- Side effects & fallback ---
    side_effects: list[str] = []
    fallback: Optional[str] = None
    fallback_tools: list[str] = []

    # --- Agent skills (Layer 3 references) ---
    # Names of installed agent skills in .agents/skills/ that teach the
    # underlying technology. The orchestrator uses these to load relevant
    # API knowledge when planning tool usage.
    agent_skills: list[str] = []

    # --- Verification ---
    user_visible_verification: list[str] = []

    # --- Optional telemetry / quality hints for the scoring engine ---
    # If set (0.0-1.0), lib/scoring.py uses these directly instead of falling
    # back to stability-based heuristics. Leave unset unless the tool has a
    # real measured or well-calibrated value.
    quality_score: Optional[float] = None
    historical_success_rate: Optional[float] = None
    latency_p50_seconds: Optional[float] = None

    # ---- Status reporting ----

    def get_status(self) -> ToolStatus:
        """Check if this tool's dependencies are satisfied."""
        try:
            self.check_dependencies()
            return ToolStatus.AVAILABLE
        except DependencyError:
            return ToolStatus.UNAVAILABLE

    def check_dependencies(self) -> None:
        """Verify all dependencies are installed. Raises DependencyError if not."""
        for dep in self.dependencies:
            if dep.startswith("cmd:"):
                cmd_name = dep[4:]
                if shutil.which(cmd_name) is None:
                    raise DependencyError(
                        f"Command {cmd_name!r} not found. {self.install_instructions}"
                    )
            elif dep.startswith("env:"):
                env_name = dep[4:]
                if not os.environ.get(env_name):
                    raise DependencyError(
                        f"Environment variable {env_name!r} not set. {self.install_instructions}"
                    )
            elif dep.startswith("python:"):
                module_name = dep[7:]
                try:
                    __import__(module_name)
                except ImportError:
                    raise DependencyError(
                        f"Python module {module_name!r} not installed. {self.install_instructions}"
                    )

    def get_info(self) -> dict[str, Any]:
        """Return full tool contract info for registry/discovery."""
        usage_location = inspect.getfile(self.__class__)
        return {
            "name": self.name,
            "version": self.version,
            "tier": self.tier.value,
            "capability": self.capability,
            "provider": self.provider,
            "stability": self.stability.value,
            "status": self.get_status().value,
            "execution_mode": self.execution_mode.value,
            "determinism": self.determinism.value,
            "runtime": self.runtime.value,
            "module_path": self.__class__.__module__,
            "usage_location": usage_location,
            "dependencies": self.dependencies,
            "install_instructions": self.install_instructions,
            "capabilities": self.capabilities,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "artifact_schema": self.artifact_schema,
            "supports": self.supports,
            "best_for": self.best_for,
            "not_good_for": self.not_good_for,
            "provider_matrix": self.provider_matrix,
            "resource_profile": {
                "cpu_cores": self.resource_profile.cpu_cores,
                "ram_mb": self.resource_profile.ram_mb,
                "vram_mb": self.resource_profile.vram_mb,
                "disk_mb": self.resource_profile.disk_mb,
                "network_required": self.resource_profile.network_required,
            },
            "resume_support": self.resume_support.value,
            "side_effects": self.side_effects,
            "fallback": self.fallback,
            "fallback_tools": self.fallback_tools or ([self.fallback] if self.fallback else []),
            "agent_skills": self.agent_skills,
            "related_skills": self.agent_skills,
            "user_visible_verification": self.user_visible_verification,
            "quality_score": self.quality_score,
            "historical_success_rate": self.historical_success_rate,
            "latency_p50_seconds": self.latency_p50_seconds,
        }

    # ---- Cost estimation ----

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        """Estimate cost in USD for the given inputs. Override for paid tools."""
        return 0.0

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        """Estimate runtime in seconds. Override for long-running tools."""
        return 0.0

    # ---- Idempotency ----

    def idempotency_key(self, inputs: dict[str, Any]) -> str:
        """Compute a cache key from idempotency fields."""
        key_data = {k: inputs.get(k) for k in self.idempotency_key_fields}
        raw = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    # ---- Execution ----

    @abstractmethod
    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        """Run the tool. Subclasses must implement this."""
        ...

    def execute_validated(self, inputs: dict[str, Any]) -> ToolResult:
        """Validate `inputs` against `self.input_schema` then execute.

        This is the boundary helper callers should use when dispatching
        from an HTTP layer or any context where structured error
        signaling matters more than a Python traceback.

        Validation failure → ToolResult(success=False, error_code=
        "invalid_input", error="...field-level message...") instead of
        the bare KeyError / TypeError that the underlying execute()
        would raise on a missing or wrong-typed field.

        Tools that want stricter or different validation can override
        this method. The default uses jsonschema against
        `self.input_schema`.

        Existing callers of `.execute()` are not affected — this is
        additive. Migrating a caller from `.execute()` to
        `.execute_validated()` is the opt-in for the structured-error
        boundary.
        """
        validation_error = self._validate_inputs(inputs)
        if validation_error is not None:
            return validation_error
        return self.execute(inputs)

    def _validate_inputs(self, inputs: dict[str, Any]) -> Optional[ToolResult]:
        """Run jsonschema.validate(inputs, self.input_schema).

        Returns a failed ToolResult on validation error, None when the
        inputs are schema-valid. Tools with no input_schema (or an
        empty one) skip validation — there's nothing to check.

        Errors are formatted with the json-pointer-style path of the
        offending field plus the validator's own message, so callers
        can branch on the field name without regex-matching free text.
        """
        if not self.input_schema:
            return None
        try:
            import jsonschema
        except ImportError:
            # jsonschema isn't a hard dep; if it's missing, skip
            # validation rather than fail the call.
            return None

        try:
            jsonschema.validate(instance=inputs, schema=self.input_schema)
        except jsonschema.ValidationError as exc:
            field_path = ".".join(str(p) for p in exc.absolute_path) or "<root>"
            return ToolResult(
                success=False,
                error_code=ErrorCode.INVALID_INPUT,
                error=(
                    f"{self.name}: invalid input at '{field_path}': {exc.message}"
                ),
            )
        except jsonschema.SchemaError as exc:
            # Tool's own schema is malformed — this is a tool-author
            # bug, not a caller bug. Surface it with a distinct prefix
            # so callers know they didn't cause it.
            return ToolResult(
                success=False,
                error_code=ErrorCode.INTERNAL_ERROR,
                error=(
                    f"{self.name}: tool schema is malformed "
                    f"(this is a tool bug, not a caller error): {exc.message}"
                ),
            )
        return None

    def dry_run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Preflight check without side effects. Override for paid/publishing tools."""
        return {
            "tool": self.name,
            "estimated_cost_usd": self.estimate_cost(inputs),
            "estimated_runtime_seconds": self.estimate_runtime(inputs),
            "status": self.get_status().value,
            "would_execute": True,
        }

    # ---- CLI helper ----

    def run_command(
        self,
        cmd: list[str],
        *,
        timeout: Optional[int] = None,
        cwd: Optional[Path] = None,
    ) -> subprocess.CompletedProcess:
        """Run a subprocess command with standard error handling.

        On Windows, resolves .cmd/.bat wrappers (e.g. npx, npm) via
        shutil.which() so subprocess.run() can find them without shell=True.
        """
        resolved_cmd = list(cmd)
        if platform.system() == "Windows" and resolved_cmd:
            exe = shutil.which(resolved_cmd[0])
            if exe:
                resolved_cmd[0] = exe
        return subprocess.run(
            resolved_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            check=True,
        )


class DependencyError(Exception):
    """Raised when a tool's dependency is not satisfied."""
    pass
