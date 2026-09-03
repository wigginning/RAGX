"""Configuration system (02-core.md §2.4).

Three-level override, resolved in this fixed order (later wins):

    1. :class:`Settings`            global, env prefix ``RAGX_``
    2. :class:`KBConfig`            per knowledge base, stored in the Metadata DB
    3. :class:`~ragx.core.models.RequestOverride`  per request (whitelist only)

Start-up validation: ``profile=full`` requires the mandatory plugin settings to
be present or the service refuses to start with ``ConfigError(9003)``; the
``lite`` profile has zero external-service dependency.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ragx.core.models import utcnow
from ragx.core.roles import LLMRole

Profile = Literal["lite", "full"]

# ---------------------------------------------------------------------------
# KB-scoped building blocks (referenced by KBConfig, 02-core.md §2.4)
# ---------------------------------------------------------------------------
class FeatureFlags(BaseModel):
    kg_enabled: bool = False                # graph retrieval (05-kg.md)
    vlm_enabled: bool = False               # VLM description (03-ingestion.md §3.4)
    agentic_enabled: bool = False           # agentic path (07-agentic.md)
    cache_enabled: bool = True              # semantic cache (08-llm.md §8.5)
    av_enabled: bool = False                # audio/video transcription (13-parsing.md §13.3.3)


class BudgetConfig(BaseModel):
    context_token_budget: int = Field(default=4096, ge=128)
    agentic_token_budget: int = Field(default=100_000, ge=1024)
    daily_cost_limit_usd: float = Field(default=50.0, ge=0.0)


class ChunkingOptions(BaseModel):
    """04-chunking.md §4.2."""

    target_tokens: int = Field(default=512, ge=16)
    overlap_ratio: float = 0.1
    max_tokens: int = Field(default=768, ge=16)
    semantic_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    min_tokens: int = Field(default=64, ge=1)
    keep_table_header: bool = True
    context_prefix: bool = True
    merge_short_chunks: bool = True

    @field_validator("overlap_ratio")
    @classmethod
    def _overlap_range(cls, value: float) -> float:
        if not 0.0 <= value <= 0.5:
            raise ValueError("overlap_ratio must be in [0.0, 0.5]")
        return value


class ParseOptions(BaseModel):
    """13-parsing.md §13.4.1. ``None`` means "decide per page / per element"."""

    profile: Literal["auto", "fast", "balanced", "accurate"] = "auto"
    ocr: bool | None = None
    vlm_describe: bool | None = None
    table_structure: bool = True
    formula_extract: bool = True
    cross_page_table_merge: bool = True
    av_transcribe: bool = False


class KGExtractionConfig(BaseModel):
    """05-kg.md §5.2.5."""

    extractor_mode: Literal["llm", "small", "hybrid"] = "llm"
    small_ner_model: str = "urchade/gliner_mediumv2.1"
    small_re_model: str = "Babelscape/rebel-large"
    hybrid_llm_verify: bool = True
    entity_types: list[str] = Field(
        default_factory=lambda: [
            "PERSON", "ORG", "LOCATION", "CONCEPT", "EVENT", "PRODUCT", "DATE", "OTHER",
        ]
    )
    confidence_min: float = Field(default=0.5, ge=0.0, le=1.0)


class KBConfig(BaseModel):
    """Per-knowledge-base configuration (hot-reloadable, 30s process TTL)."""

    parser: str = "text"
    processor: str | None = None           # None = VLM off
    vector_store: str = "sqlite"
    graph_store: str | None = None         # None = graph route off
    embedder: str = "hash"
    reranker: str | None = None
    flags: FeatureFlags = Field(default_factory=FeatureFlags)
    chunking: ChunkingOptions = Field(default_factory=ChunkingOptions)
    budgets: BudgetConfig = Field(default_factory=BudgetConfig)
    parsing: ParseOptions = Field(default_factory=ParseOptions)
    kg: KGExtractionConfig = Field(default_factory=KGExtractionConfig)
    prompt_overrides: dict[str, str] = Field(default_factory=dict)
    #: Semantic-cache epoch (08-llm.md §8.5.3). Incremented by
    #: ``cache.invalidate_kb``; O(1) invalidation without bulk deletes.
    cache_epoch: int = 1


# ---------------------------------------------------------------------------
# LLM router config (08-llm.md §8.2.2)
# ---------------------------------------------------------------------------
class ProviderTarget(BaseModel):
    """A concrete (provider plugin, model) pair."""

    provider: str
    model: str

    @property
    def key(self) -> str:
        """Stable ``provider/model`` key used by the circuit breaker and pricing."""
        return f"{self.provider}/{self.model}"


class CircuitConfig(BaseModel):
    """08-llm.md §8.4.1."""

    window_s: float = Field(default=60.0, gt=0)
    failure_rate: float = Field(default=0.5, ge=0.0, le=1.0)
    min_samples: int = Field(default=10, ge=1)
    open_s: float = Field(default=30.0, gt=0)


class CacheConfig(BaseModel):
    """08-llm.md §8.5.1."""

    enabled: bool = True
    ttl_s: int = Field(default=86_400, gt=0)
    similarity: float = Field(default=0.97, ge=0.0, le=1.0)
    namespace: str = "ragx_semantic_cache"
    #: When true, only the exact-match tier is consulted; the vector
    #: similarity tier is skipped. Useful for very-low-latency Fast mode.
    fast_only: bool = False
    #: top_k used by the vector tier when scanning for similar cached entries.
    top_k: int = 4


class RoleConfig(BaseModel):
    """Per-role routing policy (08-llm.md §8.2.2 / §8.2.4)."""

    primary: ProviderTarget
    fallbacks: list[ProviderTarget] = Field(default_factory=list)
    timeout: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=3, ge=0)
    strategy: Literal["priority", "weighted", "least_cost"] = "priority"
    candidates: list[ProviderTarget] = Field(default_factory=list)
    weights: dict[str, float] = Field(default_factory=dict)


class LLMRouterConfig(BaseModel):
    roles: dict[LLMRole, RoleConfig]
    providers: dict[str, dict[str, Any]] = Field(default_factory=dict)
    circuit: CircuitConfig = Field(default_factory=CircuitConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    #: ``{provider/model: {prompt_per_1k, completion_per_1k}}`` (08-llm.md §8.6.2)
    pricing: dict[str, dict[str, float]] = Field(default_factory=dict)

    def get_role(self, role: LLMRole | str) -> RoleConfig:
        from ragx.core.exceptions import ConfigError

        key = LLMRole(role)
        if key not in self.roles:
            raise ConfigError(
                "no LLM role configured",
                details={"role": key.value, "configured": sorted(r.value for r in self.roles)},
            )
        return self.roles[key]

    def unit_price(self, target: ProviderTarget) -> dict[str, float] | None:
        return self.pricing.get(target.key)


# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------
class PluginConfig(BaseModel):
    """Connection parameters per interface, keyed by plugin name.

    Example::

        plugins:
          vector_store:
            sqlite: {path: "./data/ragx.db", dim: 256}
    """

    model_config = ConfigDict(extra="forbid")

    parser: dict[str, dict[str, Any]] = Field(default_factory=dict)
    processor: dict[str, dict[str, Any]] = Field(default_factory=dict)
    vector_store: dict[str, dict[str, Any]] = Field(default_factory=dict)
    graph_store: dict[str, dict[str, Any]] = Field(default_factory=dict)
    embedder: dict[str, dict[str, Any]] = Field(default_factory=dict)
    reranker: dict[str, dict[str, Any]] = Field(default_factory=dict)
    object_store: dict[str, dict[str, Any]] = Field(default_factory=dict)
    token_counter: str = "tiktoken-cl100k"

    def for_plugin(self, interface: str, name: str) -> dict[str, Any]:
        """Return the config dict for ``plugins.<interface>[<name>]``."""
        section = getattr(self, interface, {})
        return dict(section.get(name) or {})


class QueueConfig(BaseModel):
    """03-ingestion.md §3.7.3."""

    driver: Literal["in_process", "redis"] = "in_process"
    stream_prefix: str = "ragx:ingest"
    consumer_group: str = "ragx-workers"
    read_count: int = 1
    block_ms: int = 2000
    worker_concurrency: int = 4
    visibility_timeout: int = 1800
    dlq_stream: str = "ragx:ingest:dlq"
    redis_url: str = ""
    #: When true, the API process starts the in-process queue worker on
    #: startup (lite compose smoke: upload → query works out of the box).
    #: Disabled by default so unit/integration tests keep manual control.
    auto_consume: bool = False


class OTelConfig(BaseModel):
    enabled: bool = False
    endpoint: str = ""
    service_name: str = "ragx"
    sampling: float = 1.0


class SecurityConfig(BaseModel):
    api_keys: dict[str, dict[str, Any]] = Field(default_factory=dict)
    jwt_enabled: bool = False
    jwt_secret: str = ""
    rate_limit_rps: float = Field(default=10.0, gt=0)
    rate_limit_burst: int = Field(default=20, ge=1)
    max_upload_mb: int = Field(default=50, ge=1)
    #: Per-tenant monthly LLM token quota. ``0`` disables the quota gate.
    monthly_token_quota: int = Field(default=0, ge=0)
    #: Per-tenant monthly upload (bytes) quota. ``0`` disables.
    monthly_upload_quota_bytes: int = Field(default=0, ge=0)


class MCPConfig(BaseModel):
    """MCP server configuration (09-api.md §9.7.2).

    Disabled by default. When ``enabled``, the FastAPI app exposes the SSE
    transport (``GET /v1/mcp/sse`` + ``POST /v1/mcp/messages``); the stdio
    transport is launched separately via the ``ragx-mcp`` console script.
    """

    enabled: bool = False
    transport: Literal["stdio", "sse"] = "stdio"
    sse_idle_timeout: float = 300.0
    """Max seconds an SSE connection may stay idle (no pushed message) before
    the server closes it. Protects against leaked/hung connections (proxies,
    dead clients). ``<= 0`` disables the idle close (rely on client
    disconnect only)."""


class EvalConfig(BaseModel):
    """10-observability.md §10.4."""

    backend: Literal["ragas", "deepeval"] = "ragas"
    judge_model: str = ""
    custom_metrics: list[str] = Field(default_factory=list)
    tolerance: float = 0.05


class PluginRegistryConfig(BaseModel):
    #: Extra Python import paths scanned for entry points (dev convenience).
    discover_entry_points: bool = True


class Settings(BaseSettings):
    """Global settings. Env prefix ``RAGX_``, nested delimiter ``.``."""

    model_config = SettingsConfigDict(
        env_prefix="RAGX_",
        env_nested_delimiter=".",
        extra="ignore",
    )

    profile: Profile = "lite"
    data_dir: str = "./data"
    plugins: PluginConfig = Field(default_factory=PluginConfig)
    llm: LLMRouterConfig = Field(default_factory=lambda: LLMRouterConfig(roles={}))
    queue: QueueConfig = Field(default_factory=QueueConfig)
    observability: OTelConfig = Field(default_factory=OTelConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    evaluation: EvalConfig = Field(default_factory=EvalConfig)
    registry: PluginRegistryConfig = Field(default_factory=PluginRegistryConfig)
    #: Audit backend: ``"memory"`` (default, lite profile) or ``"metadata"``
    #: (full profile; persists to the metadata DB).
    audit_backend: Literal["memory", "metadata"] = "memory"

    # -- validation --------------------------------------------------------
    def validate_profile(self) -> None:
        """Reject ``profile=full`` when mandatory plugin config is missing (9003)."""
        from ragx.core.exceptions import ConfigError

        if self.profile == "lite":
            return
        missing: list[str] = []
        if not self.plugins.vector_store:
            missing.append("plugins.vector_store")
        if not self.plugins.embedder:
            missing.append("plugins.embedder")
        if not self.llm.roles:
            missing.append("llm.roles")
        if missing:
            raise ConfigError(
                "profile=full requires explicit plugin configuration",
                details={"missing": missing},
            )


# ---------------------------------------------------------------------------
# KB registry row
# ---------------------------------------------------------------------------
class KB(BaseModel):
    """Knowledge base registry row (Metadata DB)."""

    kb_id: str                         # kb_<ulid>
    name: str
    config: KBConfig = Field(default_factory=KBConfig)
    created_at: datetime = Field(default_factory=utcnow)


# ---------------------------------------------------------------------------
# Three-level resolution helper
# ---------------------------------------------------------------------------
def effective_kb_config(
    base: KBConfig,
    override: dict[str, Any] | None = None,
) -> KBConfig:
    """Apply a request-level override to a KB config (level 3 of §2.4).

    Only ``context_token_budget`` and ``agentic_token_budget`` are allowed in
    the override object; anything else raises ``ConfigError(9003)``.
    """

    if not override:
        return base.model_copy(deep=True)
    budgets = {k: v for k, v in override.items() if k in BudgetConfig.model_fields}
    if budgets:
        merged = dict(base.budgets.model_dump())
        merged.update(budgets)
        base = base.model_copy(deep=True)
        base.budgets = BudgetConfig(**merged)
    return base
