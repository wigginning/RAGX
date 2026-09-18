"""Domain models (02-core.md §2.1).

All models live in ``core`` because they are the data currency passed between
layers (00-overview.md §0.1 rule 2). ``kb_id`` / ``doc_id`` etc. are plain
strings so that plugins never import ragx internals.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: Bounding tuple for :attr:`Atom.bbox` / :attr:`Chunk.bbox`.
BBox = tuple[float, float, float, float]

QueryMode = Literal["fast", "standard", "agentic"]
RetrievalSource = Literal["dense", "bm25", "graph_low", "graph_high"]


def utcnow() -> datetime:
    """UTC aware now - the single time source for all persisted timestamps."""
    return datetime.now(UTC)


# ---------------------------------------------------------------------------
# 2.1.1 ingestion side
# ---------------------------------------------------------------------------
class RawDocument(BaseModel):
    """A document as submitted to ``POST /v1/documents`` (02-core.md §2.1.1)."""

    model_config = ConfigDict(frozen=False)

    doc_id: str | None = None          # assigned on submit (doc_<ulid>)
    kb_id: str
    filename: str
    mimetype: str
    content: bytes | str               # bytes = binary file, str = raw text
    source_uri: str | None = None
    doc_hash: str = ""                 # filled by the ingestion submit path
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class AtomType(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    FORMULA = "formula"


class Atom(BaseModel):
    """Atomic content unit produced by a Parser (02-core.md §2.1.1).

    ``atom_id`` is ``f"{doc_id}#{seq:04d}"`` and MUST be stable across reruns
    for the same input + options (01-spi.md §1.2 parser contract, 04-chunking
    §4.3.4 relies on it for overlap markers).
    """

    atom_id: str
    doc_id: str
    type: AtomType
    text: str | None = None            # TEXT / TABLE(markdown) / FORMULA(latex)
    payload_ref: str | None = None     # ObjectStore key for IMAGE
    page: int | None = None
    bbox: BBox | None = None           # x0, y0, x1, y1 normalised to [0, 1]
    content_hash: str = ""             # description-cache key (§3.4.1)
    context: str | None = None         # section title / caption
    metadata: dict[str, Any] = Field(default_factory=dict)


class AtomDescription(BaseModel):
    """Structured description of a non-text atom (VLM / OCR output)."""

    atom_id: str
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    model: str                         # cost attribution
    cached: bool = False               # observability: cache hit?


# ---------------------------------------------------------------------------
# 2.1.2 retrieval side
# ---------------------------------------------------------------------------
class Chunk(BaseModel):
    """Chunk - the retrieval unit (02-core.md §2.1.2, 04-chunking.md)."""

    chunk_id: str                      # chk_<ulid>
    doc_id: str
    kb_id: str
    atom_ids: list[str]
    text: str
    token_count: int
    page: int | None = None
    bbox: BBox | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    edited: bool = False
    version: int = 1


class EmbeddedChunk(Chunk):
    """Chunk with its embedding attached (SPI VectorStore.upsert input)."""

    vector: list[float]


class ScoredChunk(BaseModel):
    """One retrieval hit. ``score`` is route-local (dense cosine, BM25, entity
    confidence, topic weight) - the fusion layer is responsible for any
    cross-route comparison (06-retrieval.md §6.3.2)."""

    chunk: Chunk
    score: float
    source: RetrievalSource


class Entity(BaseModel):
    entity_id: str                     # ent_<ulid>
    kb_id: str
    name: str
    name_norm: str                     # dedup key (05-kg.md §5.3)
    type: str                          # PERSON/ORG/CONCEPT/...
    description: str
    source_doc_ids: list[str]          # reference count for incremental delete
    confidence: float = Field(ge=0.0, le=1.0)
    vector: list[float] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Relation(BaseModel):
    relation_id: str                   # rel_<ulid>
    kb_id: str
    head_id: str
    tail_id: str
    type: str
    description: str
    weight: float = 1.0
    source_doc_ids: list[str] = Field(default_factory=list)


class SubGraph(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)


class TopicSummary(BaseModel):
    """High-level retrieval unit (05-kg.md §5.5)."""

    topic_id: str                      # tpc_<ulid>
    kb_id: str
    title: str
    summary: str
    member_entity_ids: list[str] = Field(default_factory=list)
    vector: list[float] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 2.1.3 query / generation side
# ---------------------------------------------------------------------------
_ALLOWED_FILTER_OPS: tuple[str, ...] = ("eq", "ne", "gt", "ge", "lt", "le", "in", "contains")

#: Symbolic aliases accepted in addition to the canonical names above.
#:
#: NOTE (design conflict, 02-core.md §2.1.3 vs its own example / 06 §6.6.1 /
#: 09 §9.4.2): the op *list* names textual operators (``ge`` ...) while every
#: usage example writes ``">="``. Both forms are accepted and normalised to the
#: canonical textual name so no caller has to choose.
_SYMBOLIC_OPS: dict[str, str] = {
    ">=": "ge",
    "<=": "le",
    ">": "gt",
    "<": "lt",
    "=": "eq",
    "==": "eq",
    "!=": "ne",
    "<>": "ne",
}


def normalize_filter_op(op: str) -> str:
    """Map a symbolic operator to its canonical name; raises ``ValueError``."""
    canonical = _SYMBOLIC_OPS.get(op, op)
    if canonical not in _ALLOWED_FILTER_OPS:
        raise ValueError(f"filter op {op!r} not in {_ALLOWED_FILTER_OPS}")
    return canonical


class FilterExpr(BaseModel):
    """Metadata filter DSL (02-core.md §2.1.3).

    Shape: ``{"and": [{"field": "page", "op": ">=", "value": 3}, ...]}``.
    Operators are normalised to ``eq|ne|gt|ge|lt|le|in|contains``.
    """

    and_: list[dict[str, Any]] | None = Field(default=None, alias="and")
    or_: list[dict[str, Any]] | None = Field(default=None, alias="or")

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("and_", "or_")
    @classmethod
    def _normalize_clauses(
        cls, value: list[dict[str, Any]] | None
    ) -> list[dict[str, Any]] | None:
        if value is None:
            return None
        for idx, clause in enumerate(value):
            if not isinstance(clause, dict):
                raise ValueError(f"filter clause {idx} must be an object")
            if "field" not in clause or "op" not in clause:
                raise ValueError(f"filter clause {idx} requires 'field' and 'op'")
            clause["op"] = normalize_filter_op(str(clause["op"]))
        return value

    def as_list(self) -> list[dict[str, Any]]:
        """All clauses, and-group first then or-group (for downstream pushdown)."""
        return list(self.and_ or []) + list(self.or_ or [])


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    filename: str
    page: int | None = None
    snippet: str                       # <= 200 chars
    score: float


class TokenUsage(BaseModel):
    """Token accounting (01-spi.md §1.2: ChatResponse.usage contract)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total: int = 0
    #: True when the provider did not report usage and we estimated (08 §8.7).
    estimated: bool = False

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total=self.total + other.total,
            estimated=self.estimated or other.estimated,
        )

    def __iadd__(self, other: TokenUsage) -> TokenUsage:
        return self.__add__(other)


class QueryResult(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    mode: QueryMode
    trace_id: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    cost_usd: float = 0.0
    degraded: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class TaskStatus(StrEnum):
    """IngestTask state machine (03-ingestion.md §3.2)."""

    PENDING = "pending"
    PARSING = "parsing"
    PROCESSING = "processing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    KG_BUILDING = "kg_building"
    DONE = "done"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in (TaskStatus.DONE, TaskStatus.FAILED)


#: Ordered working stages (KG excluded: it is optional and only runs last).
INGEST_STAGES: tuple[TaskStatus, ...] = (
    TaskStatus.PARSING,
    TaskStatus.PROCESSING,
    TaskStatus.CHUNKING,
    TaskStatus.EMBEDDING,
    TaskStatus.KG_BUILDING,
)


class IngestTask(BaseModel):
    task_id: str                       # task_<ulid>
    doc_id: str
    kb_id: str
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    error: str | None = None
    attempts: int = 0
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# Request-level override (02-core.md §2.4 whitelist, 06-retrieval.md §6.9)
# ---------------------------------------------------------------------------
_REQUEST_OVERRIDE_WHITELIST: tuple[str, ...] = ("rerank_top_k", "budgets", "mode")


class RequestOverride(BaseModel):
    """Request-scoped override. Only whitelisted keys are honoured.

    ``from_dict`` rejects any non-whitelisted key with ``ConfigError(9003)``
    so that a typo cannot silently change retrieval behaviour.
    """

    rerank_top_k: int | None = Field(default=None, ge=1, le=50)
    budgets: dict[str, int | float] | None = None
    mode: Literal["auto", "fast", "standard", "agentic"] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> RequestOverride:
        from ragx.core.exceptions import ConfigError

        if not data:
            return cls()
        unknown = sorted(set(data) - set(_REQUEST_OVERRIDE_WHITELIST))
        if unknown:
            raise ConfigError(
                "unknown override key(s)",
                details={"unknown": unknown, "whitelist": list(_REQUEST_OVERRIDE_WHITELIST)},
            )
        budgets = data.get("budgets")
        if budgets is not None and not isinstance(budgets, dict):
            raise ConfigError("override.budgets must be an object", details={"value": budgets})
        return cls(
            rerank_top_k=data.get("rerank_top_k"),
            budgets=budgets,
            mode=data.get("mode"),
        )
