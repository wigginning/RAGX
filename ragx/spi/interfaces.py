"""SPI - the seven plugin contracts (01-spi.md).

Design principle: *narrow and small* - each interface does one thing. Signatures
freeze at v1.0 (01-spi.md §1.5).

Layering note: the canonical :class:`LLMRole` lives in
``ragx.core.roles`` because both :class:`ChatRequest` (this module) and
``core.settings.LLMRouterConfig`` need it; ``ragx.llm.roles`` re-exports it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from ragx.core.models import (
    Atom,
    AtomDescription,
    AtomType,
    Chunk,
    EmbeddedChunk,
    Entity,
    FilterExpr,
    RawDocument,
    Relation,
    ScoredChunk,
    SubGraph,
    TokenUsage,
    TopicSummary,
)
from ragx.core.roles import LLMRole
from ragx.core.settings import ParseOptions

#: Entry point group per interface (01-spi.md §1.1).
ENTRY_POINT_GROUPS: dict[str, str] = {  # group name -> entry-point group string
    "ragx.parsers": "parser",
    "ragx.processors": "processor",
    "ragx.embedders": "embedder",
    "ragx.rerankers": "reranker",
    "ragx.vector_stores": "vector_store",
    "ragx.graph_stores": "graph_store",
    "ragx.llm_providers": "llm_provider",
}

# ---------------------------------------------------------------------------
# Shared request / response models (kept in core+spi types only, 01-spi §1.2)
# ---------------------------------------------------------------------------
class ParseResult(BaseModel):
    atoms: list[Atom] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DescribeOptions(BaseModel):
    """Processor options (timeout is enforced by the caller, 01-spi.md §1.2)."""

    timeout_s: float = 30.0
    prompt_name: str | None = None      # e.g. describe_image / describe_table
    metadata: dict[str, Any] = Field(default_factory=dict)


class RerankResult(BaseModel):
    chunk_id: str
    score: float                        # normalised to [0, 1]


class VectorStoreCapabilities(BaseModel):
    supports_bm25: bool = False
    supports_filter: bool = False


class GraphStoreCapabilities(BaseModel):
    supports_community: bool = False


class LLMCapabilities(BaseModel):
    supports_json_mode: bool = False
    supports_stream: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    """Router -> provider request (08-llm.md §8.3 uses ``req.role`` / ``req.kb_id``)."""

    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessage] = Field(default_factory=list)
    role: LLMRole = LLMRole.GENERATE
    model: str | None = None            # provider-level override
    temperature: float = 0.2
    max_tokens: int | None = None
    json_mode: bool = False
    kb_id: str = "default"
    trace_id: str | None = None
    cacheable: bool = False             # semantic-cache eligible (08 §8.5.1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def messages_text(self) -> str:
        """Flattened message text (semantic-cache key input, 08 §8.5.1)."""
        return "\n".join(m.content for m in self.messages)


class ChatChunk(BaseModel):
    delta: str
    finish_reason: str | None = None


class ChatResponse(BaseModel):
    """Provider response. ``usage`` is mandatory for the Cost Ledger (01-spi §1.2)."""

    text: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    model: str | None = None
    cached: bool = False
    raw: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# The seven protocols
# ---------------------------------------------------------------------------
@runtime_checkable
class Parser(Protocol):
    """Document -> atomic content units (01-spi.md §1.2)."""

    name: str
    supported_mimetypes: list[str]

    async def parse(self, doc: RawDocument, *, options: ParseOptions) -> ParseResult: ...


@runtime_checkable
class Processor(Protocol):
    """Atomic units -> structured descriptions (VLM / OCR)."""

    name: str
    supported_atom_types: list[AtomType]

    async def describe(
        self, atoms: list[Atom], *, options: DescribeOptions
    ) -> list[AtomDescription]: ...


@runtime_checkable
class Embedder(Protocol):
    name: str
    dimension: int
    max_batch_size: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class Reranker(Protocol):
    name: str

    async def rerank(
        self, query: str, chunks: list[Chunk], *, top_k: int
    ) -> list[RerankResult]: ...


@runtime_checkable
class VectorStore(Protocol):
    name: str
    capabilities: VectorStoreCapabilities

    async def upsert(self, chunks: list[EmbeddedChunk]) -> None: ...

    async def delete(self, chunk_ids: list[str]) -> None: ...

    async def search_dense(
        self, vector: list[float], *, top_k: int, filter_expr: FilterExpr | None = None
    ) -> list[ScoredChunk]: ...

    async def search_keyword(
        self, query: str, *, top_k: int, filter_expr: FilterExpr | None = None
    ) -> list[ScoredChunk]: ...

    async def startup(self) -> None: ...

    async def shutdown(self) -> None: ...


@runtime_checkable
class GraphStore(Protocol):
    name: str
    capabilities: GraphStoreCapabilities

    async def upsert_entities(self, entities: list[Entity]) -> None: ...

    async def upsert_relations(self, relations: list[Relation]) -> None: ...

    async def delete_by_doc(self, doc_id: str) -> None: ...

    async def match_entities(self, query_vector: list[float], *, top_k: int) -> list[Entity]: ...

    async def neighbors(
        self, entity_ids: list[str], *, hops: int, limit: int
    ) -> SubGraph: ...

    async def topics(self, query_vector: list[float], *, top_k: int) -> list[TopicSummary]: ...

    async def startup(self) -> None: ...

    async def shutdown(self) -> None: ...


@runtime_checkable
class LLMProvider(Protocol):
    name: str
    capabilities: LLMCapabilities

    async def chat(self, req: ChatRequest) -> ChatResponse: ...

    def chat_stream(self, req: ChatRequest) -> AsyncIterator[ChatChunk]: ...

    async def structured(
        self, req: ChatRequest, *, schema: type[BaseModel]
    ) -> BaseModel: ...

    async def startup(self) -> None: ...

    async def shutdown(self) -> None: ...


#: Canonical name -> protocol class (01-spi.md §1.1).
INTERFACES: dict[str, type] = {
    "parser": Parser,
    "processor": Processor,
    "embedder": Embedder,
    "reranker": Reranker,
    "vector_store": VectorStore,
    "graph_store": GraphStore,
    "llm_provider": LLMProvider,
}
