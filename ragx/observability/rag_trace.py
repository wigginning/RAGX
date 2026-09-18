"""RAG Trace collection and replay (10-observability.md §10.5).

A RAG Trace captures one query's full retrieval path: every recall route's
hits, rerank scores, assembled context + citations, the LLM call sequence
(with role/model/tokens/cost/retry/fallback), and the final result metadata.
Stored 7 days (§10.5.3) for "why did this answer come out wrong?" debugging.

Lite profile: SQLite ``rag_traces`` table. Full profile: can be configured to
OTel Tempo/Jaeger + a side store.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import aiosqlite
from pydantic import BaseModel, Field

from ragx.core.models import Citation, QueryMode, TokenUsage

#: Default retention (§10.5.3).
TRACE_TTL_DAYS = 7

_SCHEMA = """
CREATE TABLE IF NOT EXISTS rag_traces (
    trace_id   TEXT PRIMARY KEY,
    kb_id      TEXT NOT NULL,
    mode       TEXT NOT NULL,
    query      TEXT NOT NULL,
    payload    TEXT NOT NULL,
    cost_usd   REAL NOT NULL DEFAULT 0.0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_traces_kb ON rag_traces(kb_id);
CREATE INDEX IF NOT EXISTS idx_traces_created ON rag_traces(created_at);
"""


class TraceLLMCall(BaseModel):
    """One LLM call inside a RAG Trace (§10.5.1 ``llm_calls[]``)."""

    role: str
    model: str
    tokens: TokenUsage = Field(default_factory=TokenUsage)
    cost_usd: float = 0.0
    retry_count: int = 0
    fallback_from: str | None = None          # source model after a fallback
    prompt_name: str | None = None
    prompt_version: int | None = None


class TraceRetrieval(BaseModel):
    """Per-route recall results inside a RAG Trace (§10.5.1 ``retrieval``)."""

    dense: list[dict[str, Any]] = Field(default_factory=list)
    bm25: list[dict[str, Any]] = Field(default_factory=list)
    graph_low: list[dict[str, Any]] = Field(default_factory=list)
    graph_high: list[dict[str, Any]] = Field(default_factory=list)


class TraceAssembled(BaseModel):
    """Assembled context summary (§10.5.1 ``assembled``)."""

    tokens: int = 0
    citations: list[str] = Field(default_factory=list)
    budget: int = 0


class RAGTrace(BaseModel):
    """The full replayable trace for one query (§10.5.1)."""

    trace_id: str
    kb_id: str
    mode: QueryMode
    query: str
    retrieval: TraceRetrieval = Field(default_factory=TraceRetrieval)
    rerank: list[dict[str, Any]] = Field(default_factory=list)
    assembled: TraceAssembled = Field(default_factory=TraceAssembled)
    llm_calls: list[TraceLLMCall] = Field(default_factory=list)
    cache_hit: bool = False
    degraded: bool = False
    usage: TokenUsage = Field(default_factory=TokenUsage)
    cost_usd: float = 0.0


class RAGTraceCollector:
    """Accumulates trace data during a single query, then flushes to a store.

    Domain services call :meth:`add_dense_hits` / :meth:`add_llm_call` etc.
    as each stage completes; :meth:`flush` persists the complete trace.
    """

    def __init__(self, trace_id: str, kb_id: str, query: str) -> None:
        self.trace_id = trace_id
        self.kb_id = kb_id
        self.query = query
        self._retrieval = TraceRetrieval()
        self._rerank: list[dict[str, Any]] = []
        self._assembled = TraceAssembled()
        self._llm_calls: list[TraceLLMCall] = []
        self._cache_hit = False
        self._degraded = False
        self._usage = TokenUsage()

    def set_mode(self, mode: QueryMode) -> None:
        self._mode = mode

    def set_cache_hit(self, hit: bool) -> None:
        self._cache_hit = hit

    def set_degraded(self, degraded: bool) -> None:
        self._degraded = degraded

    def add_dense_hits(self, hits: list[Any]) -> None:
        self._retrieval.dense = [_scored_to_dict(h) for h in hits]

    def add_bm25_hits(self, hits: list[Any]) -> None:
        self._retrieval.bm25 = [_scored_to_dict(h) for h in hits]

    def add_graph_low_hits(self, hits: list[Any]) -> None:
        self._retrieval.graph_low = [_scored_to_dict(h) for h in hits]

    def add_graph_high_hits(self, hits: list[Any]) -> None:
        self._retrieval.graph_high = [_scored_to_dict(h) for h in hits]

    def set_rerank(self, hits: list[Any]) -> None:
        self._rerank = [_scored_to_dict(h) for h in hits]

    def set_assembled(self, tokens: int, citations: list[Citation], budget: int) -> None:
        self._assembled = TraceAssembled(
            tokens=tokens,
            citations=[c.chunk_id for c in citations],
            budget=budget,
        )

    def add_llm_call(self, call: TraceLLMCall) -> None:
        self._llm_calls.append(call)
        self._usage += call.tokens
        self._usage.estimated = self._usage.estimated or call.tokens.estimated

    def set_usage(self, usage: TokenUsage) -> None:
        self._usage = usage

    def build(self) -> RAGTrace:
        total_cost = sum(c.cost_usd for c in self._llm_calls)
        return RAGTrace(
            trace_id=self.trace_id,
            kb_id=self.kb_id,
            mode=getattr(self, "_mode", "standard"),
            query=self.query,
            retrieval=self._retrieval,
            rerank=self._rerank,
            assembled=self._assembled,
            llm_calls=list(self._llm_calls),
            cache_hit=self._cache_hit,
            degraded=self._degraded,
            usage=self._usage,
            cost_usd=total_cost,
        )


def _scored_to_dict(hit: Any) -> dict[str, Any]:
    """Flatten a ScoredChunk / RetrievalHit to {chunk_id, score, source}."""
    chunk_id = getattr(getattr(hit, "chunk", hit), "chunk_id", "")
    score = (
        getattr(hit, "rerank_score", None)
        or getattr(hit, "rrf_score", None)
        or getattr(hit, "score", 0.0)
    )
    if score is None:
        score = 0.0
    sources = getattr(hit, "sources", None)
    source = getattr(hit, "source", None) or (sources[0] if sources else "")
    return {"chunk_id": chunk_id, "score": float(score), "source": source}


class RAGTraceStore:
    """SQLite-backed RAG Trace store (lite profile, §10.5.3).

    Shares the same SQLite database as :class:`~ragx.ingestion.store.MetadataStore`
    when the same path is used, or runs standalone.
    """

    def __init__(self, path: str = ":memory:") -> None:
        self.path = path
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self._db = await aiosqlite.connect(self.path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_SCHEMA)
        await self._db.commit()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    def _conn(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("RAGTraceStore not connected")
        return self._db

    async def save(self, trace: RAGTrace) -> None:
        await self._conn().execute(
            """INSERT OR REPLACE INTO rag_traces
               (trace_id, kb_id, mode, query, payload, cost_usd, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (
                trace.trace_id,
                trace.kb_id,
                trace.mode,
                trace.query,
                trace.model_dump_json(),
                trace.cost_usd,
                datetime.now(UTC).isoformat(),
            ),
        )
        await self._conn().commit()

    async def get(self, trace_id: str) -> RAGTrace | None:
        cur = await self._conn().execute(
            "SELECT payload FROM rag_traces WHERE trace_id=?", (trace_id,)
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return RAGTrace.model_validate_json(row["payload"])

    async def list_by_kb(
        self, kb_id: str, *, limit: int = 50, offset: int = 0
    ) -> list[RAGTrace]:
        cur = await self._conn().execute(
            "SELECT payload FROM rag_traces WHERE kb_id=? "
            "ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (kb_id, limit, offset),
        )
        rows = await cur.fetchall()
        return [RAGTrace.model_validate_json(r["payload"]) for r in rows]

    async def cleanup_expired(self, ttl_days: int = TRACE_TTL_DAYS) -> int:
        """Delete traces older than ``ttl_days``. Returns the deletion count."""
        cutoff = (datetime.now(UTC) - timedelta(days=ttl_days)).isoformat()
        cur = await self._conn().execute(
            "DELETE FROM rag_traces WHERE created_at < ?", (cutoff,)
        )
        await self._conn().commit()
        return cur.rowcount or 0
