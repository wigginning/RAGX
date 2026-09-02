"""RAG Trace collection and storage tests (10-observability.md §10.5).

Covers: trace field mapping from domain models, SQLite store round-trip,
and the 7-day TTL cleanup.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from ragx.core.models import Chunk, Citation, ScoredChunk, TokenUsage
from ragx.observability.rag_trace import (
    TRACE_TTL_DAYS,
    RAGTrace,
    RAGTraceCollector,
    RAGTraceStore,
    TraceLLMCall,
)
from ragx.retrieval.models import RetrievalHit


def _make_chunk(cid: str = "chk_1") -> Chunk:
    return Chunk(
        chunk_id=cid, doc_id="doc_1", kb_id="kb_1", atom_ids=["doc_1#0001"],
        text="sample chunk text", token_count=128, page=3,
    )


def _make_hit(cid: str, score: float, sources: list[str]) -> RetrievalHit:
    return RetrievalHit(chunk=_make_chunk(cid), rrf_score=score, sources=sources)


class TestRAGTraceCollector:
    def test_build_trace_has_all_fields(self) -> None:
        c = RAGTraceCollector("trace_1", "kb_1", "what is RAG?")
        c.set_mode("standard")
        c.add_dense_hits([_make_hit("chk_1", 0.93, ["dense"])])
        c.add_bm25_hits([_make_hit("chk_2", 0.71, ["bm25"])])
        c.set_rerank([_make_hit("chk_1", 0.92, ["dense"])])
        c.set_assembled(
            tokens=1800,
            citations=[
                Citation(chunk_id="chk_1", doc_id="doc_1", filename="doc.pdf",
                         page=3, snippet="sample", score=0.92),
            ],
            budget=4096,
        )
        c.add_llm_call(TraceLLMCall(
            role="generate", model="deepseek-v3",
            tokens=TokenUsage(prompt_tokens=1200, completion_tokens=210, total=1410),
            cost_usd=0.0042, retry_count=0,
        ))
        trace = c.build()

        assert trace.trace_id == "trace_1"
        assert trace.kb_id == "kb_1"
        assert trace.mode == "standard"
        assert trace.query == "what is RAG?"
        assert len(trace.retrieval.dense) == 1
        assert trace.retrieval.dense[0]["chunk_id"] == "chk_1"
        assert trace.retrieval.dense[0]["score"] == 0.93
        assert len(trace.retrieval.bm25) == 1
        assert trace.rerank[0]["score"] == 0.92
        assert trace.assembled.tokens == 1800
        assert trace.assembled.citations == ["chk_1"]
        assert trace.assembled.budget == 4096
        assert len(trace.llm_calls) == 1
        assert trace.llm_calls[0].role == "generate"
        assert trace.llm_calls[0].model == "deepseek-v3"
        assert trace.llm_calls[0].fallback_from is None
        assert trace.usage.total == 1410
        assert trace.cost_usd == pytest.approx(0.0042)

    def test_fallback_from_recorded(self) -> None:
        c = RAGTraceCollector("trace_2", "kb_1", "query")
        c.set_mode("standard")
        c.add_llm_call(TraceLLMCall(
            role="generate", model="deepseek-v3",
            cost_usd=0.004, fallback_from="gpt-4o",
        ))
        trace = c.build()
        assert trace.llm_calls[0].fallback_from == "gpt-4o"

    def test_degraded_flag(self) -> None:
        c = RAGTraceCollector("trace_3", "kb_1", "query")
        c.set_degraded(True)
        trace = c.build()
        assert trace.degraded is True

    def test_cache_hit_flag(self) -> None:
        c = RAGTraceCollector("trace_4", "kb_1", "query")
        c.set_cache_hit(True)
        trace = c.build()
        assert trace.cache_hit is True

    def test_scored_chunk_source_mapping(self) -> None:
        """ScoredChunk.source maps to the trace's source field (§10.5.2)."""
        sc = ScoredChunk(chunk=_make_chunk("chk_3"), score=0.66, source="graph_low")
        c = RAGTraceCollector("trace_5", "kb_1", "query")
        c.add_graph_low_hits([sc])
        trace = c.build()
        assert trace.retrieval.graph_low[0]["source"] == "graph_low"
        assert trace.retrieval.graph_low[0]["score"] == 0.66


class TestRAGTraceStore:
    @pytest.fixture()
    async def store(self) -> RAGTraceStore:
        s = RAGTraceStore(":memory:")
        await s.connect()
        yield s
        await s.close()

    async def test_save_and_get_roundtrip(self, store: RAGTraceStore) -> None:
        trace = RAGTrace(
            trace_id="trace_rt", kb_id="kb_1", mode="standard",
            query="test query", cost_usd=0.01,
        )
        await store.save(trace)
        loaded = await store.get("trace_rt")
        assert loaded is not None
        assert loaded.trace_id == "trace_rt"
        assert loaded.kb_id == "kb_1"
        assert loaded.mode == "standard"

    async def test_get_missing_returns_none(self, store: RAGTraceStore) -> None:
        assert await store.get("nonexistent") is None

    async def test_list_by_kb(self, store: RAGTraceStore) -> None:
        for i in range(3):
            await store.save(RAGTrace(
                trace_id=f"trace_{i}", kb_id="kb_1", mode="standard",
                query=f"q{i}",
            ))
        traces = await store.list_by_kb("kb_1", limit=10)
        assert len(traces) == 3

    async def test_cleanup_expired(self, store: RAGTraceStore) -> None:
        await store.save(RAGTrace(
            trace_id="old_trace", kb_id="kb_1", mode="standard", query="old",
        ))
        # Manually backdate the created_at timestamp
        old_ts = (datetime.now(UTC) - timedelta(days=TRACE_TTL_DAYS + 1)).isoformat()
        await store._conn().execute(
            "UPDATE rag_traces SET created_at=? WHERE trace_id=?", (old_ts, "old_trace")
        )
        await store._conn().commit()
        deleted = await store.cleanup_expired()
        assert deleted == 1
        assert await store.get("old_trace") is None

    async def test_fresh_trace_not_cleaned(self, store: RAGTraceStore) -> None:
        await store.save(RAGTrace(
            trace_id="fresh", kb_id="kb_1", mode="standard", query="fresh",
        ))
        deleted = await store.cleanup_expired()
        assert deleted == 0
        assert await store.get("fresh") is not None
