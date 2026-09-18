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


class TestQueryServiceTraceCapture:
    """OBS-04 DoD: a query produces a replayable trace (§10.5)."""

    async def test_query_persists_replayable_trace(self) -> None:
        from ragx.core.settings import KBConfig
        from ragx.llm.prompts import PromptRegistry
        from ragx.retrieval.models import RetrievalConfig
        from ragx.retrieval.pipeline import QueryService

        store = RAGTraceStore(":memory:")
        await store.connect()
        hits = [_make_hit("chk_1", 0.9, ["dense"])]
        cit = [Citation(chunk_id="chk_1", doc_id="doc_1", filename="f",
                        page=1, snippet="s", score=0.9)]
        svc = QueryService(
            _FakeRetriever(hits), _FakeAssembler("ctx", cit), _FakeRouter(),
            PromptRegistry(), _FakeLLM(text="the answer"),
            kb_cfg=KBConfig(), retrieval_cfg=RetrievalConfig(),
            trace_store=store,
        )
        res = await svc.query("what?", [0.1] * 8, kb_id="kb_1", trace_id="trace_x")
        assert res.answer == "the answer"

        trace = await store.get("trace_x")
        assert trace is not None
        assert trace.mode == "standard"
        assert trace.query == "what?"
        assert len(trace.retrieval.dense) == 1
        assert trace.retrieval.dense[0]["chunk_id"] == "chk_1"
        assert trace.assembled.citations == ["chk_1"]
        assert len(trace.llm_calls) == 1
        assert trace.llm_calls[0].role == "generate"
        await store.close()

    async def test_query_without_trace_store_is_noop(self) -> None:
        from ragx.core.settings import KBConfig
        from ragx.llm.prompts import PromptRegistry
        from ragx.retrieval.models import RetrievalConfig
        from ragx.retrieval.pipeline import QueryService

        # No trace_store wired → query must still succeed, no trace captured.
        svc = QueryService(
            _FakeRetriever([_make_hit("chk_1", 0.9, ["dense"])]),
            _FakeAssembler("ctx", []), _FakeRouter(),
            PromptRegistry(), _FakeLLM(text="ans"),
            kb_cfg=KBConfig(), retrieval_cfg=RetrievalConfig(),
        )
        res = await svc.query("q?", [0.1] * 8, kb_id="kb_1", trace_id="t_nostore")
        assert res.answer == "ans"


class _FakeRetriever:
    def __init__(self, hits: list[RetrievalHit]) -> None:
        self._hits = hits

    async def retrieve(self, query, qvec, filter_expr=None):
        return self._hits


class _FakeAssembler:
    def __init__(self, context: str, citations: list[Citation]) -> None:
        self._context = context
        self._citations = citations

    def assemble(self, hits):
        return self._context, self._citations


class _FakeLLM:
    def __init__(self, text: str = "ans", model: str = "m", usage=None) -> None:
        self._text = text
        self._model = model
        self._usage = usage or TokenUsage()

    async def chat(self, req):
        class _Resp:
            pass

        r = _Resp()
        r.text = self._text
        r.model = self._model
        r.usage = self._usage
        r.cost_usd = 0.0
        return r


class _FakeRouter:
    def __init__(self, mode: str = "standard") -> None:
        self._mode = mode

    async def route(self, query, kb_cfg, override):
        return self._mode
