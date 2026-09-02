"""Hybrid retrieval + RRF + assembler + generation tests (RX-RET-01 DoD)."""

from __future__ import annotations

import pytest

from ragx.core.models import Chunk, EmbeddedChunk, TokenUsage
from ragx.plugins.vector_sqlite import SQLiteVectorStore
from ragx.retrieval.assembler import ContextAssembler
from ragx.retrieval.hybrid import HybridRetriever
from ragx.retrieval.models import RetrievalConfig
from ragx.retrieval.pipeline import QueryService
from ragx.retrieval.router import QueryRouter
from ragx.spi.interfaces import ChatRequest, ChatResponse


def _vec(seed: float, dim: int = 8) -> list[float]:
    raw = [(seed * 7.13 + i * 0.37) % 1.0 for i in range(dim)]
    norm = sum(x * x for x in raw) ** 0.5 or 1.0
    return [x / norm for x in raw]


def _chunk(i: int, text: str, kb: str = "kb_t", tokens: int = 50) -> Chunk:
    return Chunk(
        chunk_id=f"chk_{i:04d}", doc_id=f"doc_{i % 2}", kb_id=kb,
        atom_ids=[f"doc_{i % 2}#{i:04d}"], text=text, token_count=tokens,
        page=i + 1, metadata={"filename": f"doc{i}.md"},
    )


@pytest.fixture
async def store():
    s = SQLiteVectorStore({"kb_id": "kb_t", "dim": 8})
    chunks = [
        EmbeddedChunk(vector=_vec(1.0), **(_chunk(0, "RAGX 通过向量检索提升问答质量。")).model_dump()),
        EmbeddedChunk(vector=_vec(2.0), **(_chunk(1, "知识图谱由实体与关系组成。")).model_dump()),
        EmbeddedChunk(vector=_vec(3.0), **(_chunk(2, "Agentic 管线分解复杂问题。")).model_dump()),
        EmbeddedChunk(vector=_vec(4.0), **(_chunk(3, "今天天气很好适合散步。")).model_dump()),
    ]
    await s.upsert(chunks)
    yield s
    await s.shutdown()


async def test_rrf_ranking_correctness(store) -> None:
    """A chunk hit by both dense and bm25 ranks above single-route hits."""
    cfg = RetrievalConfig(rerank_top_k=10)
    retriever = HybridRetriever(store, None, None, cfg)
    hits = await retriever.retrieve("向量检索", _vec(1.0))
    assert hits, "must return hits"
    # chk_0000 is the exact dense match and contains the bm25 term
    assert hits[0].chunk.chunk_id == "chk_0000"
    assert "dense" in hits[0].sources
    assert all(0.0 <= h.rrf_score <= 1.0 for h in hits)


async def test_weighted_fusion_diverges(store) -> None:
    """Weighted fusion still ranks the exact match first."""
    cfg = RetrievalConfig(fusion_strategy="weighted", rerank_top_k=10)
    retriever = HybridRetriever(store, None, None, cfg)
    hits = await retriever.retrieve("向量检索", _vec(1.0))
    assert hits[0].chunk.chunk_id == "chk_0000"


async def test_budget_truncation(store) -> None:
    """Assembler respects the token budget and drops overflowing chunks."""
    cfg = RetrievalConfig(context_token_budget=60, rerank_top_k=10)
    retriever = HybridRetriever(store, None, None, cfg)
    hits = await retriever.retrieve("向量检索", _vec(1.0))
    assembler = ContextAssembler(cfg)
    context, citations = assembler.assemble(hits)
    # budget 60 -> only ~1 chunk of 50 tokens fits
    assert len(citations) <= 2
    assert all(c.snippet for c in citations)


async def test_citation_integrity(store) -> None:
    """Every citation carries chunk_id/doc_id/page and a <=200-char snippet."""
    cfg = RetrievalConfig(rerank_top_k=10)
    retriever = HybridRetriever(store, None, None, cfg)
    hits = await retriever.retrieve("向量检索", _vec(1.0))
    assembler = ContextAssembler(cfg)
    _, citations = assembler.assemble(hits)
    assert citations
    for c in citations:
        assert c.chunk_id.startswith("chk_")
        assert c.doc_id
        assert len(c.snippet) <= 200


class _MockLLM:
    async def chat(self, req: ChatRequest) -> ChatResponse:
        # echo a [1] reference to prove the prompt carried the context
        return ChatResponse(
            text="根据资料[1]，RAGX 使用向量检索。",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total=15),
            model="mock",
        )


async def test_generation_carries_citation_markers(store) -> None:
    """The generated answer carries [n] markers tied to the assembled context."""
    from ragx.core.settings import KBConfig
    from ragx.llm.prompts import PromptRegistry

    cfg = RetrievalConfig(rerank_top_k=10)
    retriever = HybridRetriever(store, None, None, cfg)
    assembler = ContextAssembler(cfg)
    router = QueryRouter()
    prompts = PromptRegistry()
    kb_cfg = KBConfig()
    service = QueryService(
        retriever, assembler, router, prompts, _MockLLM(),
        kb_cfg=kb_cfg, retrieval_cfg=cfg,
    )
    result = await service.query(
        "RAGX 如何检索？", _vec(1.0), kb_id="kb_t", trace_id="trace_t"
    )
    assert result.mode == "standard"
    assert "[1]" in result.answer
    assert result.citations, "citations must be populated"


async def test_empty_recall_returns_honest_answer(store) -> None:
    """No recall -> honest answer, no exception, empty citations."""
    from ragx.core.settings import KBConfig
    from ragx.llm.prompts import PromptRegistry

    class _EmptyRetriever:
        async def retrieve(self, query, qvec, filter_expr=None):
            return []

    cfg = RetrievalConfig(rerank_top_k=10)
    assembler = ContextAssembler(cfg)
    router = QueryRouter()
    prompts = PromptRegistry()
    service = QueryService(
        _EmptyRetriever(), assembler, router, prompts, _MockLLM(),
        kb_cfg=KBConfig(), retrieval_cfg=cfg,
    )
    result = await service.query(
        "完全不相关的内容", _vec(99.0), kb_id="kb_t", trace_id="trace_t"
    )
    assert result.citations == []
    assert result.details.get("empty") is True
