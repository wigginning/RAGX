"""OBS-02 wiring tests (10-observability.md §10.2).

``test_metrics.py`` covers the ``Metrics`` catalogue itself; this module locks
in that the *call sites* actually feed it. Without these tests the counters can
stay permanently at zero while every other test still passes.
"""

from __future__ import annotations

import pytest
from prometheus_client import CollectorRegistry

from ragx.core.models import Chunk
from ragx.core.settings import KBConfig
from ragx.llm.prompts import PromptRegistry
from ragx.observability.metrics import Metrics
from ragx.retrieval.assembler import ContextAssembler
from ragx.retrieval.models import RetrievalConfig, RetrievalHit
from ragx.retrieval.pipeline import QueryService
from ragx.retrieval.router import QueryRouter


def _make_hit(chunk_id: str = "c1", doc_id: str = "d1") -> RetrievalHit:
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        kb_id="kb_w",
        atom_ids=[],
        text="RAGX 使用 dense + BM25 混合检索。",
        token_count=8,
    )
    return RetrievalHit(chunk=chunk, rrf_score=0.9, sources=["dense"])


class _FakeRetriever:
    def __init__(self, hits: list | None = None) -> None:
        self._hits = hits if hits is not None else [_make_hit()]

    async def retrieve(self, query, qvec, filter_expr=None):
        return list(self._hits)


class _FakeLLM:
    async def chat(self, req):
        from ragx.spi.interfaces import ChatResponse

        return ChatResponse(text="根据资料[1]，RAGX 使用混合检索。", model="mock")


def _service(metrics: Metrics, retriever) -> QueryService:
    cfg = RetrievalConfig()
    return QueryService(
        retriever,
        ContextAssembler(cfg),
        QueryRouter(),
        PromptRegistry(),
        _FakeLLM(),
        kb_cfg=KBConfig(),
        retrieval_cfg=cfg,
        metrics=metrics,
    )


@pytest.fixture()
def metrics() -> Metrics:
    return Metrics(CollectorRegistry())


@pytest.mark.asyncio
async def test_query_records_total_and_stage_latency(metrics: Metrics) -> None:
    """A successful standard query records the counter + retrieve/llm latency."""
    service = _service(metrics, _FakeRetriever())
    await service.query("RAGX 如何检索？", [0.1] * 8, kb_id="kb_w", trace_id="t1")

    output = metrics.render().decode()
    assert 'ragx_query_total{kb="kb_w",mode="standard",status="ok"} 1.0' in output
    assert 'ragx_query_latency_seconds_count{kb="kb_w",mode="standard",stage="retrieve"} 1.0' in output
    assert 'ragx_query_latency_seconds_count{kb="kb_w",mode="standard",stage="llm"} 1.0' in output
    assert 'ragx_query_latency_seconds_count{kb="kb_w",mode="standard",stage="total"} 1.0' in output


@pytest.mark.asyncio
async def test_empty_recall_records_empty_status(metrics: Metrics) -> None:
    """Empty recall (4002) is counted instead of silently succeeding."""
    service = _service(metrics, _FakeRetriever(hits=[]))
    result = await service.query("无关内容", [0.9] * 8, kb_id="kb_w", trace_id="t2")

    assert result.details.get("empty") is True
    output = metrics.render().decode()
    assert 'ragx_query_total{kb="kb_w",mode="standard",status="empty"} 1.0' in output
    assert 'ragx_retrieval_recall_empty_total{kb="kb_w"} 1.0' in output


@pytest.mark.asyncio
async def test_query_falls_back_to_process_singleton() -> None:
    """No injected metrics -> the process-wide singleton is used, no crash."""
    cfg = RetrievalConfig()
    svc = QueryService(
        _FakeRetriever(),
        ContextAssembler(cfg),
        QueryRouter(),
        PromptRegistry(),
        _FakeLLM(),
        kb_cfg=KBConfig(),
        retrieval_cfg=cfg,
    )
    result = await svc.query("RAGX 如何检索？", [0.1] * 8, kb_id="kb_w", trace_id="t3")
    assert result.mode == "standard"
