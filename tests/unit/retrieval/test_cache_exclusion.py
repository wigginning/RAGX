"""LLM-02 isolation: query results must never surface semantic-cache chunks.

The cache's vector tier writes synthetic ``__semantic_cache__`` docs into the
retrieval store (08-llm.md §8.5). ``QueryService.query`` must exclude them so
they never appear in answers or citations.
"""

from __future__ import annotations

from typing import Any

import pytest

from ragx.core.models import FilterExpr
from ragx.core.settings import KBConfig
from ragx.llm.prompts import PromptRegistry
from ragx.retrieval.assembler import ContextAssembler
from ragx.retrieval.models import RetrievalConfig
from ragx.retrieval.pipeline import CACHE_DOC_ID, QueryService
from ragx.retrieval.router import QueryRouter
from ragx.spi.interfaces import ChatResponse


class _FakeRetriever:
    def __init__(self) -> None:
        self.last_filter: FilterExpr | None = None

    async def retrieve(
        self, query: str, qvec: list[float], filter_expr: FilterExpr | None = None
    ) -> list[Any]:
        self.last_filter = filter_expr
        return []


class _FakeLLM:
    async def chat(self, req: Any) -> ChatResponse:
        return ChatResponse(text="x", model="m")


def _service() -> QueryService:
    return QueryService(
        _FakeRetriever(),
        ContextAssembler(RetrievalConfig()),
        QueryRouter(),
        PromptRegistry(),
        _FakeLLM(),
        kb_cfg=KBConfig(),
        retrieval_cfg=RetrievalConfig(),
    )


def _has_exclusion(f: FilterExpr | None) -> bool:
    assert f is not None
    return any(
        c.get("field") == "doc_id" and c.get("op") == "ne" and c.get("value") == CACHE_DOC_ID
        for c in f.as_list()
    )


@pytest.mark.asyncio
async def test_query_excludes_cache_docs() -> None:
    svc = _service()
    await svc.query("q", [0.1] * 8, kb_id="kb_x", trace_id="t")
    assert _has_exclusion(svc.retriever.last_filter)


@pytest.mark.asyncio
async def test_query_preserves_caller_filter() -> None:
    svc = _service()
    fe = FilterExpr(and_=[{"field": "page", "op": "ge", "value": 3}])
    await svc.query("q", [0.1] * 8, kb_id="kb_x", trace_id="t", filter_expr=fe)
    f = svc.retriever.last_filter
    assert f is not None
    pairs = {(c.get("field"), c.get("op"), c.get("value")) for c in f.as_list()}
    assert ("page", "ge", 3) in pairs
    assert ("doc_id", "ne", CACHE_DOC_ID) in pairs
