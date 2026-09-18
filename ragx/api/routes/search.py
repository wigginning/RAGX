"""POST /v1/search (09-api.md §9.4.2)."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from ragx.api.middleware import require_kb_access
from ragx.core.models import FilterExpr
from ragx.retrieval.filters import validate_filter

router = APIRouter()


class SearchRequest(BaseModel):
    kb_id: str
    query: str
    top_k: int = Field(default=8, ge=1, le=50)
    filter: FilterExpr | None = None
    rerank: bool = True
    mode: str = "standard"


class SearchResultItem(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    page: int | None = None
    score: float
    source: str


class SearchResult(BaseModel):
    results: list[SearchResultItem] = Field(default_factory=list)
    trace_id: str


@router.post("/search", response_model=SearchResult)
async def search(req: SearchRequest, request: Request) -> SearchResult:
    validate_filter(req.filter)
    require_kb_access(request, req.kb_id)
    # resolve kb-scoped plugins for this request's kb
    registry = request.app.state.registry
    kb_cfg = request.app.state.kb_cfg
    embedder = registry.resolve_from_kb("embedder", kb_cfg, kb_id=req.kb_id)
    store = registry.resolve_from_kb("vector_store", kb_cfg, kb_id=req.kb_id)
    graph_store = registry.resolve_from_kb("graph_store", kb_cfg, kb_id=req.kb_id) \
        if kb_cfg.graph_store else None
    from ragx.retrieval.hybrid import HybridRetriever
    from ragx.retrieval.models import RetrievalConfig

    retriever = HybridRetriever(store, graph_store, None, RetrievalConfig())
    qvec = (await embedder.embed([req.query]))[0]
    hits = await retriever.retrieve(req.query, qvec, req.filter)
    items = [
        SearchResultItem(
            chunk_id=h.chunk.chunk_id,
            doc_id=h.chunk.doc_id,
            text=h.chunk.text,
            page=h.chunk.page,
            score=h.rerank_score if h.rerank_score is not None else h.rrf_score,
            source=",".join(h.sources),
        )
        for h in hits[: req.top_k]
    ]
    return SearchResult(results=items, trace_id=request.state.trace_id)
