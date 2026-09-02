"""GET /v1/traces — RAG Trace list + replay (10-observability.md §10.5).

Exposes the replayable per-query trace collected by :class:`~ragx.retrieval.
pipeline.QueryService` (OBS-04). ``GET /v1/traces`` lists traces for a kb
(newest first); ``GET /v1/traces/{trace_id}`` returns one full trace for
"why did this answer come out wrong?" debugging.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ragx.observability.rag_trace import RAGTrace

router = APIRouter()


class TraceListResponse(BaseModel):
    traces: list[RAGTrace] = Field(default_factory=list)
    count: int = 0
    trace_id: str


@router.get("/traces", response_model=TraceListResponse)
async def list_traces(
    request: Request,
    kb_id: str = "default",
    limit: int = 50,
    offset: int = 0,
) -> TraceListResponse:
    """List recent RAG Traces for a kb (newest first, §10.5)."""
    state: Any = request.state
    store = getattr(request.app.state, "trace_store", None)
    if store is None:
        return TraceListResponse(traces=[], count=0, trace_id=state.trace_id)
    traces = await store.list_by_kb(kb_id, limit=limit, offset=offset)
    return TraceListResponse(
        traces=traces, count=len(traces), trace_id=state.trace_id
    )


@router.get("/traces/{trace_id}", response_model=RAGTrace)
async def get_trace(request: Request, trace_id: str) -> RAGTrace:
    """Replay one full RAG Trace by id (404 when missing, §10.5)."""
    store = getattr(request.app.state, "trace_store", None)
    if store is None:
        raise HTTPException(status_code=404, detail="trace store unavailable")
    trace = await store.get(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"trace {trace_id} not found")
    return trace
