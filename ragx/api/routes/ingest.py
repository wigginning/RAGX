"""Ingest routes (09-api.md §9.4.3 / §9.4.4 / §9.4.5 / §9.4.6 / §9.4.7)."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, File, Form, Request, UploadFile
from pydantic import BaseModel

from ragx.api.middleware import require_kb_access
from ragx.core.exceptions import (
    ChunkNotFoundError,
    DuplicateDocumentError,
    EditConflictError,
    TaskNotFoundError,
)
from ragx.core.models import RawDocument
from ragx.ingestion.dedup import submit_ingest
from ragx.ingestion.reindex import incremental_reindex

router = APIRouter()


class SubmitResponse(BaseModel):
    doc_id: str
    task_id: str
    trace_id: str
    duplicate: bool = False


class ChunkUpdateRequest(BaseModel):
    text: str
    version: int
    metadata: dict[str, Any] | None = None


class ChunkPatchRequest(BaseModel):
    metadata: dict[str, Any]
    version: int


@router.post("/documents", response_model=SubmitResponse, status_code=202)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    kb_id: str = Form(...),
    metadata: str = Form("{}"),
) -> SubmitResponse:
    require_kb_access(request, kb_id)
    content = await file.read()
    raw = RawDocument(
        kb_id=kb_id,
        filename=file.filename or "unnamed",
        mimetype=file.content_type or "application/octet-stream",
        content=content,
        metadata=json.loads(metadata or "{}"),
    )
    db = request.app.state.db
    queue = request.app.state.queue
    try:
        task = await submit_ingest(raw, db, queue)
    except DuplicateDocumentError as e:
        return SubmitResponse(
            doc_id=e.details["existing_doc_id"],
            task_id=e.details.get("existing_task_id") or "",
            trace_id=request.state.trace_id,
            duplicate=True,
        )
    return SubmitResponse(
        doc_id=task.doc_id, task_id=task.task_id, trace_id=request.state.trace_id
    )


@router.get("/tasks/{task_id}")
async def get_task(task_id: str, request: Request) -> dict:
    task = await request.app.state.db.get_task(task_id)
    if task is None:
        raise TaskNotFoundError(
            code=2003, message="task not found", details={"task_id": task_id}
        )
    require_kb_access(request, task.kb_id)
    return {
        "task_id": task.task_id,
        "doc_id": task.doc_id,
        "kb_id": task.kb_id,
        "status": task.status.value,
        "progress": task.progress,
        "error": task.error,
        "attempts": task.attempts,
        "created_at": task.created_at.isoformat(),
        "updated_at": task.updated_at.isoformat(),
    }


@router.get("/documents/{doc_id}/chunks")
async def list_chunks(doc_id: str, request: Request) -> dict:
    chunks = await request.app.state.db.get_chunks_by_doc(doc_id)
    if chunks:
        require_kb_access(request, chunks[0].kb_id)
    return {"chunks": [c.model_dump() for c in chunks]}


@router.put("/chunks/{chunk_id}")
async def put_chunk(chunk_id: str, req: ChunkUpdateRequest, request: Request) -> dict:
    db = request.app.state.db
    chunk = await db.get_chunk(chunk_id)
    if chunk is None:
        raise ChunkNotFoundError(
            code=3001, message="chunk not found", details={"chunk_id": chunk_id}
        )
    require_kb_access(request, chunk.kb_id)
    if chunk.version != req.version:
        raise EditConflictError(
            code=3002,
            message="version conflict",
            details={"expected": chunk.version, "provided": req.version},
        )
    updated = await incremental_reindex(
        chunk_id,
        db,
        request.app.state.vector_store,
        request.app.state.embedder,
        new_text=req.text,
        new_meta=req.metadata,
        version_expected=req.version,
        cache=getattr(request.app.state, "semantic_cache", None),
    )
    return updated.model_dump()


@router.patch("/chunks/{chunk_id}")
async def patch_chunk(chunk_id: str, req: ChunkPatchRequest, request: Request) -> dict:
    db = request.app.state.db
    chunk = await db.get_chunk(chunk_id)
    if chunk is None:
        raise ChunkNotFoundError(
            code=3001, message="chunk not found", details={"chunk_id": chunk_id}
        )
    require_kb_access(request, chunk.kb_id)
    if chunk.version != req.version:
        raise EditConflictError(
            code=3002,
            message="version conflict",
            details={"expected": chunk.version, "provided": req.version},
        )
    chunk.metadata.update(req.metadata)
    chunk.version += 1
    await db.save_chunks([chunk])
    return chunk.model_dump()
