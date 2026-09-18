"""Document-level dedup + submit flow (03-ingestion.md §3.3).

``submit_ingest`` computes the doc hash, checks ``(kb_id, doc_hash)`` and either
returns the existing task (soft 2004) or creates a new pending task and enqueues
it.
"""

from __future__ import annotations

from ragx.core.exceptions import DuplicateDocumentError
from ragx.core.hashing import compute_doc_hash
from ragx.core.ids import new_id
from ragx.core.models import IngestTask, RawDocument, TaskStatus, utcnow
from ragx.ingestion.queue import TaskQueue
from ragx.ingestion.store import MetadataStore


async def submit_ingest(
    raw: RawDocument, db: MetadataStore, queue: TaskQueue
) -> IngestTask:
    """Submit a document for ingestion. Raises ``DuplicateDocumentError(2004)``
    when the same ``(kb_id, doc_hash)`` already exists (soft, HTTP 200)."""
    raw.doc_hash = compute_doc_hash(raw.content)

    existing = await db.find_doc_by_hash(raw.kb_id, raw.doc_hash)
    if existing is not None:
        assert existing.doc_id is not None, "hash-matched doc must carry an id"
        existing_task = await db.get_task_by_doc(existing.doc_id)
        raise DuplicateDocumentError(
            code=2004,
            message="document already ingested",
            details={
                "existing_doc_id": existing.doc_id,
                "existing_task_id": existing_task.task_id if existing_task else None,
            },
        )

    doc_id = new_id("doc_")
    task_id = new_id("task_")
    now = utcnow()
    task = IngestTask(
        task_id=task_id,
        doc_id=doc_id,
        kb_id=raw.kb_id,
        status=TaskStatus.PENDING,
        attempts=0,
        created_at=now,
        updated_at=now,
    )
    await db.save_doc(raw.model_copy(update={"doc_id": doc_id}))
    await db.save_task(task)
    await queue.enqueue(task_id, kb_id=raw.kb_id)
    return task
