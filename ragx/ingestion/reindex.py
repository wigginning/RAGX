"""Incremental reindex (03-ingestion.md §3.5).

Triggered by a chunk edit (PUT full-text / PATCH metadata). Never rebuilds the
whole document - only re-indexes the affected chunk and its graph nodes, with
an optimistic ``version`` lock (3002 on mismatch).

Invalidation propagation (08-llm.md §8.5.3): whenever a chunk is edited or
deleted, the semantic cache for the owning kb is invalidated so subsequent
queries don't surface a stale answer. Pass ``cache=None`` to skip (the
critical path doesn't always wire one in).
"""

from __future__ import annotations

import logging
from typing import Any

from ragx.core.exceptions import ChunkNotFoundError, EditConflictError
from ragx.core.models import Chunk, EmbeddedChunk
from ragx.core.tokens import count_tokens
from ragx.ingestion.store import MetadataStore

logger = logging.getLogger("ragx.ingestion.reindex")


async def incremental_reindex(
    chunk_id: str,
    db: MetadataStore,
    vector_store: Any,
    embedder: Any,
    *,
    new_text: str | None = None,
    new_meta: dict[str, Any] | None = None,
    version_expected: int,
    graph_store: Any | None = None,
    kg_enabled: bool = False,
    llm: Any = None,
    kb_cfg: Any | None = None,
    cache: Any | None = None,
) -> Chunk:
    """Re-index one edited chunk. Returns the updated chunk."""
    chunk = await db.get_chunk(chunk_id)
    if chunk is None:
        raise ChunkNotFoundError(
            code=3001, message="chunk not found", details={"chunk_id": chunk_id}
        )
    if chunk.version != version_expected:
        raise EditConflictError(
            code=3002,
            message="version mismatch",
            details={"expected": version_expected, "actual": chunk.version},
        )

    kb_id = chunk.kb_id

    # step 1: remove the old chunk's vector + graph associations
    await vector_store.delete([chunk_id])
    if kg_enabled and graph_store is not None:
        await graph_store.delete_by_doc(chunk.doc_id)

    # step 2: rewrite the chunk (version++, edited=True)
    if new_text is not None:
        chunk.text = new_text
        chunk.token_count = count_tokens(new_text)
    if new_meta is not None:
        chunk.metadata.update(new_meta)
    chunk.version += 1
    chunk.edited = True
    await db.save_chunks([chunk])

    # step 3: re-embed + upsert
    vectors = await embedder.embed([chunk.text])
    embedded = EmbeddedChunk(**chunk.model_dump(), vector=vectors[0])
    await vector_store.upsert([embedded])

    # step 4 (optional): rebuild the graph for this chunk (05-kg.md §5.4.3)
    if kg_enabled and graph_store is not None and llm is not None and kb_cfg is not None:
        from ragx.kg.builder import KGBuilder

        builder = KGBuilder(llm, graph_store, embedder, kb_cfg)
        await builder.build_for_chunks([chunk])

    # step 5: invalidate the semantic cache for this kb (08 §8.5.3).
    # Best-effort: a cache failure must not roll back the chunk edit.
    if cache is not None:
        try:
            invalidated = await cache.invalidate_kb(kb_id)
            logger.info(
                "semantic cache invalidated (kb=%s, success=%s) after chunk %s edit",
                kb_id,
                invalidated,
                chunk_id,
            )
        except Exception as exc:  # noqa: BLE001 - degrade, never block
            logger.warning("cache invalidate_kb failed: %s", exc)

    return chunk
