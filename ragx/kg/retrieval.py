"""Dual-Level graph retrieval (05-kg.md §5.6).

Two parallel graph recall routes, gated by ``FeatureFlags.kg_enabled``:

* **Low-Level** (§5.6.1): entity subgraph recall — ``match_entities`` seeds →
  ``neighbors(hops=1)`` expansion → associated chunks → ``ScoredChunk(source="graph_low")``.
* **High-Level** (§5.6.2): topic summary recall — ``topics`` vector search →
  member entities → associated chunks → ``ScoredChunk(source="graph_high")``.

The HybridRetriever (06-retrieval.md) orchestrates both routes and fuses the
results with dense/bm25. This module provides the per-route logic.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ragx.core.models import ScoredChunk
from ragx.retrieval.hybrid import GraphChunkResolver

logger = logging.getLogger("ragx.kg.retrieval")

#: Multiplier for top_k in scope-discovery (§7.3.2) and retrieval.
_LOW_TOP_K = 10
_HIGH_TOP_K = 3


async def retrieve_low_level(
    query: str,
    query_vec: list[float],
    kb_id: str,
    graph_store: Any,
    resolver: GraphChunkResolver,
    top_k: int = _LOW_TOP_K,
) -> list[ScoredChunk]:
    """Entity subgraph recall (§5.6.1).

    1. Seed entities: ``match_entities(query_vec, top_k)``
    2. Expand subgraph: ``neighbors(entity_ids, hops=1, limit=50)``
    3. Resolve to chunks: ``resolver.resolve_entities(entity_ids, top_k)``
    4. Return ``ScoredChunk(source="graph_low")``
    """
    if graph_store is None:
        return []

    seed_entities = await graph_store.match_entities(query_vec, top_k=top_k)
    if not seed_entities:
        return []

    subgraph = await graph_store.neighbors(
        [e.entity_id for e in seed_entities], hops=1, limit=50
    )

    # Merge seed + neighbour entities
    merged: dict[str, Any] = {e.entity_id: e for e in seed_entities}
    for e in subgraph.entities:
        merged[e.entity_id] = e

    return await resolver.resolve_entities(list(merged), top_k=top_k * 2)


async def retrieve_high_level(
    query_vec: list[float],
    kb_id: str,
    graph_store: Any,
    resolver: GraphChunkResolver,
    top_k: int = _HIGH_TOP_K,
) -> list[ScoredChunk]:
    """Topic summary recall (§5.6.2).

    1. Topic search: ``topics(query_vec, top_k)``
    2. Expand to member entities → source chunks
    3. Return ``ScoredChunk(source="graph_high")``
    """
    if graph_store is None:
        return []

    caps = getattr(graph_store, "capabilities", None)
    if caps is not None and not getattr(caps, "supports_community", False):
        return []

    topics = await graph_store.topics(query_vec, top_k=top_k)
    if not topics:
        return []

    member_ids: list[str] = []
    for t in topics:
        member_ids.extend(t.member_entity_ids)

    if not member_ids:
        return []

    return await resolver.resolve_entities(member_ids, top_k=top_k * 2)


async def dual_level_retrieve(
    query: str,
    query_vec: list[float],
    kb_id: str,
    graph_store: Any,
    resolver: GraphChunkResolver | None,
    kb_cfg: Any | None = None,
    top_k: int = _LOW_TOP_K,
) -> list[ScoredChunk]:
    """Dual-level retrieval: low + high routes in parallel (§5.6.3).

    Gated by ``kb_cfg.flags.kg_enabled`` — returns ``[]`` when disabled.
    Results are deduplicated by ``chunk_id`` (higher score wins).
    """
    if kb_cfg is not None and not kb_cfg.flags.kg_enabled:
        return []
    if graph_store is None or resolver is None:
        return []

    low, high = await asyncio.gather(
        retrieve_low_level(query, query_vec, kb_id, graph_store, resolver, top_k),
        retrieve_high_level(query_vec, kb_id, graph_store, resolver, top_k),
    )

    # Deduplicate by chunk_id, keeping the higher score.
    seen: dict[str, ScoredChunk] = {}
    for sc in low + high:
        cid = sc.chunk.chunk_id
        if cid not in seen or sc.score > seen[cid].score:
            seen[cid] = sc

    return list(seen.values())
