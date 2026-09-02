"""HybridRetriever (06-retrieval.md §6.3).

Parallel recall across dense / bm25 / graph_low / graph_high, RRF or
WeightedRank fusion, then optional rerank. A single-route failure degrades to
an empty route (span warning) rather than blocking the whole query.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ragx.core.models import Chunk, FilterExpr, ScoredChunk
from ragx.retrieval.models import RetrievalConfig, RetrievalHit

logger = logging.getLogger("ragx.retrieval.hybrid")


class GraphChunkResolver:
    """Maps graph hits (entities / topics) to chunks via ``Chunk.metadata["entity_ids"]``."""

    def __init__(self, store: Any) -> None:
        self.store = store

    async def resolve_entities(self, entity_ids: list[str], top_k: int) -> list[ScoredChunk]:
        """Find chunks whose ``entity_ids`` overlap the given entities."""
        if not entity_ids:
            return []
        # brute-force: fetch all chunks and filter by entity_ids overlap.
        # A production store would push this down; the lite store scans.
        hits = await self.store.search_dense([0.0] * 8, top_k=10_000)
        out: list[ScoredChunk] = []
        for h in hits:
            ids = set(h.chunk.metadata.get("entity_ids", []))
            overlap = ids & set(entity_ids)
            if overlap:
                conf = max(
                    (h.chunk.metadata.get("entity_conf", {}) or {}).get(e, 0.0)
                    for e in overlap
                )
                out.append(ScoredChunk(chunk=h.chunk, score=conf, source="graph_low"))
        out.sort(key=lambda s: s.score, reverse=True)
        return out[:top_k]


class HybridRetriever:
    """Four-route recall + fusion + rerank."""

    def __init__(
        self,
        store: Any,
        graph_store: Any | None,
        resolver: GraphChunkResolver | None,
        cfg: RetrievalConfig,
        reranker: Any | None = None,
        *,
        kg_enabled: bool = False,
    ) -> None:
        self.store = store
        self.graph_store = graph_store
        self.resolver = resolver
        self.cfg = cfg
        self.reranker = reranker
        #: Feature flag (KBConfig.flags.kg_enabled, 06-retrieval.md §6.3.1):
        #: graph routes only fire when the flag is on AND a graph store is
        #: wired in. A graph store alone (flag off) must NOT enable the routes.
        self.kg_enabled = kg_enabled

    async def retrieve(
        self,
        query: str,
        qvec: list[float],
        filter_expr: FilterExpr | None = None,
    ) -> list[RetrievalHit]:
        tasks: dict[str, asyncio.Task] = {}
        tasks["dense"] = asyncio.create_task(
            self.store.search_dense(qvec, top_k=self.cfg.dense_top_k, filter_expr=filter_expr)
        )
        if getattr(self.store.capabilities, "supports_bm25", False):
            tasks["bm25"] = asyncio.create_task(
                self.store.search_keyword(query, top_k=self.cfg.bm25_top_k, filter_expr=filter_expr)
            )
        if self.kg_enabled and self.graph_store is not None:
            tasks["graph_low"] = asyncio.create_task(self._graph_low(qvec))
            if getattr(self.graph_store.capabilities, "supports_community", False):
                tasks["graph_high"] = asyncio.create_task(self._graph_high(qvec))

        raw: dict[str, list[ScoredChunk]] = {}
        for name, task in tasks.items():
            try:
                raw[name] = await task
            except Exception as exc:  # noqa: BLE001 - degrade, don't block
                logger.warning("retrieval route %s failed: %s", name, exc)
                raw[name] = []

        fused = self._fuse(raw)
        return await self._rerank(query, fused)

    # -- graph routes -------------------------------------------------------
    async def _graph_low(self, qvec: list[float]) -> list[ScoredChunk]:
        if self.resolver is None or self.graph_store is None:
            return []
        entities = await self.graph_store.match_entities(qvec, top_k=self.cfg.graph_low_top_k)
        if not entities:
            return []
        sub = await self.graph_store.neighbors(
            [e.entity_id for e in entities],
            hops=self.cfg.graph_neighbors_hops,
            limit=self.cfg.graph_neighbors_limit,
        )
        merged = {e.entity_id: e for e in entities}
        merged.update({e.entity_id: e for e in sub.entities})
        return await self.resolver.resolve_entities(list(merged), top_k=self.cfg.dense_top_k)

    async def _graph_high(self, qvec: list[float]) -> list[ScoredChunk]:
        if self.resolver is None or self.graph_store is None:
            return []
        topics = await self.graph_store.topics(qvec, top_k=self.cfg.graph_high_top_k)
        member_ids: list[str] = []
        for t in topics:
            member_ids.extend(t.member_entity_ids)
        return await self.resolver.resolve_entities(member_ids, top_k=self.cfg.dense_top_k)

    # -- fusion -------------------------------------------------------------
    def _fuse(self, raw: dict[str, list[ScoredChunk]]) -> list[RetrievalHit]:
        if self.cfg.fusion_strategy == "weighted":
            return self._weighted_fuse(raw)
        return self._rrf_fuse(raw)

    def _rrf_fuse(self, raw: dict[str, list[ScoredChunk]]) -> list[RetrievalHit]:
        k = self.cfg.rrf_k
        scores: dict[str, float] = {}
        sources: dict[str, set[str]] = {}
        chunks: dict[str, Chunk] = {}
        for route, scored in raw.items():
            w = self.cfg.rrf_weights.get(route, 0.0)
            for rank, sc in enumerate(scored, start=1):
                cid = sc.chunk.chunk_id
                scores[cid] = scores.get(cid, 0.0) + w / (k + rank)
                sources.setdefault(cid, set()).add(sc.source)
                chunks[cid] = sc.chunk
        if not scores:
            return []
        s_vals = list(scores.values())
        smin, smax = min(s_vals), max(s_vals)
        rng = (smax - smin) or 1.0
        return [
            RetrievalHit(
                chunk=chunks[cid],
                rrf_score=(s - smin) / rng,
                sources=sorted(sources[cid]),
            )
            for cid, s in scores.items()
        ]

    def _weighted_fuse(self, raw: dict[str, list[ScoredChunk]]) -> list[RetrievalHit]:
        """WeightedRank: per-route min-max normalise, then weighted sum (06 §6.3.2a)."""
        scores: dict[str, float] = {}
        sources: dict[str, set[str]] = {}
        chunks: dict[str, Chunk] = {}
        for route, scored in raw.items():
            w = self.cfg.rrf_weights.get(route, 0.0)
            if not scored:
                continue
            vals = [s.score for s in scored]
            smin, smax = min(vals), max(vals)
            rng = (smax - smin) or 1.0
            for sc in scored:
                cid = sc.chunk.chunk_id
                norm = (sc.score - smin) / rng
                scores[cid] = scores.get(cid, 0.0) + w * norm
                sources.setdefault(cid, set()).add(sc.source)
                chunks[cid] = sc.chunk
        if not scores:
            return []
        s_vals = list(scores.values())
        smin, smax = min(s_vals), max(s_vals)
        rng = (smax - smin) or 1.0
        return [
            RetrievalHit(
                chunk=chunks[cid],
                rrf_score=(s - smin) / rng,
                sources=sorted(sources[cid]),
            )
            for cid, s in scores.items()
        ]

    # -- rerank -------------------------------------------------------------
    async def _rerank(self, query: str, fused: list[RetrievalHit]) -> list[RetrievalHit]:
        if not fused:
            return []
        top_n = min(len(fused), 20)
        candidates = fused[:top_n]
        if self.reranker is not None:
            results = await self.reranker.rerank(
                query, [h.chunk for h in candidates], top_k=self.cfg.rerank_top_k
            )
            id2score = {r.chunk_id: r.score for r in results}
            out = [
                h.model_copy(update={"rerank_score": id2score[h.chunk.chunk_id]})
                for h in candidates
                if h.chunk.chunk_id in id2score
            ]
            return sorted(
                out,
                key=lambda h: h.rerank_score if h.rerank_score is not None else 0.0,
                reverse=True,
            )[: self.cfg.rerank_top_k]
        return candidates[: self.cfg.rerank_top_k]
