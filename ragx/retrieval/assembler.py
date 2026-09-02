"""ContextAssembler (06-retrieval.md §6.4).

Assembles the final context under a hard token budget, splitting hits into a
dense/bm25 pool and a graph pool by primary route, with overflow reclaim.
"""

from __future__ import annotations

from ragx.core.models import Citation
from ragx.retrieval.models import RetrievalConfig, RetrievalHit


class ContextAssembler:
    def __init__(self, cfg: RetrievalConfig) -> None:
        self.cfg = cfg

    def assemble(self, hits: list[RetrievalHit]) -> tuple[str, list[Citation]]:
        budget = self.cfg.context_token_budget
        q_db, q_g = self.cfg.quota_dense_bm25, self.cfg.quota_graph
        pool_db = budget * q_db / (q_db + q_g)
        pool_g = budget * q_g / (q_db + q_g)

        db: list[RetrievalHit] = []
        g: list[RetrievalHit] = []
        for h in hits:
            is_graph = any(s.startswith("graph") for s in h.sources)
            (g if is_graph else db).append(h)

        used = 0
        citations: list[Citation] = []
        parts: list[str] = []

        def fill(group: list[RetrievalHit], limit: int) -> None:
            nonlocal used
            for h in group:
                tc = h.chunk.token_count
                if used + tc > budget:
                    break
                parts.append(h.chunk.text)
                citations.append(
                    Citation(
                        chunk_id=h.chunk.chunk_id,
                        doc_id=h.chunk.doc_id,
                        filename=h.chunk.metadata.get("filename", ""),
                        page=h.chunk.page,
                        snippet=h.chunk.text[:200],
                        score=h.rerank_score if h.rerank_score is not None else h.rrf_score,
                    )
                )
                used += tc

        fill(db, pool_db)
        fill(g, pool_g)  # overflow from the first pool is reclaimed here
        return "\n\n".join(parts), citations
