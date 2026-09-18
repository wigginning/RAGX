"""Retrieval-layer models (06-retrieval.md §6.2)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ragx.core.models import Chunk


class RetrievalHit(BaseModel):
    """One fused hit with per-route provenance (06-retrieval.md §6.2)."""

    chunk: Chunk
    rrf_score: float                 # normalised fusion score [0, 1]
    sources: list[str] = Field(default_factory=list)
    rerank_score: float | None = None


class RetrievalConfig(BaseModel):
    """Retrieval parameters (06-retrieval.md §6.2)."""

    dense_top_k: int = 20
    bm25_top_k: int = 20
    graph_low_top_k: int = 10
    graph_neighbors_hops: int = 1
    graph_neighbors_limit: int = 30
    graph_high_top_k: int = 3
    rrf_k: int = 60
    rrf_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "dense": 1.0, "bm25": 1.0, "graph_low": 0.8, "graph_high": 0.6,
        }
    )
    fusion_strategy: Literal["rrf", "weighted"] = "rrf"
    rerank_top_k: int = 8
    context_token_budget: int = 4096
    quota_dense_bm25: float = 0.70
    quota_graph: float = 0.30
