"""Reranker contract (01-spi.md §1.4)."""

from __future__ import annotations

import pytest

from ragx.core.models import Chunk
from ragx.spi.contracts.base import SAMPLE_TEXTS, ContractBase
from ragx.spi.interfaces import Reranker


class RerankerContract(ContractBase):
    async def make(self) -> Reranker:
        return await self.make_reranker()

    async def make_reranker(self) -> Reranker:  # pragma: no cover
        raise NotImplementedError

    @staticmethod
    def _chunks() -> list[Chunk]:
        return [
            Chunk(
                chunk_id=f"chk_{i:04d}",
                doc_id=f"doc_{i % 2}",
                kb_id="kb_t",
                atom_ids=[f"doc_{i % 2}#{i:04d}"],
                text=SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)],
                token_count=10,
                page=i,
            )
            for i in range(len(SAMPLE_TEXTS))
        ]

    @pytest.mark.contract
    async def test_scores_are_normalized(self) -> None:
        """Scores must be normalised to [0, 1] (01-spi.md §1.2, 11.5.7)."""
        reranker = await self._get()
        out = await reranker.rerank("向量检索", self._chunks(), top_k=5)
        assert out
        for res in out:
            assert 0.0 <= res.score <= 1.0, res.score

    @pytest.mark.contract
    async def test_top_k_respected(self) -> None:
        reranker = await self._get()
        chunks = self._chunks()
        for top_k in (1, 3, 50):
            out = await reranker.rerank("向量检索", chunks, top_k=top_k)
            assert len(out) == min(top_k, len(chunks))

    @pytest.mark.contract
    async def test_results_sorted_by_score_desc(self) -> None:
        reranker = await self._get()
        out = await reranker.rerank("向量检索", self._chunks(), top_k=6)
        scores = [r.score for r in out]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.contract
    async def test_all_chunk_ids_are_known(self) -> None:
        reranker = await self._get()
        chunks = self._chunks()
        known = {c.chunk_id for c in chunks}
        out = await reranker.rerank("向量检索", chunks, top_k=len(chunks))
        assert {r.chunk_id for r in out} <= known

    @pytest.mark.contract
    async def test_rerank_is_idempotent(self) -> None:
        reranker = await self._get()
        chunks = self._chunks()
        a = await reranker.rerank("向量检索", chunks, top_k=4)
        b = await reranker.rerank("向量检索", chunks, top_k=4)
        assert [r.chunk_id for r in a] == [r.chunk_id for r in b]

    @pytest.mark.contract
    async def test_empty_chunks(self) -> None:
        reranker = await self._get()
        assert await reranker.rerank("任意查询", [], top_k=5) == []
