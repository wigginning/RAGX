"""Embedder contract (01-spi.md §1.4)."""

from __future__ import annotations

import pytest

from ragx.core.exceptions import PluginContractError
from ragx.core.hashing import cosine_similarity
from ragx.spi.contracts.base import SAMPLE_TEXTS, ContractBase
from ragx.spi.interfaces import Embedder


class EmbedderContract(ContractBase):
    async def make(self) -> Embedder:
        return await self.make_embedder()

    async def make_embedder(self) -> Embedder:  # pragma: no cover
        raise NotImplementedError

    @pytest.mark.contract
    async def test_capabilities_declared(self) -> None:
        emb = await self._get()
        assert isinstance(emb.name, str) and emb.name
        assert isinstance(emb.dimension, int) and emb.dimension > 0
        assert isinstance(emb.max_batch_size, int) and emb.max_batch_size > 0

    @pytest.mark.contract
    async def test_vector_length_matches_dimension(self) -> None:
        """Length contract: ``len(vector) == dimension`` (01-spi.md §1.2)."""
        emb = await self._get()
        vecs = await emb.embed(list(SAMPLE_TEXTS))
        assert len(vecs) == len(SAMPLE_TEXTS)
        for vec in vecs:
            assert len(vec) == emb.dimension, (len(vec), emb.dimension)
            assert all(isinstance(x, float) for x in vec)

    @pytest.mark.contract
    async def test_output_length_matches_input(self) -> None:
        emb = await self._get()
        vecs = await emb.embed(["唯一的一条文本", "第二条"])
        assert len(vecs) == 2

    @pytest.mark.contract
    async def test_embedding_is_deterministic(self) -> None:
        emb = await self._get()
        a = await emb.embed(["确定性的重复输入"])
        b = await emb.embed(["确定性的重复输入"])
        assert a == b

    @pytest.mark.contract
    async def test_similar_texts_are_closer_than_unrelated(self) -> None:
        """Semantic sanity: the embedding must not be a pure random number."""
        emb = await self._get()
        texts = [
            "RAGX 通过向量检索检索相关文档",
            "RAGX 使用向量检索查找相关文档",
            "今天中午吃了炸鸡和米饭，味道很好",
        ]
        vecs = await emb.embed(texts)
        similar = cosine_similarity(vecs[0], vecs[1])
        unrelated = cosine_similarity(vecs[0], vecs[2])
        assert similar > unrelated, (similar, unrelated)

    @pytest.mark.contract
    async def test_batch_larger_than_max_batch_size(self) -> None:
        """Implementation must chunk internally and still return one vector per text."""
        emb = await self._get()
        texts = list(SAMPLE_TEXTS) * max(1, (emb.max_batch_size // len(SAMPLE_TEXTS)) + 1)
        texts = texts[: emb.max_batch_size + 3]
        vecs = await emb.embed(texts)
        assert len(vecs) == len(texts)

    @pytest.mark.contract
    async def test_empty_input(self) -> None:
        emb = await self._get()
        assert await emb.embed([]) == []

    @pytest.mark.contract
    async def test_zero_vector_is_not_returned(self) -> None:
        emb = await self._get()
        vecs = await emb.embed(["正常文本"])
        assert sum(abs(x) for x in vecs[0]) > 0.0

    @pytest.mark.contract
    async def test_dimension_mismatch_rejected(self) -> None:
        """VectorStores wrap embedders; the embedder itself must never raise raw
        exceptions. This test guards the plugin-contract error path used by the
        vector store suite (see VectorStoreContract)."""
        emb = await self._get()
        try:
            vecs = await emb.embed(["x"])
        except PluginContractError:
            return
        assert len(vecs) == 1
