"""VectorStore contract (01-spi.md §1.4).

Plugin suites override ``DIMENSION`` with the dimension configured for their
store; the suite uses it to assert the vector-length contract.
"""

from __future__ import annotations

import pytest

from ragx.core.exceptions import PluginContractError
from ragx.core.hashing import cosine_similarity
from ragx.core.models import Chunk, EmbeddedChunk, FilterExpr
from ragx.spi.contracts.base import SAMPLE_TEXTS, ContractBase
from ragx.spi.interfaces import VectorStore


class VectorStoreContract(ContractBase):
    #: Vector dimension the store under test was configured with.
    DIMENSION: int = 8

    async def make(self) -> VectorStore:
        return await self.make_store()

    async def make_store(self) -> VectorStore:  # pragma: no cover
        raise NotImplementedError

    # -- helpers -----------------------------------------------------------
    @classmethod
    def _dim(cls) -> int:
        return cls.DIMENSION

    @staticmethod
    def _vec(seed: float, dim: int) -> list[float]:
        """Deterministic, distinct unit-ish vectors indexed by ``seed``."""
        raw = [(seed * 7.13 + i * 0.37) % 1.0 for i in range(dim)]
        norm = sum(x * x for x in raw) ** 0.5 or 1.0
        return [x / norm for x in raw]

    @classmethod
    def _chunks(cls, n: int = 5) -> list[EmbeddedChunk]:
        out: list[EmbeddedChunk] = []
        for i in range(n):
            chunk = Chunk(
                chunk_id=f"chk_{i:04d}",
                doc_id=f"doc_{i % 2}",
                kb_id="kb_contract",
                atom_ids=[f"doc_{i % 2}#{i:04d}"],
                text=SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)],
                token_count=12,
                page=i + 1,
                metadata={"page": i + 1, "tag": "ragx" if i % 2 == 0 else "other"},
            )
            out.append(EmbeddedChunk(vector=cls._vec(i + 1.0, cls._dim()), **chunk.model_dump()))
        return out

    # -- suite (01-spi.md §1.4 list) ---------------------------------------
    @pytest.mark.contract
    async def test_capabilities_declared(self) -> None:
        store = await self._get()
        caps = store.capabilities
        assert isinstance(caps.supports_bm25, bool)
        assert isinstance(caps.supports_filter, bool)
        assert caps.supports_filter, "filters are mandatory (06-retrieval.md §6.6.1)"
        assert isinstance(store.name, str) and store.name

    @pytest.mark.contract
    async def test_upsert_idempotent(self) -> None:
        """Repeated upsert must not duplicate results (03-ingestion.md §3.6.1)."""
        store = await self._get()
        chunks = self._chunks(4)
        await store.upsert(chunks)
        await store.upsert(chunks)
        hits = await store.search_dense(
            chunks[0].vector, top_k=self._dim() * 4 + 10
        )
        ids = [h.chunk.chunk_id for h in hits]
        assert len(ids) == len(set(ids)), "duplicate chunk ids after idempotent upsert"
        assert len(ids) == len(chunks)

    @pytest.mark.contract
    async def test_search_dense_returns_scored(self) -> None:
        """Scores ordered descending; top_k honoured; source tag is 'dense'."""
        store = await self._get()
        chunks = self._chunks(6)
        await store.upsert(chunks)
        probe = chunks[2].vector  # exact match must rank first
        hits = await store.search_dense(probe, top_k=3)
        assert hits and len(hits) <= 3
        assert hits[0].chunk.chunk_id == chunks[2].chunk_id
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)
        assert all(0.0 <= s <= 1.0 for s in scores)
        assert all(h.source == "dense" for h in hits)
        # the store must return similarity, so an exact probe scores ~1.0
        assert hits[0].score > 0.99, hits[0].score

    @pytest.mark.contract
    async def test_search_dense_never_leaks_other_kb(self) -> None:
        """Store instances are kb-scoped: rows from another kb are unreachable.

        Either the store rejects the foreign chunk on upsert (PluginContractError)
        or it stores it but never returns it - both are valid isolation semantics.
        """
        store = await self._get()
        await store.upsert(self._chunks(3))
        foreign = Chunk(
            chunk_id="chk_foreign",
            doc_id="doc_x",
            kb_id="kb_someone_else",
            atom_ids=["doc_x#0000"],
            text="应该看不到的内容",
            token_count=3,
            metadata={},
        )
        try:
            await store.upsert(
                [EmbeddedChunk(vector=self._vec(99.0, self._dim()), **foreign.model_dump())]
            )
        except PluginContractError:
            pass  # rejected at write time - equally acceptable
        hits = await store.search_dense(self._vec(1.0, self._dim()), top_k=50)
        assert all(h.chunk.kb_id != "kb_someone_else" for h in hits)

    @pytest.mark.contract
    async def test_filter_semantics(self) -> None:
        store = await self._get()
        await store.upsert(self._chunks(5))

        eq = await store.search_dense(
            self._vec(1.0, self._dim()), top_k=50,
            filter_expr=FilterExpr(**{"and": [{"field": "page", "op": "eq", "value": 3}]}),
        )
        assert [h.chunk.chunk_id for h in eq] == ["chk_0002"]

        ge = await store.search_dense(
            self._vec(1.0, self._dim()), top_k=50,
            filter_expr=FilterExpr(**{"and": [{"field": "page", "op": "ge", "value": 4}]}),
        )
        assert {h.chunk.chunk_id for h in ge} == {"chk_0003", "chk_0004"}

        contains = await store.search_dense(
            self._vec(1.0, self._dim()), top_k=50,
            filter_expr=FilterExpr(**{"and": [{"field": "tag", "op": "contains", "value": "ragx"}]}),
        )
        assert {h.chunk.chunk_id for h in contains} == {"chk_0000", "chk_0002", "chk_0004"}

        missing_field = await store.search_dense(
            self._vec(1.0, self._dim()), top_k=50,
            filter_expr=FilterExpr(**{"and": [{"field": "nope", "op": "eq", "value": 1}]}),
        )
        assert missing_field == []

    @pytest.mark.contract
    async def test_delete_removes_from_search(self) -> None:
        store = await self._get()
        chunks = self._chunks(4)
        await store.upsert(chunks)
        await store.delete(["chk_0001", "chk_0002"])
        hits = await store.search_dense(self._vec(1.0, self._dim()), top_k=50)
        assert {h.chunk.chunk_id for h in hits} == {"chk_0000", "chk_0003"}
        # deleting again is a no-op, not an error (01-spi.md §1.2 re-entrant)
        await store.delete(["chk_0001"])
        assert await store.search_dense(self._vec(1.0, self._dim()), top_k=50)

    @pytest.mark.contract
    async def test_dimension_mismatch_rejected(self) -> None:
        """A wrong-length vector is a plugin bug -> PluginContractError (01-spi §1.2)."""
        store = await self._get()
        bad = Chunk(
            chunk_id="chk_bad", doc_id="doc_b", kb_id="kb_contract",
            atom_ids=["doc_b#0000"], text="bad", token_count=1, metadata={},
        )
        with pytest.raises(PluginContractError):
            await store.upsert([EmbeddedChunk(vector=[0.1, 0.2], **bad.model_dump())])

    @pytest.mark.contract
    async def test_keyword_search_if_supported(self) -> None:
        store = await self._get()
        self._skip_if_absent(store, "supports_bm25", "store has no keyword route")
        if not store.capabilities.supports_bm25:
            pytest.skip("supports_bm25=False")
        await store.upsert(self._chunks(5))
        hits = await store.search_keyword("向量检索", top_k=3)
        assert hits and len(hits) <= 3
        assert all(h.source == "bm25" for h in hits)
        assert all(h.score >= 0.0 for h in hits)

    @pytest.mark.contract
    async def test_keyword_search_honours_filter(self) -> None:
        store = await self._get()
        if not store.capabilities.supports_bm25:
            pytest.skip("supports_bm25=False")
        await store.upsert(self._chunks(5))
        hits = await store.search_keyword(
            "ragx", top_k=50,
            filter_expr=FilterExpr(**{"and": [{"field": "page", "op": "eq", "value": 1}]}),
        )
        assert all(h.chunk.metadata.get("page") == 1 for h in hits)

    @pytest.mark.contract
    async def test_dense_similarity_is_consistent(self) -> None:
        """The reported score must be the real cosine similarity (06-retrieval §6.3.2)."""
        store = await self._get()
        a = Chunk(chunk_id="chk_a", doc_id="d", kb_id="kb_contract",
                  atom_ids=["d#0"], text="甲", token_count=1, metadata={})
        b = Chunk(chunk_id="chk_b", doc_id="d", kb_id="kb_contract",
                  atom_ids=["d#1"], text="乙", token_count=1, metadata={})
        va, vb = self._vec(1.0, self._dim()), self._vec(2.0, self._dim())
        await store.upsert([
            EmbeddedChunk(vector=va, **a.model_dump()),
            EmbeddedChunk(vector=vb, **b.model_dump()),
        ])
        hits = await store.search_dense(va, top_k=2)
        top = {h.chunk.chunk_id: h.score for h in hits}
        assert abs(top["chk_a"] - cosine_similarity(va, va)) < 1e-6
