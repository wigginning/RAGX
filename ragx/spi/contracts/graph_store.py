"""GraphStore contract (01-spi.md §1.4)."""

from __future__ import annotations

import pytest

from ragx.core.models import Entity, Relation, SubGraph
from ragx.spi.contracts.base import ContractBase
from ragx.spi.interfaces import GraphStore


def _vec(seed: float, dim: int = 8) -> list[float]:
    raw = [(seed * 3.17 + i * 0.29) % 1.0 for i in range(dim)]
    norm = sum(x * x for x in raw) ** 0.5 or 1.0
    return [x / norm for x in raw]


class GraphStoreContract(ContractBase):
    async def make(self) -> GraphStore:
        return await self.make_graph_store()

    async def make_graph_store(self) -> GraphStore:  # pragma: no cover
        raise NotImplementedError

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _entity(name: str, doc_id: str, *, etype: str = "CONCEPT",
                seed: float = 1.0, desc: str = "", conf: float = 0.9) -> Entity:
        return Entity(
            entity_id=f"ent_{name}",
            kb_id="kb_contract",
            name=name,
            name_norm=name.strip().lower(),
            type=etype,
            description=desc or f"{name} 的描述",
            source_doc_ids=[doc_id],
            confidence=conf,
            vector=_vec(seed),
        )

    @staticmethod
    def _relation(head: str, tail: str, doc_id: str, rtype: str = "uses") -> Relation:
        return Relation(
            relation_id=f"rel_{head}_{tail}",
            kb_id="kb_contract",
            head_id=head,
            tail_id=tail,
            type=rtype,
            description=f"{head} {rtype} {tail}",
            weight=1.0,
            source_doc_ids=[doc_id],
        )

    # -- suite -------------------------------------------------------------
    @pytest.mark.contract
    async def test_capabilities_declared(self) -> None:
        store = await self._get()
        assert isinstance(store.capabilities.supports_community, bool)
        assert isinstance(store.name, str) and store.name

    @pytest.mark.contract
    async def test_upsert_entities_idempotent_and_merged(self) -> None:
        """Same name_norm from two docs -> one entity, union of source docs (05-kg §5.3)."""
        store = await self._get()
        e1 = self._entity("LightRAG", "doc_a")
        e2 = Entity(**{**e1.model_dump(), "source_doc_ids": ["doc_b"], "confidence": 0.7})
        await store.upsert_entities([e1])
        await store.upsert_entities([e2])
        await store.upsert_entities([e1])  # exact repeat
        matched = await store.match_entities(e1.vector, top_k=5)
        hits = [m for m in matched if m.name_norm == e1.name_norm]
        assert hits, "upserted entity must be retrievable"
        merged = hits[0]
        assert set(merged.source_doc_ids) == {"doc_a", "doc_b"}
        assert merged.confidence >= 0.8, "merge keeps the higher confidence"

    @pytest.mark.contract
    async def test_upsert_relations_idempotent(self) -> None:
        store = await self._get()
        await store.upsert_entities(
            [self._entity("A", "doc_a", seed=1), self._entity("B", "doc_a", seed=2)]
        )
        rel = self._relation("ent_A", "ent_B", "doc_a")
        await store.upsert_relations([rel])
        await store.upsert_relations([rel])
        sub = await store.neighbors(["ent_A"], hops=1, limit=20)
        rels = [r for r in sub.relations if r.head_id == "ent_A" and r.tail_id == "ent_B"]
        assert len(rels) == 1, "duplicate relation rows after idempotent upsert"

    @pytest.mark.contract
    async def test_neighbors_expands_hops(self) -> None:
        store = await self._get()
        nodes = [
            self._entity("A", "doc_a", seed=1),
            self._entity("B", "doc_a", seed=2),
            self._entity("C", "doc_a", seed=3),
        ]
        await store.upsert_entities(nodes)
        await store.upsert_relations([
            self._relation("ent_A", "ent_B", "doc_a"),
            self._relation("ent_B", "ent_C", "doc_a"),
        ])
        one_hop = await store.neighbors(["ent_A"], hops=1, limit=20)
        assert isinstance(one_hop, SubGraph)
        one_ids = {e.entity_id for e in one_hop.entities}
        assert "ent_B" in one_ids
        assert "ent_C" not in one_ids, "1-hop must not reach C"
        two_hop = await store.neighbors(["ent_A"], hops=2, limit=20)
        assert "ent_C" in {e.entity_id for e in two_hop.entities}

    @pytest.mark.contract
    async def test_delete_by_doc_decrements_reference_count(self) -> None:
        """Shared nodes survive; private nodes are removed (01-spi.md §1.2, 05-kg §5.4)."""
        store = await self._get()
        shared = Entity(
            entity_id="ent_shared", kb_id="kb_contract", name="共享实体",
            name_norm="共享实体", type="CONCEPT", description="被两篇文档提及",
            source_doc_ids=["doc_a", "doc_b"], confidence=0.9, vector=_vec(5.0),
        )
        private = self._entity("私有实体", "doc_a", seed=6.0)
        other = self._entity("邻居", "doc_a", seed=7.0)
        await store.upsert_entities([shared, private, other])
        await store.upsert_relations([
            self._relation("ent_shared", "ent_邻居", "doc_a"),
        ])

        await store.delete_by_doc("doc_a")

        survived = await store.match_entities(_vec(5.0), top_k=10)
        ids = {e.entity_id for e in survived}
        assert "ent_shared" in ids, "entity shared with doc_b must survive delete_by_doc(doc_a)"
        kept = next(e for e in survived if e.entity_id == "ent_shared")
        assert kept.source_doc_ids == ["doc_b"], kept.source_doc_ids
        assert "ent_私有实体" not in ids, "doc_a-only entity must be deleted"

        # delete the last reference -> gone
        await store.delete_by_doc("doc_b")
        survived2 = {e.entity_id for e in await store.match_entities(_vec(5.0), top_k=10)}
        assert "ent_shared" not in survived2

    @pytest.mark.contract
    async def test_match_entities_ranks_by_similarity(self) -> None:
        store = await self._get()
        await store.upsert_entities([
            self._entity("近", "doc_a", seed=1.0),
            self._entity("远", "doc_a", seed=9.0),
        ])
        hits = await store.match_entities(_vec(1.0), top_k=2)
        assert hits
        assert hits[0].name == "近"

    @pytest.mark.contract
    async def test_topics_if_supported(self) -> None:
        """High-level retrieval needs community detection (06-retrieval §6.3.1)."""
        store = await self._get()
        if not store.capabilities.supports_community:
            pytest.skip("supports_community=False")
        await store.upsert_entities([
            self._entity(f"E{i}", "doc_a", seed=float(i + 1)) for i in range(5)
        ])
        topics = await store.topics(_vec(1.0), top_k=3)
        assert topics
        assert all(t.kb_id == "kb_contract" for t in topics)
        assert topics[0].title and topics[0].summary
        assert topics[0].member_entity_ids

    @pytest.mark.contract
    async def test_empty_store_returns_empty(self) -> None:
        store = await self._get()
        assert await store.match_entities(_vec(1.0), top_k=5) == []
        assert await store.neighbors(["nope"], hops=1, limit=5) == SubGraph()
        if store.capabilities.supports_community:
            assert await store.topics(_vec(1.0), top_k=3) == []
