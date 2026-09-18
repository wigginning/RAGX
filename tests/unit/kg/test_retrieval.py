"""Dual-Level graph retrieval tests (05-kg.md §5.6, RX-KG-02 DoD).

Covers the three functions in ``ragx.kg.retrieval``:
* ``retrieve_low_level`` — entity subgraph recall
* ``retrieve_high_level`` — topic summary recall (capability-gated)
* ``dual_level_retrieve`` — parallel low+high, dedup, and ``kg_enabled`` flag
  gating (the behaviour difference the RX-KG-02 DoD asks to assert).
"""

from __future__ import annotations

from types import SimpleNamespace

from ragx.core.models import Chunk, Entity, ScoredChunk, SubGraph, TopicSummary
from ragx.kg.retrieval import (
    dual_level_retrieve,
    retrieve_high_level,
    retrieve_low_level,
)
from ragx.retrieval.hybrid import GraphChunkResolver


def _vec(seed: float, dim: int = 8) -> list[float]:
    raw = [(seed * 3.17 + i * 0.29) % 1.0 for i in range(dim)]
    norm = sum(x * x for x in raw) ** 0.5 or 1.0
    return [x / norm for x in raw]


def _entity(entity_id: str, name: str = "E") -> Entity:
    return Entity(
        entity_id=entity_id,
        kb_id="kb_1",
        name=name,
        name_norm=name.lower(),
        type="CONCEPT",
        description=f"{name} 的描述",
        source_doc_ids=["doc_1"],
        confidence=0.9,
        vector=_vec(1.0),
    )


class _GraphStore:
    """Mock GraphStore: scripted seeds, neighbours and topics."""

    def __init__(self, *, supports_community: bool = True) -> None:
        self.supports_community = supports_community
        self.capabilities = SimpleNamespace(supports_community=supports_community)
        self.seed_entities: list = []
        self.subgraph = SubGraph()
        self.topic_summaries: list[TopicSummary] = []
        self.match_calls = 0
        self.neighbors_calls = 0
        self.topics_calls = 0

    async def match_entities(self, query_vector, *, top_k):
        self.match_calls += 1
        return self.seed_entities

    async def neighbors(self, entity_ids, *, hops, limit):
        self.neighbors_calls += 1
        return self.subgraph

    async def topics(self, query_vector, *, top_k):
        self.topics_calls += 1
        return self.topic_summaries


class _Resolver(GraphChunkResolver):
    """Records the entity ids it is asked to resolve."""

    def __init__(self, chunks: list[ScoredChunk] | None = None) -> None:
        # bypass the base constructor (no store needed for the mock)
        self.chunks = chunks or []
        self.last_entity_ids: list[str] = []

    async def resolve_entities(self, entity_ids, top_k):
        self.last_entity_ids = list(entity_ids)
        return self.chunks


def _chunk(chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, doc_id="doc_1", kb_id="kb_1",
        atom_ids=["doc_1#0001"], text="text", token_count=10,
    )


class TestRetrieveLowLevel:
    async def test_returns_empty_without_graph_store(self) -> None:
        out = await retrieve_low_level(
            "q", _vec(1.0), "kb_1", None, _Resolver(), top_k=10
        )
        assert out == []

    async def test_expands_seed_neighbours_and_resolves(self) -> None:
        store = _GraphStore()
        store.seed_entities = [_entity("ent_a")]
        store.subgraph = SubGraph(entities=[_entity("ent_b")])
        resolver = _Resolver([ScoredChunk(chunk=_chunk("chk_1"), score=0.9, source="graph_low")])

        out = await retrieve_low_level(
            "q", _vec(1.0), "kb_1", store, resolver, top_k=10
        )

        assert store.match_calls == 1
        assert store.neighbors_calls == 1
        assert set(resolver.last_entity_ids) == {"ent_a", "ent_b"}
        assert out and out[0].chunk.chunk_id == "chk_1"

    async def test_no_seed_entities_short_circuits(self) -> None:
        store = _GraphStore()
        store.seed_entities = []
        out = await retrieve_low_level(
            "q", _vec(1.0), "kb_1", store, _Resolver(), top_k=10
        )
        assert out == []
        assert store.neighbors_calls == 0


class TestRetrieveHighLevel:
    async def test_skips_when_community_unsupported(self) -> None:
        store = _GraphStore(supports_community=False)
        out = await retrieve_high_level(
            _vec(1.0), "kb_1", store, _Resolver(), top_k=3
        )
        assert out == []
        assert store.topics_calls == 0

    async def test_resolves_member_entity_chunks(self) -> None:
        store = _GraphStore()
        store.topic_summaries = [
            TopicSummary(
                topic_id="tpc_1", kb_id="kb_1", title="主题",
                summary="摘要", member_entity_ids=["ent_a", "ent_b"],
                vector=_vec(1.0),
            )
        ]
        resolver = _Resolver([ScoredChunk(chunk=_chunk("chk_2"), score=0.8, source="graph_high")])

        out = await retrieve_high_level(
            _vec(1.0), "kb_1", store, resolver, top_k=3
        )

        assert store.topics_calls == 1
        assert set(resolver.last_entity_ids) == {"ent_a", "ent_b"}
        assert out and out[0].chunk.chunk_id == "chk_2"


class TestDualLevelRetrieve:
    def _kb_cfg(self, kg_enabled: bool) -> SimpleNamespace:
        return SimpleNamespace(flags=SimpleNamespace(kg_enabled=kg_enabled))

    async def test_flag_disabled_returns_empty(self) -> None:
        """kg_enabled=False → both routes skipped (RX-KG-02 behaviour diff)."""
        store = _GraphStore()
        store.seed_entities = [_entity("ent_a")]
        out = await dual_level_retrieve(
            "q", _vec(1.0), "kb_1", store, _Resolver(),
            kb_cfg=self._kb_cfg(False),
        )
        assert out == []
        assert store.match_calls == 0
        assert store.topics_calls == 0

    async def test_flag_enabled_runs_both_routes_dedup(self) -> None:
        """kg_enabled=True → low+high run; duplicate chunks deduped by id."""
        store = _GraphStore()
        store.seed_entities = [_entity("ent_a")]
        store.subgraph = SubGraph(entities=[_entity("ent_b")])
        store.topic_summaries = [
            TopicSummary(
                topic_id="tpc_1", kb_id="kb_1", title="主题", summary="摘要",
                member_entity_ids=["ent_a"], vector=_vec(1.0),
            )
        ]
        # Same chunk returned by both routes with different scores.
        resolver = _Resolver([
            ScoredChunk(chunk=_chunk("chk_dup"), score=0.9, source="graph_low"),
            ScoredChunk(chunk=_chunk("chk_dup"), score=0.6, source="graph_high"),
            ScoredChunk(chunk=_chunk("chk_only"), score=0.7, source="graph_high"),
        ])

        out = await dual_level_retrieve(
            "q", _vec(1.0), "kb_1", store, resolver,
            kb_cfg=self._kb_cfg(True),
        )

        ids = [s.chunk.chunk_id for s in out]
        assert len(ids) == len(set(ids)), "dedup by chunk_id expected"
        # the surviving chk_dup keeps the higher score (0.9)
        dup = next(s for s in out if s.chunk.chunk_id == "chk_dup")
        assert dup.score == 0.9

    async def test_none_store_or_resolver_returns_empty(self) -> None:
        store = _GraphStore()
        store.seed_entities = [_entity("ent_a")]
        assert await dual_level_retrieve(
            "q", _vec(1.0), "kb_1", None, _Resolver(), kb_cfg=None
        ) == []
        assert await dual_level_retrieve(
            "q", _vec(1.0), "kb_1", store, None, kb_cfg=None
        ) == []
