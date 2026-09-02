"""KGBuilder tests (05-kg.md §5.4 + §5.9).

Covers: build_for_chunks (extract → filter → merge → upsert),
delete_chunk_from_graph (reference counting, shared entity retention),
and relation mapping.
"""

from __future__ import annotations

from ragx.core.exceptions import ExtractionSchemaError
from ragx.core.models import Chunk, Entity, Relation
from ragx.core.settings import KBConfig
from ragx.kg.builder import KGBuilder
from ragx.kg.schemas import ExtractedEntity, ExtractedRelation, ExtractionResult


def _make_chunk(text: str = "text", kb_id: str = "kb_1", doc_id: str = "doc_1") -> Chunk:
    return Chunk(
        chunk_id="chk_1", doc_id=doc_id, kb_id=kb_id,
        atom_ids=[f"{doc_id}#0001"], text=text, token_count=10,
    )


def _make_store() -> dict:
    """A dict mimicking NetworkXGraphStore.entities (name_norm → Entity)."""
    return {}


class _MockStore:
    """Mock GraphStore with internal entity dict + upsert/delete methods."""

    def __init__(self) -> None:
        self.entities: dict[str, Entity] = {}
        self.relations: dict[tuple, Relation] = {}
        self.upsert_entities_calls: list[list[Entity]] = []
        self.upsert_relations_calls: list[list[Relation]] = []
        self.delete_entity_calls: list[str] = []

    async def upsert_entities(self, entities: list[Entity]) -> None:
        self.upsert_entities_calls.append(entities)
        for e in entities:
            self.entities[e.name_norm] = e

    async def upsert_relations(self, relations: list[Relation]) -> None:
        self.upsert_relations_calls.append(relations)
        for r in relations:
            self.relations[(r.head_id, r.tail_id, r.type)] = r

    async def match_entities(self, vec, *, top_k):
        return list(self.entities.values())

    async def delete_entity(self, entity_id: str) -> None:
        self.delete_entity_calls.append(entity_id)
        for norm, ent in list(self.entities.items()):
            if ent.entity_id == entity_id:
                del self.entities[norm]
                break

    async def delete_relations_by_entity(self, entity_ids: list[str]) -> None:
        id_set = set(entity_ids)
        for key, rel in list(self.relations.items()):
            if rel.head_id in id_set or rel.tail_id in id_set:
                del self.relations[key]


class _MockLLM:
    def __init__(self, result: ExtractionResult) -> None:
        self._result = result
        self.call_count = 0

    async def structured(self, req, *, schema):
        self.call_count += 1
        return self._result


class _MockEmbedder:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class TestBuildForChunks:
    async def test_build_extracts_and_upserts(self) -> None:
        store = _MockStore()
        llm = _MockLLM(ExtractionResult(
            entities=[
                ExtractedEntity(name="OpenAI", type="ORG",
                                description="AI company", confidence=0.9),
            ],
            relations=[],
        ))
        builder = KGBuilder(llm, store, _MockEmbedder(), KBConfig())
        chunks = [_make_chunk()]

        await builder.build_for_chunks(chunks)

        assert len(store.upsert_entities_calls) == 1
        entities_upserted = store.upsert_entities_calls[0]
        assert len(entities_upserted) == 1
        assert entities_upserted[0].name == "OpenAI"
        assert "doc_1" in entities_upserted[0].source_doc_ids
        assert llm.call_count == 1

    async def test_build_creates_relations(self) -> None:
        """Extracted relations map to Entity IDs (§5.9)."""
        store = _MockStore()
        # Pre-populate with an entity so the relation head/tail resolves
        from ragx.core.ids import new_id

        ent_a = Entity(
            entity_id=new_id("ent_"), kb_id="kb_1", name="A",
            name_norm="a", type="ORG", description="d",
            source_doc_ids=["doc_1"], confidence=0.9,
        )
        ent_b = Entity(
            entity_id=new_id("ent_"), kb_id="kb_1", name="B",
            name_norm="b", type="ORG", description="d",
            source_doc_ids=["doc_1"], confidence=0.9,
        )
        store.entities["a"] = ent_a
        store.entities["b"] = ent_b

        llm = _MockLLM(ExtractionResult(
            entities=[
                ExtractedEntity(name="A", type="ORG", description="d", confidence=0.9),
                ExtractedEntity(name="B", type="ORG", description="d", confidence=0.9),
            ],
            relations=[
                ExtractedRelation(head="A", tail="B", type="related_to",
                                  description="rel", weight=0.8),
            ],
        ))
        builder = KGBuilder(llm, store, _MockEmbedder(), KBConfig())

        await builder.build_for_chunks([_make_chunk()])

        assert len(store.upsert_relations_calls) == 1
        relations = store.upsert_relations_calls[0]
        assert len(relations) == 1
        assert relations[0].head_id == ent_a.entity_id
        assert relations[0].tail_id == ent_b.entity_id

    async def test_extraction_failure_skips_chunk(self) -> None:
        """Failed extraction is non-fatal (§5.2.3)."""
        store = _MockStore()

        class FailingLLM:
            async def structured(self, req, *, schema):
                raise ExtractionSchemaError("schema error")

        builder = KGBuilder(FailingLLM(), store, _MockEmbedder(), KBConfig())
        await builder.build_for_chunks([_make_chunk()])
        # No upsert calls (chunk skipped)
        assert store.upsert_entities_calls == []
        assert store.upsert_relations_calls == []


class TestDeleteChunkFromGraph:
    async def test_shared_entity_retained(self) -> None:
        """Entity referenced by 2 docs survives delete of doc_1 (§5.4.2)."""
        store = _MockStore()
        from ragx.core.ids import new_id

        ent = Entity(
            entity_id=new_id("ent_"), kb_id="kb_1", name="Beijing",
            name_norm="beijing", type="LOCATION", description="d",
            source_doc_ids=["doc_1", "doc_2"], confidence=0.9,
        )
        store.entities["beijing"] = ent

        builder = KGBuilder(_MockLLM(ExtractionResult()), store, _MockEmbedder(), KBConfig())
        chunk = _make_chunk(doc_id="doc_1")

        await builder.delete_chunk_from_graph(chunk)

        # Entity is still present with only doc_2
        assert "beijing" in store.entities
        assert store.entities["beijing"].source_doc_ids == ["doc_2"]
        assert store.delete_entity_calls == []

    async def test_orphan_entity_deleted(self) -> None:
        """Entity referenced only by doc_1 is deleted."""
        store = _MockStore()
        from ragx.core.ids import new_id

        ent = Entity(
            entity_id=new_id("ent_"), kb_id="kb_1", name="Temp",
            name_norm="temp", type="OTHER", description="d",
            source_doc_ids=["doc_1"], confidence=0.9,
        )
        store.entities["temp"] = ent

        builder = KGBuilder(_MockLLM(ExtractionResult()), store, _MockEmbedder(), KBConfig())
        chunk = _make_chunk(doc_id="doc_1")

        await builder.delete_chunk_from_graph(chunk)

        assert "temp" not in store.entities
        assert store.delete_entity_calls == [ent.entity_id]

    async def test_relation_cascade_delete(self) -> None:
        """Relations pointing to deleted entities are removed."""
        store = _MockStore()
        from ragx.core.ids import new_id

        ent = Entity(
            entity_id=new_id("ent_"), kb_id="kb_1", name="X",
            name_norm="x", type="ORG", description="d",
            source_doc_ids=["doc_1"], confidence=0.9,
        )
        store.entities["x"] = ent

        rel = Relation(
            relation_id=new_id("rel_"), kb_id="kb_1",
            head_id=ent.entity_id, tail_id=new_id("ent_"),
            type="owns", description="d", weight=1.0,
            source_doc_ids=["doc_1"],
        )
        store.relations[(rel.head_id, rel.tail_id, rel.type)] = rel

        builder = KGBuilder(_MockLLM(ExtractionResult()), store, _MockEmbedder(), KBConfig())
        chunk = _make_chunk(doc_id="doc_1")

        await builder.delete_chunk_from_graph(chunk)

        # Entity deleted → relation cascade-deleted
        assert "x" not in store.entities
        assert (rel.head_id, rel.tail_id, rel.type) not in store.relations

    async def test_delete_by_doc_fallback(self) -> None:
        """Stores without internal entity dict use SPI delete_by_doc."""
        store = _MockStore()
        store.entities = None  # simulate a store without internal access

        class SPIStore:
            def __init__(self):
                self.delete_by_doc_calls = []

            async def delete_by_doc(self, doc_id: str) -> None:
                self.delete_by_doc_calls.append(doc_id)

        spi = SPIStore()
        builder = KGBuilder(_MockLLM(ExtractionResult()), spi, _MockEmbedder(), KBConfig())
        chunk = _make_chunk(doc_id="doc_1")

        await builder.delete_chunk_from_graph(chunk)

        assert spi.delete_by_doc_calls == ["doc_1"]
