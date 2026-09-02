"""Merge tests (05-kg.md §5.3).

Covers: normalize_name, two-level merge, source_doc_ids union,
confidence max, description longest-wins.
"""

from __future__ import annotations

from ragx.core.models import Entity
from ragx.kg.merge import (
    merge_entities,
    normalize_name,
)
from ragx.kg.schemas import ExtractedEntity


def _make_entity(
    name: str, name_norm: str, type: str = "ORG",
    description: str = "desc", confidence: float = 0.9,
    source_doc_ids: list[str] | None = None,
    vector: list[float] | None = None,
    entity_id: str | None = None,
) -> Entity:
    from ragx.core.ids import new_id

    return Entity(
        entity_id=entity_id or new_id("ent_"),
        kb_id="kb_1",
        name=name,
        name_norm=name_norm,
        type=type,
        description=description,
        source_doc_ids=source_doc_ids or ["doc_1"],
        confidence=confidence,
        vector=vector or [],
    )


class TestNormalizeName:
    def test_lowercase_and_strip(self) -> None:
        assert normalize_name("  OpenAI  ") == "openai"

    def test_full_width_to_half_width(self) -> None:
        # NFKC: full-width ＡＢＣ → half-width ABC
        assert normalize_name("ＡＢＣ") == "abc"

    def test_strip_punctuation(self) -> None:
        """Punctuation is removed (not replaced with space) per §5.3.1."""
        assert normalize_name("Open-AI/LLC") == "openaillc"

    def test_collapse_whitespace(self) -> None:
        assert normalize_name("Open  AI") == "open ai"


class _MockStore:
    """Mock GraphStore with instance-level entities dict."""

    def __init__(self, entities: dict | None = None) -> None:
        self.entities = entities or {}

    async def match_entities(self, vec, *, top_k):
        return []


class _MockEmbedder:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class TestTwoLevelMerge:
    """Uses a MockStore with an instance-level ``entities`` dict."""

    async def _merge(
        self, extracted: list[ExtractedEntity], existing: list[Entity]
    ) -> list[Entity]:
        store_dict: dict[str, Entity] = {}
        for e in existing:
            store_dict[e.name_norm] = e
        return await merge_entities(
            extracted, "kb_1", "doc_2", _MockStore(store_dict), _MockEmbedder()
        )

    async def test_level1_exact_name_norm_merge(self) -> None:
        """Same name_norm → merge: description longest, docs union, confidence max."""
        existing = [_make_entity(
            "OpenAI", "openai", description="Short desc",
            confidence=0.7, source_doc_ids=["doc_1"],
        )]
        extracted = [ExtractedEntity(
            name="OpenAI", type="ORG",
            description="A longer description of OpenAI", confidence=0.95,
        )]
        result = await self._merge(extracted, existing)
        assert len(result) == 1
        ent = result[0]
        assert ent.entity_id == existing[0].entity_id  # merged into existing
        assert ent.description == "A longer description of OpenAI"  # longer wins
        assert sorted(ent.source_doc_ids) == ["doc_1", "doc_2"]  # union
        assert ent.confidence == 0.95  # max

    async def test_new_entity_when_no_match(self) -> None:
        extracted = [ExtractedEntity(
            name="Google", type="ORG",
            description="Search company", confidence=0.9,
        )]
        result = await self._merge(extracted, [])
        assert len(result) == 1
        ent = result[0]
        assert ent.name == "Google"
        assert ent.source_doc_ids == ["doc_2"]
        assert ent.entity_id.startswith("ent_")

    async def test_multiple_entities_merged(self) -> None:
        extracted = [
            ExtractedEntity(name="A", type="ORG", description="a", confidence=0.9),
            ExtractedEntity(name="B", type="ORG", description="b", confidence=0.9),
        ]
        existing = [_make_entity("A", "a", source_doc_ids=["doc_1"])]
        result = await self._merge(extracted, existing)
        assert len(result) == 2  # A merged, B new
        assert result[0].entity_id == existing[0].entity_id
        assert result[1].name == "B"


class TestSourceDocIdsUnion:
    async def test_reference_count_accumulates(self) -> None:
        """Same entity from 3 docs → merged result has 3 source_doc_ids."""
        store = _MockStore()
        ent = _make_entity("OpenAI", "openai", source_doc_ids=["doc_1"])
        store.entities["openai"] = ent

        ext = [ExtractedEntity(name="OpenAI", type="ORG", description="d", confidence=0.9)]

        # doc_2: merge returns an entity with doc_1+doc_2
        r1 = await merge_entities(ext, "kb_1", "doc_2", store, _MockEmbedder())
        # Simulate upsert: update the store with the merged entity
        store.entities["openai"] = r1[0]

        # doc_3: merge returns an entity with doc_1+doc_2+doc_3
        r2 = await merge_entities(ext, "kb_1", "doc_3", store, _MockEmbedder())
        assert sorted(r2[0].source_doc_ids) == ["doc_1", "doc_2", "doc_3"]
