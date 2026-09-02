"""Topic summary tests (05-kg.md §5.5).

Covers ``build_topics`` (by-type grouping for small graphs, community
detection for large graphs), the summary cache, and rebuild triggers.
"""

from __future__ import annotations

from ragx.core.models import Entity, Relation
from ragx.kg.topics import (
    build_topics,
    invalidate_topic_cache,
    maybe_rebuild_topics,
)


def _entity(entity_id: str, name: str, etype: str, *, vector=None) -> Entity:
    return Entity(
        entity_id=entity_id, kb_id="kb_1", name=name, name_norm=name.lower(),
        type=etype, description=f"{name} 的描述", source_doc_ids=["doc_1"],
        confidence=0.9, vector=vector or [0.1, 0.2, 0.3],
    )


class _GraphStore:
    def __init__(self, entities: list[Entity], relations: list[Relation] | None = None):
        self.entities = {e.name_norm: e for e in entities}
        self.relations = {r.relation_id: r for r in (relations or [])}
        self._last_topic_build_count: int = 0


class TestBuildTopics:
    async def test_groups_by_type_for_small_graph(self) -> None:
        store = _GraphStore([
            _entity("ent_a", "OpenAI", "ORG"),
            _entity("ent_b", "Anthropic", "ORG"),
            _entity("ent_c", "Beijing", "LOCATION"),
        ])
        topics = await build_topics(store, "kb_1", llm=None)

        assert topics
        by_type = {t.title for t in topics}
        assert "ORG 类实体" in by_type
        assert "LOCATION 类实体" in by_type
        for t in topics:
            assert t.kb_id == "kb_1"
            assert t.summary
            assert t.member_entity_ids

    async def test_empty_graph_returns_empty(self) -> None:
        store = _GraphStore([])
        assert await build_topics(store, "kb_1", llm=None) == []

    async def test_summary_cache_reused(self) -> None:
        """Identical member sets reuse the cached summary (LLM called once)."""
        invalidate_topic_cache()  # isolate from other tests' module-level cache
        calls = []

        class CountingLLM:
            async def chat(self, req):
                calls.append(1)
                return type(
                    "Resp", (), {"text": "一份主题摘要。"}
                )()

        store = _GraphStore([
            _entity("ent_a", "OpenAI", "ORG"),
            _entity("ent_b", "Anthropic", "ORG"),
        ])
        t1 = await build_topics(store, "kb_1", llm=CountingLLM())
        t2 = await build_topics(store, "kb_1", llm=CountingLLM())

        assert t1 and t2
        assert t1[0].summary == t2[0].summary
        # summary cache hit → no second LLM call
        assert len(calls) == 1


class TestInvalidateTopicCache:
    async def test_invalidate_forces_recompute(self) -> None:
        invalidate_topic_cache()
        calls = []

        class CountingLLM:
            async def chat(self, req):
                calls.append(1)
                return type("Resp", (), {"text": "新主题摘要。"})()

        store = _GraphStore([_entity("ent_a", "OpenAI", "ORG")])
        await build_topics(store, "kb_1", llm=CountingLLM())
        assert len(calls) == 1

        invalidate_topic_cache()
        await build_topics(store, "kb_1", llm=CountingLLM())
        assert len(calls) == 2


class TestMaybeRebuildTopics:
    async def test_manual_rebuild_always_runs(self) -> None:
        store = _GraphStore([_entity("ent_a", "OpenAI", "ORG")])
        # no error even with llm=None
        await maybe_rebuild_topics(store, "kb_1", llm=None, manual=True)

    async def test_auto_skips_below_delta_threshold(self) -> None:
        store = _GraphStore([_entity("ent_a", "OpenAI", "ORG")])
        store._last_topic_build_count = 1
        # 0% delta → no rebuild
        await maybe_rebuild_topics(store, "kb_1", llm=None)
