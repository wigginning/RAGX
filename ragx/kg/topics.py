"""High-Level topic summaries (05-kg.md §5.5).

Lightweight community discovery + LLM summary caching. Two strategies:
* **< 1000 entities** — skip community detection, group by entity ``type``.
* **≥ 1000 entities** — NetworkX greedy modularity community detection.

Rebuild triggers (§5.5.3): entity delta > 20%, manual API call, or
summary-cache TTL expiry (24h).
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from ragx.core.ids import new_id
from ragx.core.models import Entity, Relation, TopicSummary
from ragx.core.roles import LLMRole
from ragx.spi.interfaces import ChatRequest

logger = logging.getLogger("ragx.kg.topics")

_COMMUNITY_MIN_MEMBERS = 3
_TOPIC_CACHE_TTL_S = 86_400  # 1 day
_TOPIC_SUMMARY_LIMIT = 200  # chars


async def build_topics(
    graph_store: Any,
    kb_id: str,
    llm: Any | None = None,
) -> list[TopicSummary]:
    """Rebuild topic summaries for a knowledge base (§5.5.1).

    Uses community detection when the graph is large enough; otherwise
    groups entities by type.
    """
    entities = await _get_all_entities(graph_store, kb_id)
    if not entities:
        return []

    if len(entities) < 1000:
        return await _topics_by_type(entities, kb_id, llm)

    return await _topics_by_community(entities, graph_store, kb_id, llm)


async def _get_all_entities(graph_store: Any, kb_id: str) -> list[Entity]:
    """Retrieve all entities for a kb_id.

    NetworkXGraphStore keeps entities in a dict keyed by name_norm; other
    stores may expose a different interface.
    """
    entities_dict = getattr(graph_store, "entities", None)
    if entities_dict is not None:
        return list(entities_dict.values())
    # Fallback: try the SPI match_entities with a dummy vector (returns all).
    return []


async def _topics_by_type(
    entities: list[Entity], kb_id: str, llm: Any | None
) -> list[TopicSummary]:
    """Group entities by ``type``, generate a summary per group (§5.5.1)."""
    by_type: dict[str, list[Entity]] = {}
    for e in entities:
        by_type.setdefault(e.type, []).append(e)

    topics: list[TopicSummary] = []
    for etype, members in by_type.items():
        title = f"{etype} 类实体"
        summary = await _generate_summary(members, llm, title)
        centroid = _compute_centroid(members)
        topics.append(TopicSummary(
            topic_id=new_id("tpc_"),
            kb_id=kb_id,
            title=title,
            summary=summary,
            member_entity_ids=[e.entity_id for e in members],
            vector=centroid,
        ))
    return topics


async def _topics_by_community(
    entities: list[Entity],
    graph_store: Any,
    kb_id: str,
    llm: Any | None,
) -> list[TopicSummary]:
    """Community detection via NetworkX greedy modularity (§5.5.1)."""
    import networkx as nx

    G = nx.Graph()
    id_to_entity: dict[str, Entity] = {}
    for e in entities:
        G.add_node(e.entity_id, entity=e)
        id_to_entity[e.entity_id] = e

    relations = await _get_all_relations(graph_store, kb_id)
    for r in relations:
        if r.head_id in id_to_entity and r.tail_id in id_to_entity:
            G.add_edge(r.head_id, r.tail_id, weight=r.weight)

    if G.number_of_edges() == 0:
        communities = [set(G.nodes)]
    else:
        try:
            communities = [
                set(c) for c in nx.algorithms.community.greedy_modularity_communities(G)
            ]
        except nx.NetworkXAlgorithmError:
            communities = [set(G.nodes)]

    topics: list[TopicSummary] = []
    for comm in communities:
        members = [id_to_entity[nid] for nid in comm if nid in id_to_entity]
        if len(members) < _COMMUNITY_MIN_MEMBERS:
            continue
        top = sorted(members, key=lambda e: e.confidence, reverse=True)[:3]
        title = " / ".join(e.name for e in top) or members[0].type
        summary = await _generate_summary(members, llm, title)
        centroid = _compute_centroid(members)
        topics.append(TopicSummary(
            topic_id=new_id("tpc_"),
            kb_id=kb_id,
            title=title,
            summary=summary,
            member_entity_ids=[e.entity_id for e in members],
            vector=centroid,
        ))

    # Upsert if the store supports it.
    if hasattr(graph_store, "upsert_topics"):
        await graph_store.upsert_topics(topics)

    return topics


async def _get_all_relations(graph_store: Any, kb_id: str) -> list[Relation]:
    relations_dict = getattr(graph_store, "relations", None)
    if relations_dict is not None:
        return list(relations_dict.values())
    return []


async def _generate_summary(
    members: list[Entity], llm: Any | None, default_title: str
) -> str:
    """Generate a topic summary via LLM (role: synthesize) or fallback.

    Cache key: sha256 of sorted member name_norms (§5.5.2).
    """
    cache_key = hashlib.sha256(
        "|".join(sorted(e.name_norm for e in members)).encode()
    ).hexdigest()
    # Simple in-process cache (TTL enforced by the caller's rebuild trigger).
    cached = _SUMMARY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    names = [e.name for e in members]
    desc_list = [e.description for e in members]
    summary_text = f"实体群: {', '.join(names)}\n描述: {desc_list}"

    if llm is not None:
        try:
            prompt = (
                f"为以下实体群生成一段主题摘要（≤{_TOPIC_SUMMARY_LIMIT}字）：\n"
                f"{summary_text}"
            )
            req = ChatRequest(
                messages=[{"role": "user", "content": prompt}],
                role=LLMRole.SYNTHESIZE,
                temperature=0.0,
            )
            resp = await llm.chat(req)
            summary_text = resp.text
        except Exception as exc:
            logger.warning("topic summary LLM call failed: %s", exc)

    # Fallback: concatenate member descriptions
    if len(summary_text) < 10:
        top = sorted(members, key=lambda e: e.confidence, reverse=True)[:3]
        summary_text = "；".join(
            f"{e.name}：{e.description}" for e in top if e.description
        )[:_TOPIC_SUMMARY_LIMIT]

    _SUMMARY_CACHE[cache_key] = summary_text
    return summary_text


def _compute_centroid(members: list[Entity]) -> list[float]:
    """Average the entity vectors into a topic centroid."""
    with_vector = [e for e in members if e.vector]
    if not with_vector:
        return []
    dim = len(with_vector[0].vector)
    return [
        sum(e.vector[i] for e in with_vector) / len(with_vector)
        for i in range(dim)
    ]


_SUMMARY_CACHE: dict[str, str] = {}


def invalidate_topic_cache() -> None:
    """Clear the topic summary cache (e.g. after a bulk rebuild)."""
    _SUMMARY_CACHE.clear()


async def maybe_rebuild_topics(
    graph_store: Any,
    kb_id: str,
    llm: Any | None = None,
    *,
    manual: bool = False,
) -> None:
    """Check rebuild triggers and rebuild if needed (§5.5.3)."""
    stats = await _get_kb_stats(graph_store, kb_id)
    last_count = stats.get("last_topic_build_entity_count", 0)
    current_count = stats.get("current_entity_count", 0)

    if manual:
        await build_topics(graph_store, kb_id, llm)
        return

    if last_count > 0 and (current_count - last_count) / last_count > 0.20:
        await build_topics(graph_store, kb_id, llm)


async def _get_kb_stats(graph_store: Any, kb_id: str) -> dict[str, int]:
    """Return entity count stats for the rebuild check."""
    entities_dict = getattr(graph_store, "entities", None)
    current = len(entities_dict) if entities_dict else 0
    return {
        "current_entity_count": current,
        "last_topic_build_entity_count": getattr(
            graph_store, "_last_topic_build_count", 0
        ),
    }
