"""NetworkXGraphStore (11-plugins-builtin.md §11.1.3, lite profile).

* in-memory ``networkx.MultiDiGraph``; entities are keyed by ``name_norm`` so
  repeated upserts merge instead of duplicating (05-kg.md §5.3)
* ``delete_by_doc`` decrements ``source_doc_ids`` reference counts; nodes keep
  living while another document still references them, and private nodes are
  removed together with their relations (01-spi.md §1.2)
* optional pickle persistence (``persist_path``) - note pickle is not
  cross-version safe, hence the documented limitation
* ``topics`` uses ``greedy_modularity_communities``; the design's Louvain
  implementation was removed from networkx 3.x, so the docstring records the
  substitution
"""

from __future__ import annotations

import asyncio
import threading
from collections import Counter, deque
from pathlib import Path
from typing import Any

import networkx as nx

from ragx.core.exceptions import PluginContractError
from ragx.core.hashing import cosine_similarity
from ragx.core.ids import new_id
from ragx.core.models import Entity, Relation, SubGraph, TopicSummary
from ragx.spi.interfaces import GraphStoreCapabilities

_NAME = "nx"
_COMMUNITY_MIN_MEMBERS = 3


def _vec(seed: str, dim: int = 8) -> list[float]:
    raw = [(hash(seed + str(i)) % 1000) / 1000.0 for i in range(dim)]
    norm = sum(x * x for x in raw) ** 0.5 or 1.0
    return [x / norm for x in raw]


class NetworkXGraphStore:
    """SPI ``GraphStore`` over an in-memory NetworkX multigraph."""

    name: str = _NAME
    capabilities: GraphStoreCapabilities = GraphStoreCapabilities(supports_community=True)

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.kb_id: str = str(cfg.get("kb_id", "default"))
        self.persist_path: str | None = cfg.get("persist_path")
        self.community_algo: str = str(cfg.get("community_algo", "louvain"))
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self.entities: dict[str, Entity] = {}      # key: name_norm
        self.relations: dict[tuple[str, str, str], Relation] = {}
        self._lock = threading.RLock()
        if self.persist_path:
            self._load()

    # -- persistence ---------------------------------------------------------
    def _dump_locked(self) -> None:
        if not self.persist_path:
            return
        path = Path(self.persist_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "graph": self.graph,
            "entities": self.entities,
            "relations": self.relations,
            "kb_id": self.kb_id,
        }
        import pickle

        path.write_bytes(pickle.dumps(payload))

    def _load(self) -> None:
        path = Path(self.persist_path or "")
        if not path.exists():
            return
        import pickle

        try:
            payload = pickle.loads(path.read_bytes())
        except Exception:
            return
        self.graph = payload.get("graph") or nx.MultiDiGraph()
        self.entities = payload.get("entities") or {}
        self.relations = payload.get("relations") or {}

    # -- helpers -------------------------------------------------------------
    def _check_kb(self, kb_id: str) -> None:
        if kb_id and kb_id != self.kb_id:
            raise PluginContractError(
                "entity/relation belongs to a different knowledge base",
                details={"expected": self.kb_id, "got": kb_id},
            )

    # -- SPI -----------------------------------------------------------------
    async def upsert_entities(self, entities: list[Entity]) -> None:
        def _run() -> None:
            with self._lock:
                for entity in entities:
                    self._check_kb(entity.kb_id)
                    key = entity.name_norm
                    existing = self.entities.get(key)
                    if existing is None:
                        self.entities[key] = entity.model_copy(deep=True)
                        self.graph.add_node(entity.entity_id, **entity.model_dump())
                        continue
                    merged = existing.model_copy(deep=True)
                    merged.source_doc_ids = sorted(set(merged.source_doc_ids) | set(entity.source_doc_ids))
                    merged.confidence = max(merged.confidence, entity.confidence)
                    if len(entity.description) > len(merged.description):
                        merged.description = entity.description
                    if not merged.vector and entity.vector:
                        merged.vector = entity.vector
                    self.entities[key] = merged
                    self.graph.nodes[entity.entity_id].update(merged.model_dump())
                self._dump_locked()
        await asyncio.to_thread(_run)

    async def upsert_relations(self, relations: list[Relation]) -> None:
        def _run() -> None:
            with self._lock:
                for relation in relations:
                    self._check_kb(relation.kb_id)
                    key = (relation.head_id, relation.tail_id, relation.type)
                    existing = self.relations.get(key)
                    if existing is None:
                        self.relations[key] = relation.model_copy(deep=True)
                        self.graph.add_edge(
                            relation.head_id, relation.tail_id,
                            key=relation.relation_id, **relation.model_dump(),
                        )
                        continue
                    merged = existing.model_copy(deep=True)
                    merged.source_doc_ids = sorted(set(merged.source_doc_ids) | set(relation.source_doc_ids))
                    merged.weight = max(merged.weight, relation.weight)
                    if len(relation.description) > len(merged.description):
                        merged.description = relation.description
                    self.relations[key] = merged
                self._dump_locked()
        await asyncio.to_thread(_run)

    async def delete_by_doc(self, doc_id: str) -> None:
        """Decrement reference counts; delete nodes whose count reaches zero."""
        def _run() -> None:
            with self._lock:
                removed_nodes: set[str] = set()
                for entity in list(self.entities.values()):
                    if doc_id in entity.source_doc_ids:
                        entity.source_doc_ids = [d for d in entity.source_doc_ids if d != doc_id]
                        if not entity.source_doc_ids:
                            removed_nodes.add(entity.entity_id)
                            self.entities.pop(entity.name_norm, None)
                for node in removed_nodes:
                    if self.graph.has_node(node):
                        self.graph.remove_node(node)   # cascades incident edges
                for relation in list(self.relations.values()):
                    if relation.head_id in removed_nodes or relation.tail_id in removed_nodes:
                        self.relations.pop((relation.head_id, relation.tail_id, relation.type), None)
                        continue
                    if doc_id in relation.source_doc_ids:
                        relation.source_doc_ids = [d for d in relation.source_doc_ids if d != doc_id]
                        if not relation.source_doc_ids:
                            self.relations.pop((relation.head_id, relation.tail_id, relation.type), None)
                self._dump_locked()
        await asyncio.to_thread(_run)

    async def match_entities(self, query_vector: list[float], *, top_k: int) -> list[Entity]:
        def _run() -> list[Entity]:
            with self._lock:
                scored: list[tuple[float, Entity]] = []
                for entity in self.entities.values():
                    if not entity.vector or len(entity.vector) != len(query_vector):
                        continue
                    scored.append((cosine_similarity(query_vector, entity.vector), entity))
                scored.sort(key=lambda pair: pair[0], reverse=True)
                return [entity for _, entity in scored[: max(top_k, 0)]]
        return await asyncio.to_thread(_run)

    async def neighbors(
        self, entity_ids: list[str], *, hops: int, limit: int
    ) -> SubGraph:
        def _run() -> SubGraph:
            with self._lock:
                undirected = self.graph.to_undirected(as_view=True)
                visited: dict[str, int] = {}
                queue: deque[tuple[str, int]] = deque((eid, 0) for eid in entity_ids)
                while queue:
                    node, depth = queue.popleft()
                    if depth > 0 and node in visited:
                        continue
                    if depth > 0:
                        visited[node] = depth
                    if node not in self.graph:
                        continue
                    if depth < hops:
                        for neighbour in undirected.neighbors(node) if undirected.has_node(node) else []:
                            if neighbour not in visited:
                                queue.append((neighbour, depth + 1))
                    if len(visited) >= limit:
                        break
                ids = set(entity_ids) | set(visited)
                id_to_entity = {e.entity_id: e for e in self.entities.values()}
                entities = [id_to_entity[i] for i in ids if i in id_to_entity]
                relations = [
                    rel for key, rel in self.relations.items()
                    if key[0] in ids and key[1] in ids
                ]
                return SubGraph(entities=entities, relations=relations)
        return await asyncio.to_thread(_run)

    async def topics(self, query_vector: list[float], *, top_k: int) -> list[TopicSummary]:
        if not self.capabilities.supports_community:
            return []

        def _run() -> list[TopicSummary]:
            with self._lock:
                if not self.entities:
                    return []
                members_by_group: list[list[str]]
                undirected = self.graph.to_undirected(as_view=True)
                if undirected.number_of_edges() == 0:
                    members_by_group = [list(self.graph.nodes)]
                else:
                    try:
                        members_by_group = [
                            [str(n) for n in community]
                            for community in nx.algorithms.community.greedy_modularity_communities(
                                undirected
                            )
                        ]
                    except nx.NetworkXAlgorithmError:
                        members_by_group = [list(self.graph.nodes)]
                members_by_group = [g for g in members_by_group if len(g) >= _COMMUNITY_MIN_MEMBERS]
                if not members_by_group:
                    members_by_group = [list(self.graph.nodes)] if len(self.graph.nodes) >= 1 else []

                id_to_entity = {e.entity_id: e for e in self.entities.values()}
                topics: list[tuple[float, TopicSummary]] = []
                for members in members_by_group:
                    ents = [id_to_entity[m] for m in members if m in id_to_entity]
                    if not ents:
                        continue
                    counts = Counter(e.type for e in ents)
                    top = sorted(ents, key=lambda e: e.confidence, reverse=True)[:3]
                    title = " / ".join(e.name for e in top) or counts.most_common(1)[0][0]
                    summary = "；".join(
                        f"{e.name}：{e.description}" for e in top if e.description
                    )[:300]
                    centroid: list[float] = []
                    with_vector = [e for e in ents if e.vector]
                    dim = len(with_vector[0].vector) if with_vector else 0
                    if dim:
                        centroid = [
                            sum(e.vector[i] for e in with_vector) / len(with_vector)
                            for i in range(dim)
                        ]
                    topic = TopicSummary(
                        topic_id=new_id("tpc_"),
                        kb_id=self.kb_id,
                        title=title,
                        summary=summary or title,
                        member_entity_ids=[e.entity_id for e in ents],
                        vector=centroid,
                    )
                    score = (
                        cosine_similarity(query_vector, centroid)
                        if query_vector and centroid and len(centroid) == len(query_vector)
                        else 0.0
                    )
                    topics.append((score, topic))
                topics.sort(key=lambda pair: (pair[0], len(pair[1].member_entity_ids)), reverse=True)
                return [t for _, t in topics[: max(top_k, 0)]]
        return await asyncio.to_thread(_run)

    # -- introspection -------------------------------------------------------
    async def entity_count(self) -> int:
        def _run() -> int:
            with self._lock:
                return len(self.entities)
        return await asyncio.to_thread(_run)

    async def relation_count(self) -> int:
        def _run() -> int:
            with self._lock:
                return len(self.relations)
        return await asyncio.to_thread(_run)

    async def startup(self) -> None:
        return None

    async def shutdown(self) -> None:
        with self._lock:
            self._dump_locked()
