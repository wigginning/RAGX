"""Neo4jGraphStore (11-plugins-builtin.md §11.2.3, full profile).

* entities stored as ``:Entity`` nodes; ``upsert_entities`` uses ``MERGE`` on
  ``name_norm`` (the dedup key, 05-kg.md §5.3) so repeated upserts merge instead
  of duplicating — ``source_doc_ids`` becomes the union, ``confidence`` takes
  the max, ``description`` takes the longer
* ``delete_by_doc`` decrements ``source_doc_ids``; nodes whose reference count
  reaches zero are deleted (``DETACH DELETE`` cascades incident relations);
  shared nodes survive (01-spi.md §1.2, 05-kg.md §5.4)
* ``match_entities`` does vector near-neighbor — a Neo4j vector index is created
  on startup; if unavailable (Community edition / old version) a brute-force
  scan with Python cosine is used as fallback so the contract always holds
* ``neighbors`` does BFS via variable-length ``[:RELATION*0..hops]`` traversal
* ``topics`` runs community detection: data is loaded from Neo4j into an
  in-memory NetworkX graph (core dependency) and ``greedy_modularity_communities``
  is used (the design names Louvain but networkx 3.x removed it — the NetworkX
  store records the same substitution)

Third-party exceptions (``neo4j.*``) are translated at the plugin boundary
(02-core.md §2.3); raw exceptions never cross the SPI.
"""

from __future__ import annotations

import json
import os
from typing import Any

import networkx as nx  # type: ignore[import-untyped]

from ragx.core.exceptions import (
    GraphStoreUnavailableError,
    PluginContractError,
    PluginTimeoutError,
)
from ragx.core.hashing import cosine_similarity
from ragx.core.ids import new_id
from ragx.core.models import Entity, Relation, SubGraph, TopicSummary
from ragx.spi.interfaces import GraphStoreCapabilities

_NAME = "neo4j"
_COMMUNITY_MIN_MEMBERS = 3


async def _fetch_all(result: Any) -> list[Any]:
    """Collect every record from an ``AsyncResult``.

    Cross-version helper: neo4j ≥6 changed ``AsyncResult.fetch()`` to require a
    positional ``n`` (5.x defaulted to fetch-all), so we iterate instead — the
    async-iterator protocol is stable across 5.x and 6.x.
    """
    return [record async for record in result]


def _parse_metadata(raw: Any) -> dict[str, Any]:
    """Deserialise an entity's ``metadata`` property.

    Neo4j cannot store a map as a property value (only primitives + arrays), so
    ``metadata`` is persisted as a JSON string. Tolerate legacy/empty values.
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw:
        try:
            loaded = json.loads(raw)
            return dict(loaded) if isinstance(loaded, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


class Neo4jGraphStore:
    """SPI ``GraphStore`` backed by Neo4j 5.x."""

    name: str = _NAME
    capabilities: GraphStoreCapabilities = GraphStoreCapabilities(
        supports_community=True
    )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.uri: str = str(cfg.get("uri", "bolt://localhost:7687"))
        self.auth_env: str = str(cfg.get("auth_env", "NEO4J_AUTH"))
        self.database: str = str(cfg.get("database", "neo4j"))
        self.kb_id: str = str(cfg.get("kb_id", "default"))
        self.dimension: int = int(cfg.get("dim", 8))
        self._driver: Any = None  # AsyncGraphDatabase driver (lazy)
        self._vector_index = f"ragx_entity_vec_{self.kb_id}"
        self._has_vector_index: bool = False

    # -- driver / schema ---------------------------------------------------
    def _ensure_driver(self) -> Any:
        if self._driver is not None:
            return self._driver
        try:
            from neo4j import AsyncGraphDatabase  # type: ignore[import-not-found]
        except ImportError as exc:
            raise PluginContractError(
                "neo4j package not installed (pip install ragx[neo4j])",
                details={"error": str(exc)},
            ) from exc
        auth_str = os.environ.get(self.auth_env, "")
        auth: tuple[str, str] | None = None
        if auth_str:
            # Neo4j's official image uses ``NEO4J_AUTH=user/password`` (slash);
            # tolerate the colon form too. ``split(..., 1)`` keeps passwords
            # that themselves contain '/' or ':' intact.
            sep = "/" if "/" in auth_str else (":" if ":" in auth_str else None)
            if sep is not None:
                user, password = auth_str.split(sep, 1)
                auth = (user, password)
            else:
                auth = ("neo4j", auth_str)
        self._driver = AsyncGraphDatabase.driver(self.uri, auth=auth)
        return self._driver

    async def _ensure_vector_index(self) -> None:
        """Create a vector index for entity near-neighbor (best-effort)."""
        driver = self._ensure_driver()
        try:
            cypher = (
                "CREATE VECTOR INDEX $index_name IF NOT EXISTS "
                "FOR (e:Entity) ON (e.vector) "
                "OPTIONS {indexConfig: {`vector.dimensions`: $dim, "
                "`vector.similarity_function`: 'cosine'}}"
            )
            async with driver.session(database=self.database) as session:
                await (await session.run(cypher, index_name=self._vector_index,
                                         dim=self.dimension)).consume()
            self._has_vector_index = True
        except Exception:
            # Vector index not available (Community edition / wrong version).
            # Fall back to brute-force cosine; the contract still holds.
            self._has_vector_index = False

    async def _ensure_constraints(self) -> None:
        driver = self._ensure_driver()
        try:
            async with driver.session(database=self.database) as session:
                await (await session.run(
                    "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS "
                    "FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE"
                )).consume()
                await (await session.run(
                    "CREATE INDEX entity_name_norm IF NOT EXISTS "
                    "FOR (e:Entity) ON (e.name_norm, e.kb_id)"
                )).consume()
        except Exception as exc:
            raise self._translate(exc, "ensure constraints") from exc

    # -- helpers ------------------------------------------------------------
    def _check_kb(self, kb_id: str) -> None:
        if kb_id and kb_id != self.kb_id:
            raise PluginContractError(
                "entity/relation belongs to a different knowledge base",
                details={"expected": self.kb_id, "got": kb_id},
            )

    @staticmethod
    def _node_to_entity(node: Any) -> Entity:
        """Neo4j Node → Entity (deserialise vector + lists)."""
        props = dict(node)
        vec = props.get("vector") or []
        if not isinstance(vec, list):
            vec = list(vec) if vec else []
        return Entity(
            entity_id=props.get("entity_id", ""),
            kb_id=props.get("kb_id", ""),
            name=props.get("name", ""),
            name_norm=props.get("name_norm", ""),
            type=props.get("type", ""),
            description=props.get("description", ""),
            source_doc_ids=list(props.get("source_doc_ids") or []),
            confidence=float(props.get("confidence", 0.0)),
            vector=[float(x) for x in vec],
            metadata=_parse_metadata(props.get("metadata")),
        )

    @staticmethod
    def _rel_to_relation(rel: Any, head_id: str, tail_id: str) -> Relation:
        props = dict(rel)
        return Relation(
            relation_id=props.get("relation_id", ""),
            kb_id=props.get("kb_id", ""),
            head_id=head_id,
            tail_id=tail_id,
            type=props.get("type", ""),
            description=props.get("description", ""),
            weight=float(props.get("weight", 1.0)),
            source_doc_ids=list(props.get("source_doc_ids") or []),
        )

    # -- SPI ----------------------------------------------------------------
    async def upsert_entities(self, entities: list[Entity]) -> None:
        if not entities:
            return
        driver = self._ensure_driver()
        for entity in entities:
            self._check_kb(entity.kb_id)
        try:
            async with driver.session(database=self.database) as session:
                for entity in entities:
                    cypher = (
                        "MERGE (e:Entity {name_norm: $name_norm, kb_id: $kb_id}) "
                        "ON CREATE SET e.entity_id = $entity_id, e.name = $name, "
                        "  e.type = $type, e.description = $description, "
                        "  e.source_doc_ids = $source_doc_ids, "
                        "  e.confidence = $confidence, e.vector = $vector, "
                        "  e.metadata = $metadata "
                        "ON MATCH SET "
                        "  e.source_doc_ids = REDUCE(acc = [], x IN "
                        "    (e.source_doc_ids + $source_doc_ids) | "
                        "    CASE WHEN x IN acc THEN acc ELSE acc + x END), "
                        "  e.confidence = CASE "
                        "    WHEN $confidence > e.confidence THEN $confidence "
                        "    ELSE e.confidence END, "
                        "  e.description = CASE "
                        "    WHEN size($description) > size(e.description) "
                        "    THEN $description ELSE e.description END, "
                        "  e.name = CASE "
                        "    WHEN size($name) >= size(e.name) THEN $name "
                        "    ELSE e.name END, "
                        "  e.vector = CASE "
                        "    WHEN (e.vector IS NULL OR size(e.vector) = 0) "
                        "      AND size($vector) > 0 THEN $vector "
                        "    ELSE e.vector END"
                    )
                    await (await session.run(
                        cypher,
                        name_norm=entity.name_norm,
                        kb_id=entity.kb_id,
                        entity_id=entity.entity_id,
                        name=entity.name,
                        type=entity.type,
                        description=entity.description,
                        source_doc_ids=list(entity.source_doc_ids),
                        confidence=float(entity.confidence),
                        vector=[float(x) for x in entity.vector],
                        metadata=json.dumps(entity.metadata, ensure_ascii=False),
                    )).consume()
        except Exception as exc:
            raise self._translate(exc, "upsert_entities") from exc

    async def upsert_relations(self, relations: list[Relation]) -> None:
        if not relations:
            return
        driver = self._ensure_driver()
        for relation in relations:
            self._check_kb(relation.kb_id)
        try:
            async with driver.session(database=self.database) as session:
                for relation in relations:
                    cypher = (
                        "MATCH (h:Entity {entity_id: $head_id}), "
                        "      (t:Entity {entity_id: $tail_id}) "
                        "MERGE (h)-[r:RELATION {relation_id: $relation_id}]->(t) "
                        "ON CREATE SET r.kb_id = $kb_id, r.type = $type, "
                        "  r.description = $description, r.weight = $weight, "
                        "  r.source_doc_ids = $source_doc_ids "
                        "ON MATCH SET "
                        "  r.source_doc_ids = REDUCE(acc = [], x IN "
                        "    (r.source_doc_ids + $source_doc_ids) | "
                        "    CASE WHEN x IN acc THEN acc ELSE acc + x END), "
                        "  r.weight = CASE "
                        "    WHEN $weight > r.weight THEN $weight "
                        "    ELSE r.weight END, "
                        "  r.description = CASE "
                        "    WHEN size($description) > size(r.description) "
                        "    THEN $description ELSE r.description END"
                    )
                    await (await session.run(
                        cypher,
                        head_id=relation.head_id,
                        tail_id=relation.tail_id,
                        relation_id=relation.relation_id,
                        kb_id=relation.kb_id,
                        type=relation.type,
                        description=relation.description,
                        weight=float(relation.weight),
                        source_doc_ids=list(relation.source_doc_ids),
                    )).consume()
        except Exception as exc:
            raise self._translate(exc, "upsert_relations") from exc

    async def delete_by_doc(self, doc_id: str) -> None:
        driver = self._ensure_driver()
        try:
            async with driver.session(database=self.database) as session:
                # 1. Remove doc_id from entity source_doc_ids
                await (await session.run(
                    "MATCH (e:Entity) WHERE $doc_id IN e.source_doc_ids "
                    "SET e.source_doc_ids = "
                    "  [x IN e.source_doc_ids WHERE x <> $doc_id]",
                    doc_id=doc_id,
                )).consume()
                # 2. Delete entities with empty source_doc_ids (cascade relations)
                await (await session.run(
                    "MATCH (e:Entity) WHERE size(e.source_doc_ids) = 0 "
                    "DETACH DELETE e"
                )).consume()
                # 3. Remove doc_id from relation source_doc_ids
                await (await session.run(
                    "MATCH ()-[r:RELATION]->() "
                    "WHERE $doc_id IN r.source_doc_ids "
                    "SET r.source_doc_ids = "
                    "  [x IN r.source_doc_ids WHERE x <> $doc_id]",
                    doc_id=doc_id,
                )).consume()
                # 4. Delete relations with empty source_doc_ids
                await (await session.run(
                    "MATCH ()-[r:RELATION]->() "
                    "WHERE size(r.source_doc_ids) = 0 DELETE r"
                )).consume()
        except Exception as exc:
            raise self._translate(exc, "delete_by_doc") from exc

    async def match_entities(
        self, query_vector: list[float], *, top_k: int
    ) -> list[Entity]:
        driver = self._ensure_driver()

        # Try vector index first (if available)
        if self._has_vector_index:
            try:
                cypher = (
                    "CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector) "
                    "YIELD node, score RETURN node, score ORDER BY score DESC"
                )
                async with driver.session(database=self.database) as session:
                    result = await session.run(
                        cypher,
                        index_name=self._vector_index,
                        top_k=max(top_k, 1),
                        query_vector=[float(x) for x in query_vector],
                    )
                    records = await _fetch_all(result)
                    if records:
                        return [self._node_to_entity(r["node"]) for r in records]
            except Exception:
                pass  # fall through to brute-force

        # Brute-force: fetch all entities with vectors, compute cosine in Python
        return await self._brute_match(query_vector, top_k)

    async def _brute_match(
        self, query_vector: list[float], top_k: int
    ) -> list[Entity]:
        driver = self._ensure_driver()
        try:
            async with driver.session(database=self.database) as session:
                result = await session.run(
                    "MATCH (e:Entity {kb_id: $kb_id}) "
                    "WHERE e.vector IS NOT NULL AND size(e.vector) = $dim "
                    "RETURN e",
                    kb_id=self.kb_id,
                    dim=len(query_vector),
                )
                records = await _fetch_all(result)
        except Exception as exc:
            raise self._translate(exc, "match_entities") from exc

        scored: list[tuple[float, Entity]] = []
        for record in records:
            entity = self._node_to_entity(record["e"])
            if entity.vector and len(entity.vector) == len(query_vector):
                sim = cosine_similarity(query_vector, entity.vector)
                scored.append((sim, entity))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [entity for _, entity in scored[: max(top_k, 0)]]

    async def neighbors(
        self, entity_ids: list[str], *, hops: int, limit: int
    ) -> SubGraph:
        driver = self._ensure_driver()
        try:
            # Use string-interpolated hops (int, no injection risk).
            cypher_entities = (
                f"MATCH (start:Entity) WHERE start.entity_id IN $entity_ids "
                f"MATCH (start)-[:RELATION*0..{int(hops)}]-(neighbor:Entity) "
                f"WITH DISTINCT neighbor "
                f"LIMIT {int(limit)} "
                f"RETURN collect(neighbor) AS entities"
            )
            cypher_relations = (
                "MATCH (h:Entity)-[r:RELATION]->(t:Entity) "
                "WHERE h.entity_id IN $ids AND t.entity_id IN $ids "
                "RETURN h, r, t"
            )
            async with driver.session(database=self.database) as session:
                result_e = await session.run(
                    cypher_entities, entity_ids=entity_ids
                )
                record_e = await result_e.single()
                result_r = await session.run(
                    cypher_relations,
                    ids=list({record["entity_id"]
                              for record in (record_e["entities"] if record_e else [])})
                    if record_e and record_e["entities"]
                    else list(entity_ids),
                )
                records_r = await _fetch_all(result_r)
        except Exception as exc:
            raise self._translate(exc, "neighbors") from exc

        nodes = record_e["entities"] if record_e and record_e["entities"] else []
        entities = [self._node_to_entity(n) for n in nodes]
        relations = [
            self._rel_to_relation(r["r"], r["h"]["entity_id"], r["t"]["entity_id"])
            for r in records_r
        ]
        return SubGraph(entities=entities, relations=relations)

    async def topics(
        self, query_vector: list[float], *, top_k: int
    ) -> list[TopicSummary]:
        if not self.capabilities.supports_community:
            return []
        driver = self._ensure_driver()
        try:
            async with driver.session(database=self.database) as session:
                # Fetch all entities + relations into memory.
                result_e = await session.run(
                    "MATCH (e:Entity {kb_id: $kb_id}) RETURN e",
                    kb_id=self.kb_id,
                )
                entities_records = await _fetch_all(result_e)
                result_r = await session.run(
                    "MATCH (h:Entity {kb_id: $kb_id})-[r:RELATION]->"
                    "(t:Entity {kb_id: $kb_id}) "
                    "RETURN h.entity_id AS head_id, t.entity_id AS tail_id",
                    kb_id=self.kb_id,
                )
                relations_records = await _fetch_all(result_r)
        except Exception as exc:
            raise self._translate(exc, "topics") from exc

        entities = [self._node_to_entity(r["e"]) for r in entities_records]
        if not entities:
            return []

        # Build in-memory graph and run community detection (reuses
        # networkx — a core dependency; the NetworkX store does the same).
        graph = nx.Graph()
        id_to_entity = {e.entity_id: e for e in entities}
        for e in entities:
            graph.add_node(e.entity_id)
        for r in relations_records:
            head, tail = r["head_id"], r["tail_id"]
            if head in id_to_entity and tail in id_to_entity:
                graph.add_edge(head, tail)

        if graph.number_of_edges() == 0:
            members_by_group: list[list[str]] = [list(graph.nodes)]
        else:
            try:
                members_by_group = [
                    [str(n) for n in community]
                    for community in nx.algorithms.community.greedy_modularity_communities(graph)
                ]
            except Exception:
                members_by_group = [list(graph.nodes)]

        members_by_group = [g for g in members_by_group if len(g) >= 1]
        if not members_by_group:
            return []

        topics: list[tuple[float, TopicSummary]] = []
        for members in members_by_group:
            ents = [id_to_entity[m] for m in members if m in id_to_entity]
            if not ents:
                continue
            top = sorted(ents, key=lambda e: e.confidence, reverse=True)[:3]
            title = " / ".join(e.name for e in top) or ents[0].type
            summary = "；".join(
                f"{e.name}：{e.description}" for e in top if e.description
            )[:300] or title
            # Centroid vector for similarity scoring.
            with_vec = [e for e in ents if e.vector]
            dim = len(with_vec[0].vector) if with_vec else 0
            centroid: list[float] = []
            if dim:
                centroid = [
                    sum(e.vector[i] for e in with_vec) / len(with_vec)
                    for i in range(dim)
                ]
            topic = TopicSummary(
                topic_id=new_id("tpc_"),
                kb_id=self.kb_id,
                title=title,
                summary=summary,
                member_entity_ids=[e.entity_id for e in ents],
                vector=centroid,
            )
            score = (
                cosine_similarity(query_vector, centroid)
                if query_vector and centroid and len(centroid) == len(query_vector)
                else 0.0
            )
            topics.append((score, topic))
        topics.sort(
            key=lambda pair: (pair[0], len(pair[1].member_entity_ids)), reverse=True
        )
        return [t for _, t in topics[: max(top_k, 0)]]

    # -- exception translation ----------------------------------------------
    @staticmethod
    def _translate(exc: Exception, context: str) -> Exception:
        exc_str = str(exc).lower()
        if "timeout" in exc_str or "timed out" in exc_str:
            return PluginTimeoutError(
                f"Neo4j {context} timed out", details={"error": str(exc)}
            )
        if "connection" in exc_str or "unreachable" in exc_str or "refused" in exc_str:
            return GraphStoreUnavailableError(
                f"Neo4j {context}: store unreachable", code=5002,
                details={"error": str(exc)},
            )
        if "constraint" in exc_str or "validation" in exc_str:
            return PluginContractError(
                f"Neo4j {context}: contract violation", details={"error": str(exc)}
            )
        return GraphStoreUnavailableError(
            f"Neo4j {context} failed", code=5002, details={"error": str(exc)}
        )

    # -- lifecycle ----------------------------------------------------------
    async def startup(self) -> None:
        self._ensure_driver()
        await self._ensure_constraints()
        await self._ensure_vector_index()

    async def shutdown(self) -> None:
        if self._driver is not None:
            try:
                await self._driver.close()
            except Exception:
                pass
            self._driver = None

    async def entity_count(self) -> int:
        driver = self._ensure_driver()
        try:
            async with driver.session(database=self.database) as session:
                result = await session.run(
                    "MATCH (e:Entity {kb_id: $kb_id}) RETURN count(e) AS cnt",
                    kb_id=self.kb_id,
                )
                record = await result.single()
                return int(record["cnt"]) if record else 0
        except Exception as exc:
            raise self._translate(exc, "entity_count") from exc

    async def relation_count(self) -> int:
        driver = self._ensure_driver()
        try:
            async with driver.session(database=self.database) as session:
                result = await session.run(
                    "MATCH ()-[r:RELATION]->() WHERE r.kb_id = $kb_id "
                    "RETURN count(r) AS cnt",
                    kb_id=self.kb_id,
                )
                record = await result.single()
                return int(record["cnt"]) if record else 0
        except Exception as exc:
            raise self._translate(exc, "relation_count") from exc
