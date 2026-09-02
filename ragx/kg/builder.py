"""KGBuilder (05-kg.md §5.4 + §5.9).

The ingestion ``kg_building`` stage and the incremental re-index share this
entry point. ``build_for_chunks`` extracts → filters → merges → upserts;
``delete_chunk_from_graph`` decrements reference counts and removes orphaned
nodes (shared entities survive).
"""

from __future__ import annotations

import logging
from typing import Any

from ragx.core.exceptions import GraphBuildError
from ragx.core.ids import new_id
from ragx.core.models import Chunk, Entity, Relation
from ragx.core.settings import KBConfig, KGExtractionConfig
from ragx.kg.extraction import extract_auto
from ragx.kg.merge import filter_low_confidence, merge_entities
from ragx.kg.schemas import ExtractionResult

logger = logging.getLogger("ragx.kg.builder")


class KGBuilder:
    """Orchestrates entity extraction + graph build for a set of chunks."""

    def __init__(
        self,
        llm: Any,
        graph_store: Any,
        embedder: Any,
        kb_cfg: KBConfig,
        prompts: Any | None = None,
    ) -> None:
        self.llm = llm
        self.graph_store = graph_store
        self.embedder = embedder
        self.kb_cfg = kb_cfg
        self.prompts = prompts
        self._cfg: KGExtractionConfig = kb_cfg.kg

    # -- build -------------------------------------------------------------
    async def build_for_chunks(self, chunks: list[Chunk]) -> None:
        """Extract → filter → merge → upsert for each chunk (§5.4.1).

        Idempotent: upsert semantics mean repeated calls produce the same
        graph. Extraction failures are non-fatal (chunk is skipped).
        """
        all_entities: list[Entity] = []
        all_relations: list[Relation] = []

        for chunk in chunks:
            try:
                extracted = await extract_auto(
                    chunk, self.llm, self._cfg, self.prompts,
                    kb_overrides=self.kb_cfg.prompt_overrides,
                )
                if extracted is None:
                    continue
                extracted = filter_low_confidence(extracted)

                entities = await merge_entities(
                    extracted.entities,
                    chunk.kb_id,
                    chunk.doc_id,
                    self.graph_store,
                    self.embedder,
                )
                relations = await self._build_relations(
                    extracted, entities, chunk,
                )
                all_entities.extend(entities)
                all_relations.extend(relations)
            except GraphBuildError:
                raise
            except Exception as exc:
                logger.warning(
                    "kg build failed for chunk %s: %s", chunk.chunk_id, exc
                )

        if all_entities:
            await self.graph_store.upsert_entities(all_entities)
        if all_relations:
            await self.graph_store.upsert_relations(all_relations)

    # -- relations ---------------------------------------------------------
    async def _build_relations(
        self,
        extracted: ExtractionResult,
        entities: list[Entity],
        chunk: Chunk,
    ) -> list[Relation]:
        """Map extracted relations to ``Entity`` ids, drop dangling ones."""
        name_to_id: dict[str, str] = {e.name: e.entity_id for e in entities}
        # Also include existing entities (the merged list only has new/merged).
        entities_dict = getattr(self.graph_store, "entities", None)
        if entities_dict:
            for ent in entities_dict.values():
                name_to_id.setdefault(ent.name, ent.entity_id)

        relations: list[Relation] = []
        for rel in extracted.relations:
            head_id = name_to_id.get(rel.head)
            tail_id = name_to_id.get(rel.tail)
            if head_id is None or tail_id is None:
                continue  # dangling — skip
            relations.append(Relation(
                relation_id=new_id("rel_"),
                kb_id=chunk.kb_id,
                head_id=head_id,
                tail_id=tail_id,
                type=rel.type,
                description=rel.description,
                weight=rel.weight,
                source_doc_ids=[chunk.doc_id],
            ))
        return relations

    # -- delete (incremental re-index) ────────────────────────────────────
    async def delete_chunk_from_graph(self, chunk: Chunk) -> None:
        """Delete a chunk's graph contributions via reference counting (§5.4.2).

        Shared entities (referenced by other documents) are kept; their
        ``source_doc_ids`` is decremented. Entities whose count reaches zero
        are removed, along with their relations.
        """
        entities_dict = getattr(self.graph_store, "entities", None)
        if entities_dict is None:
            # The store does not expose internal state — use the SPI method.
            await self.graph_store.delete_by_doc(chunk.doc_id)
            return

        removed_nodes: set[str] = set()
        updated_entities: list[Entity] = []

        for _name_norm, entity in list(entities_dict.items()):
            if chunk.doc_id not in entity.source_doc_ids:
                continue
            remaining = [d for d in entity.source_doc_ids if d != chunk.doc_id]
            if remaining:
                # Still referenced — keep, update source_doc_ids
                updated = entity.model_copy(deep=True)
                updated.source_doc_ids = remaining
                updated_entities.append(updated)
            else:
                # Reference count reached zero — remove
                removed_nodes.add(entity.entity_id)

        if updated_entities:
            await self.graph_store.upsert_entities(updated_entities)

        # Delete entities whose reference count is zero.
        for node_id in removed_nodes:
            if hasattr(self.graph_store, "delete_entity"):
                await self.graph_store.delete_entity(node_id)

        # Cascade: delete relations whose head or tail points to a removed node.
        if removed_nodes and hasattr(self.graph_store, "delete_relations_by_entity"):
            await self.graph_store.delete_relations_by_entity(list(removed_nodes))
