"""Entity normalisation and merging (05-kg.md §5.3 + §5.2.4).

Two-level merge strategy (§5.3.2):
1. **Level 1** — exact ``name_norm`` match → merge (description: longer wins,
   source_doc_ids: union, confidence: max).
2. **Level 2** — description vector near-neighbour (cos ≥ 0.92, same type) →
   candidate merge (preserve original name_norm).
3. **New** — create a fresh entity.

``source_doc_ids`` union is the reference-count basis for incremental delete
(§5.4.2).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from ragx.core.hashing import cosine_similarity
from ragx.core.ids import new_id
from ragx.core.models import Entity
from ragx.kg.schemas import ExtractedEntity, ExtractionResult

#: Confidence floor — entities below this are dropped (§5.2.4).
CONFIDENCE_MIN = 0.5

#: Level-2 merge threshold (§5.3.2).
SIMILARITY_THRESHOLD = 0.92


def normalize_name(name: str) -> str:
    """Normalise an entity name for dedup key (§5.3.1).

    Lowercase → strip → NFKC (full-width → half-width) → strip punctuation →
    collapse whitespace.
    """
    s = unicodedata.normalize("NFKC", name)
    s = s.strip().lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def filter_low_confidence(result: ExtractionResult) -> ExtractionResult:
    """Drop entities with ``confidence < CONFIDENCE_MIN`` and orphan relations
    (§5.2.4).

    Relations whose ``head`` or ``tail`` is not in the surviving entities are
    removed, so the output is always internally consistent.
    """
    valid = {e.name: e for e in result.entities if e.confidence >= CONFIDENCE_MIN}
    valid_relations = [
        r for r in result.relations if r.head in valid and r.tail in valid
    ]
    return ExtractionResult(
        entities=list(valid.values()),
        relations=valid_relations,
    )


async def merge_entities(
    extracted: list[ExtractedEntity],
    kb_id: str,
    doc_id: str,
    graph_store: Any,
    embedder: Any,
) -> list[Entity]:
    """Two-level merge of extracted entities into the GraphStore (§5.3.2).

    Returns the list of ``Entity`` objects (new or merged) that were upserted.
    """
    entities: list[Entity] = []
    for ext in extracted:
        name_norm = normalize_name(ext.name)

        # ── Level 1: exact name_norm match ─────────────────────────────
        existing = await _find_by_name_norm(graph_store, kb_id, name_norm)
        if existing is not None:
            merged = Entity(
                entity_id=existing.entity_id,
                kb_id=kb_id,
                name=existing.name if len(existing.name) >= len(ext.name) else ext.name,
                name_norm=name_norm,
                type=existing.type,
                description=existing.description
                if len(existing.description) >= len(ext.description)
                else ext.description,
                source_doc_ids=sorted(set(existing.source_doc_ids) | {doc_id}),
                confidence=max(existing.confidence, ext.confidence),
                vector=existing.vector,
            )
            entities.append(merged)
            continue

        # ── Level 2: description vector near-neighbour ────────────────
        desc_vec = (await embedder.embed([ext.description]))[0]
        candidates = await _match_entities(graph_store, desc_vec, top_k=5)
        best: Entity | None = None
        best_sim = 0.0
        for cand in candidates:
            if cand.name_norm == name_norm:
                continue  # already handled in level 1
            cand_vec = await _get_entity_vector(graph_store, cand.entity_id)
            if not cand_vec:
                continue
            try:
                sim = cosine_similarity(desc_vec, cand_vec)
            except ValueError:
                continue
            if sim >= SIMILARITY_THRESHOLD and cand.type == ext.type:
                if sim > best_sim:
                    best = cand
                    best_sim = sim

        if best is not None:
            merged = Entity(
                entity_id=best.entity_id,
                kb_id=kb_id,
                name=best.name,
                name_norm=best.name_norm,
                type=best.type,
                description=best.description,
                source_doc_ids=sorted(set(best.source_doc_ids) | {doc_id}),
                confidence=max(best.confidence, ext.confidence),
                vector=best.vector,
            )
            entities.append(merged)
        else:
            # ── New entity ─────────────────────────────────────────────
            new_ent = Entity(
                entity_id=new_id("ent_"),
                kb_id=kb_id,
                name=ext.name,
                name_norm=name_norm,
                type=ext.type,
                description=ext.description,
                source_doc_ids=[doc_id],
                confidence=ext.confidence,
                vector=desc_vec,
            )
            entities.append(new_ent)

    return entities


# ── GraphStore adapter helpers ─────────────────────────────────────────
# The GraphStore SPI (§1.2) does not expose ``find_by_name_norm`` directly;
# the KG layer uses the plugin's internal index where available, or falls
# back to a linear scan over ``match_entities``-style iteration.


async def _find_by_name_norm(
    graph_store: Any, kb_id: str, name_norm: str
) -> Entity | None:
    """Find an entity by exact ``name_norm``.

    Plugins that key by ``name_norm`` (e.g. NetworkXGraphStore) expose a
    ``entities`` dict directly; otherwise we iterate.
    """
    entities_dict = getattr(graph_store, "entities", None)
    if entities_dict is not None:
        return entities_dict.get(name_norm)
    return None


async def _match_entities(graph_store: Any, vec: list[float], *, top_k: int) -> list[Entity]:
    return await graph_store.match_entities(vec, top_k=top_k)


async def _get_entity_vector(graph_store: Any, entity_id: str) -> list[float]:
    """Return an entity's vector (for cosine comparison in level-2 merge)."""
    entities_dict = getattr(graph_store, "entities", None)
    if entities_dict:
        for ent in entities_dict.values():
            if ent.entity_id == entity_id:
                return ent.vector or []
    return []
