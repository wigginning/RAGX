"""Knowledge Graph layer (05-kg.md): entity/relation extraction, normalisation,
incremental build, Dual-Level retrieval, and topic summaries.

Feature Flag ``kg_enabled`` (KBConfig.flags) gates both the ingest side
(``kg_building`` stage) and the retrieval side (graph_low / graph_high routes).
"""

from ragx.kg.builder import KGBuilder
from ragx.kg.extraction import (
    extract_auto,
    extract_from_chunk,
    extract_with_fallback,
)
from ragx.kg.merge import filter_low_confidence, merge_entities, normalize_name
from ragx.kg.retrieval import dual_level_retrieve
from ragx.kg.schemas import ExtractedEntity, ExtractedRelation, ExtractionResult
from ragx.kg.small_models import extract_small
from ragx.kg.topics import build_topics

__all__ = [
    "ExtractionResult",
    "ExtractedEntity",
    "ExtractedRelation",
    "KGBuilder",
    "build_topics",
    "dual_level_retrieve",
    "extract_auto",
    "extract_from_chunk",
    "extract_small",
    "extract_with_fallback",
    "filter_low_confidence",
    "merge_entities",
    "normalize_name",
]
