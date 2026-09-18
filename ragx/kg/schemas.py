"""Extraction schemas (05-kg.md §5.2.1).

These Pydantic models are the target of the LLM's ``structured()`` call.
The prompt's JSON Schema is generated from these same models (12-prompts.md
§12.0 convention: prompt YAML references ``schema: ragx.kg.schemas.ExtractionResult``).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

#: Allowed entity types (05-kg.md §5.2.5 KGExtractionConfig.entity_types).
DEFAULT_ENTITY_TYPES: list[str] = [
    "PERSON", "ORG", "LOCATION", "CONCEPT", "EVENT", "PRODUCT", "DATE", "OTHER",
]


class ExtractedEntity(BaseModel):
    """One entity extracted from a chunk (§5.2.1)."""

    name: str
    type: str                              # PERSON/ORG/CONCEPT/...
    description: str
    confidence: float = Field(ge=0.0, le=1.0)


class ExtractedRelation(BaseModel):
    """One relation extracted from a chunk (§5.2.1)."""

    head: str                              # entity name (must appear in entities)
    tail: str
    type: str
    description: str
    weight: float = 1.0


class ExtractionResult(BaseModel):
    """Target schema for LLM ``structured()`` calls (§5.2.1)."""

    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)
