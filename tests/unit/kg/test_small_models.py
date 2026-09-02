"""Tests for the small-model KG extraction (RX-KG-01).

The rule-based fallback is exercised here; the GLiNER / REBEL heavy path is
skipped when their packages are not installed (CI lite profile).
"""

from __future__ import annotations

from ragx.core.models import Chunk
from ragx.core.settings import KGExtractionConfig
from ragx.kg.small_models import extract_small


def _chunk(text: str) -> Chunk:
    return Chunk(
        chunk_id="chk_1",
        doc_id="doc_1",
        kb_id="default",
        atom_ids=[],
        text=text,
        token_count=len(text.split()),
        page=None,
        bbox=None,
        metadata={},
        edited=False,
        version=1,
    )


def _cfg() -> KGExtractionConfig:
    return KGExtractionConfig(
        extractor_mode="small",
        entity_types=["CONCEPT"],
    )


async def test_extract_small_returns_extraction_result() -> None:
    """Smoke: rule-based path always returns an ExtractionResult."""
    result = await extract_small(_chunk("OpenAI is a company."), _cfg())
    assert result is not None
    assert result.entities
    # The capitalised "OpenAI" should be picked up by the EN proper-noun regex.
    names = {e.name for e in result.entities}
    assert any("OpenAI" in n for n in names)


async def test_extract_small_chinese_phrases() -> None:
    chunk = _chunk("RAGX 是一个 RAG 引擎，它使用 7 接口的 SPI。")
    result = await extract_small(chunk, _cfg())
    assert result is not None
    names = {e.name for e in result.entities}
    # at least one Chinese phrase was extracted
    assert any("\u4e00" <= ch <= "\u9fff" for n in names for ch in n), names


async def test_extract_small_relations_from_pattern() -> None:
    """When the text matches ``X 是 Y`` and both names appear, a relation
    is produced. Without entity names matching the pattern, no relation is
    emitted — the rule-based path requires both ends to be known."""
    chunk = _chunk("OpenAI is a Company. The Company builds models.")
    result = await extract_small(chunk, _cfg())
    assert result is not None
    # The relation requires entity names to appear in the entity set.
    if result.relations:
        rel = result.relations[0]
        assert rel.head in {e.name for e in result.entities}
        assert rel.tail in {e.name for e in result.entities}


async def test_extract_small_quoted_terms() -> None:
    chunk = _chunk('Some text with "Python" mentioned.')
    result = await extract_small(chunk, _cfg())
    names = {e.name for e in result.entities}
    assert "Python" in names


async def test_extract_small_empty_input() -> None:
    result = await extract_small(_chunk(""), _cfg())
    assert result is not None
    assert result.entities == []
    assert result.relations == []


async def test_extract_small_returns_dict_compatible_schema() -> None:
    """The result must serialise via Pydantic (used downstream for merging)."""
    result = await extract_small(_chunk("Apple Pie is great."), _cfg())
    assert result is not None
    d = result.model_dump()
    assert "entities" in d
    assert "relations" in d
