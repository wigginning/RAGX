"""Extraction tests (05-kg.md §5.2).

Covers: extract_with_fallback (5003 retry → skip), filter_low_confidence,
and extract_auto mode routing.
"""

from __future__ import annotations

from ragx.core.exceptions import StructuredParseError
from ragx.core.models import Chunk
from ragx.core.settings import KGExtractionConfig
from ragx.kg.extraction import (
    extract_auto,
    extract_from_chunk,
    extract_with_fallback,
)
from ragx.kg.merge import filter_low_confidence
from ragx.kg.schemas import ExtractedEntity, ExtractionResult


def _make_chunk(text: str = "sample text", kb_id: str = "kb_1") -> Chunk:
    return Chunk(
        chunk_id="chk_1", doc_id="doc_1", kb_id=kb_id,
        atom_ids=["doc_1#0001"], text=text, token_count=10,
    )


class _MockLLM:
    """Mock LLM that returns a fixed ExtractionResult."""

    def __init__(self, result: ExtractionResult | None = None) -> None:
        self._result = result or ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="OpenAI", type="ORG",
                    description="AI company", confidence=0.9,
                ),
            ],
            relations=[],
        )
        self._fail_count = 0
        self.call_count = 0

    def fail_once(self) -> None:
        self._fail_count = 1

    async def structured(self, req, *, schema):
        self.call_count += 1
        if self._fail_count > 0:
            self._fail_count -= 1
            raise StructuredParseError("test failure")
        return self._result


class TestExtraction:
    async def test_extract_from_chunk_success(self) -> None:
        llm = _MockLLM()
        chunk = _make_chunk()
        result = await extract_from_chunk(chunk, llm)
        assert len(result.entities) == 1
        assert result.entities[0].name == "OpenAI"
        assert llm.call_count == 1

    async def test_extract_with_fallback_retry_success(self) -> None:
        """First call fails, retry succeeds (§5.2.3)."""
        llm = _MockLLM()
        llm.fail_once()
        chunk = _make_chunk()
        result = await extract_with_fallback(chunk, llm)
        assert result is not None
        assert len(result.entities) == 1
        assert llm.call_count == 2  # initial + retry

    async def test_extract_with_fallback_retry_fails_skips(self) -> None:
        """Both calls fail → returns None (§5.2.3)."""
        llm = _MockLLM()
        llm._fail_count = 100  # always fail
        chunk = _make_chunk()
        result = await extract_with_fallback(chunk, llm)
        assert result is None
        assert llm.call_count == 2  # initial + retry

    async def test_extract_with_fallback_records_metric(self) -> None:
        """Metrics are incremented when extraction is skipped."""
        llm = _MockLLM()
        llm._fail_count = 100
        chunk = _make_chunk()

        class MockMetrics:
            class _Counter:
                def __init__(self): self._n = 0
                def labels(self, **kw): return self
                def inc(self): self._n += 1

            kg_extraction_skipped = _Counter()

        metrics = MockMetrics()
        result = await extract_with_fallback(chunk, llm, metrics=metrics)
        assert result is None
        assert metrics.kg_extraction_skipped._n == 1


class TestFilterLowConfidence:
    def test_filters_low_confidence_entities(self) -> None:
        result = ExtractionResult(
            entities=[
                ExtractedEntity(name="A", type="ORG", description="x", confidence=0.8),
                ExtractedEntity(name="B", type="ORG", description="y", confidence=0.3),
                ExtractedEntity(name="C", type="ORG", description="z", confidence=0.5),
            ],
            relations=[],
        )
        filtered = filter_low_confidence(result)
        names = [e.name for e in filtered.entities]
        assert "A" in names
        assert "B" not in names  # below 0.5
        assert "C" in names  # exactly 0.5

    def test_removes_orphan_relations(self) -> None:
        """Relations whose head/tail is filtered out are removed (§5.2.4)."""
        from ragx.kg.schemas import ExtractedRelation

        result = ExtractionResult(
            entities=[
                ExtractedEntity(name="A", type="ORG", description="x", confidence=0.8),
                ExtractedEntity(name="B", type="ORG", description="y", confidence=0.3),
            ],
            relations=[
                ExtractedRelation(head="A", tail="B", type="owns", description="d", weight=1.0),
            ],
        )
        filtered = filter_low_confidence(result)
        assert filtered.relations == []  # B is filtered → relation is orphan


class TestExtractAuto:
    async def test_llm_mode(self) -> None:
        llm = _MockLLM()
        cfg = KGExtractionConfig(extractor_mode="llm")
        chunk = _make_chunk()
        result = await extract_auto(chunk, llm, cfg)
        assert result is not None
        assert len(result.entities) == 1

    async def test_small_mode_returns_empty_when_unavailable(self) -> None:
        """Small models not installed → returns empty ExtractionResult (§5.2.5)."""
        llm = _MockLLM()
        cfg = KGExtractionConfig(extractor_mode="small")
        chunk = _make_chunk()
        result = await extract_auto(chunk, llm, cfg)
        assert result is not None
        assert result.entities == []

    async def test_hybrid_mode_falls_back_to_llm(self) -> None:
        """Hybrid with no small model → falls back to LLM extraction."""
        llm = _MockLLM()
        cfg = KGExtractionConfig(extractor_mode="hybrid")
        chunk = _make_chunk()
        result = await extract_auto(chunk, llm, cfg)
        assert result is not None
        assert llm.call_count >= 1
