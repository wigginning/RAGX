"""Entity/relation extraction (05-kg.md §5.2).

Three extraction modes (§5.2.5):
* ``llm``   — LLM structured() full extraction (default)
* ``small`` — GLiNER NER + REBEL RE → ExtractionResult (zero API cost)
* ``hybrid``— small pre-extract → LLM augment/verify

All modes produce an ``ExtractionResult``; downstream merging (§5.3) and
filtering (§5.2.4) are mode-agnostic.
"""

from __future__ import annotations

import logging
from typing import Any

from ragx.core.exceptions import ExtractionSchemaError
from ragx.core.models import Chunk
from ragx.core.roles import LLMRole
from ragx.core.settings import KGExtractionConfig
from ragx.kg.schemas import ExtractionResult
from ragx.llm.prompts import PromptRegistry
from ragx.spi.interfaces import ChatMessage, ChatRequest

logger = logging.getLogger("ragx.kg.extraction")

CONFIDENCE_MIN = 0.5


async def extract_from_chunk(
    chunk: Chunk,
    llm: Any,
    prompts: PromptRegistry | None = None,
    *,
    trace_id: str | None = None,
    kb_overrides: dict[str, str] | None = None,
) -> ExtractionResult:
    """LLM structured() extraction (§5.2.2).

    Uses the ``extract`` role (cheap model) with ``temperature=0.0`` for
    determinism. The prompt is loaded from ``extract_entities.v1`` (§P1) and
    resolves per-kb overrides via ``kb_overrides`` (12-prompts.md §12.0).
    """
    if prompts is None:
        prompts = PromptRegistry()
    system, user = prompts.render_for_kb(
        "extract_entities", kb_overrides, chunk_text=chunk.text,
    )
    req = ChatRequest(
        messages=[
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=user),
        ],
        role=LLMRole.EXTRACT,
        temperature=0.0,
        json_mode=True,
        kb_id=chunk.kb_id,
        trace_id=trace_id,
    )
    try:
        return await llm.structured(req, schema=ExtractionResult)
    except ExtractionSchemaError:
        raise
    except Exception as exc:
        raise ExtractionSchemaError(
            "LLM structured extraction failed",
            details={"chunk_id": chunk.chunk_id, "error": str(exc)},
            trace_id=trace_id,
        ) from exc


async def extract_with_fallback(
    chunk: Chunk,
    llm: Any,
    prompts: PromptRegistry | None = None,
    *,
    trace_id: str | None = None,
    metrics: Any = None,
    kb_overrides: dict[str, str] | None = None,
) -> ExtractionResult | None:
    """Extract with one retry on schema failure, then degrade-skip (§5.2.3).

    Returns ``None`` when both attempts fail — the chunk is skipped and the
    skip is recorded in metrics (``ragx_kg_extraction_skipped_total``).
    """
    try:
        return await extract_from_chunk(
            chunk, llm, prompts, trace_id=trace_id, kb_overrides=kb_overrides,
        )
    except ExtractionSchemaError:
        # One retry (temperature=0.0 + schema hint already in the prompt)
        try:
            return await extract_from_chunk(
                chunk, llm, prompts, trace_id=trace_id, kb_overrides=kb_overrides,
            )
        except ExtractionSchemaError:
            logger.warning(
                "extraction schema validation failed after retry, skipping chunk %s",
                chunk.chunk_id,
            )
            if metrics is not None:
                metrics.kg_extraction_skipped.labels(kb=chunk.kb_id).inc()
            return None


async def extract_auto(
    chunk: Chunk,
    llm: Any,
    cfg: KGExtractionConfig,
    prompts: PromptRegistry | None = None,
    *,
    trace_id: str | None = None,
    metrics: Any = None,
    kb_overrides: dict[str, str] | None = None,
) -> ExtractionResult | None:
    """Route extraction by ``cfg.extractor_mode`` (§5.2.5).

    * ``llm``   → :func:`extract_with_fallback`
    * ``small`` → local NER+RE (stub: returns LLM result when models unavailable)
    * ``hybrid``→ small pre-extract → LLM augment via ``augment_extraction`` prompt
    """
    if cfg.extractor_mode == "llm":
        return await extract_with_fallback(
            chunk, llm, prompts,
            trace_id=trace_id, metrics=metrics, kb_overrides=kb_overrides,
        )

    if cfg.extractor_mode == "small":
        return await _small_extract(chunk, cfg, trace_id=trace_id)

    if cfg.extractor_mode == "hybrid":
        return await _hybrid_extract(
            chunk, llm, cfg, prompts,
            trace_id=trace_id, kb_overrides=kb_overrides,
        )

    return await extract_with_fallback(
        chunk, llm, prompts,
        trace_id=trace_id, metrics=metrics, kb_overrides=kb_overrides,
    )


async def _small_extract(
    chunk: Chunk,
    cfg: KGExtractionConfig,
    *,
    trace_id: str | None = None,
) -> ExtractionResult | None:
    """GLiNER + REBEL local extraction (§5.2.5 ``small`` mode).

    The implementation in :mod:`ragx.kg.small_models` uses the real GLiNER /
    REBEL models when their packages are installed, and falls back to a
    deterministic rule-based extractor otherwise. Both paths produce the same
    :class:`ExtractionResult` schema, so the rest of the pipeline is
    model-agnostic.
    """
    from ragx.kg.small_models import extract_small

    return await extract_small(chunk, cfg, trace_id=trace_id)


async def _hybrid_extract(
    chunk: Chunk,
    llm: Any,
    cfg: KGExtractionConfig,
    prompts: PromptRegistry | None = None,
    *,
    trace_id: str | None = None,
    kb_overrides: dict[str, str] | None = None,
) -> ExtractionResult | None:
    """Hybrid mode: small pre-extract → LLM augment (§5.2.5).

    The LLM receives the pre-extracted results and produces a merged
    ``ExtractionResult`` via the ``augment_extraction`` prompt (§P13).
    """
    pre = await _small_extract(chunk, cfg, trace_id=trace_id)
    if pre is None or (not pre.entities and not pre.relations):
        return await extract_with_fallback(
            chunk, llm, prompts,
            trace_id=trace_id, kb_overrides=kb_overrides,
        )

    if prompts is None:
        prompts = PromptRegistry()
    import json

    system, user = prompts.render_for_kb(
        "augment_extraction",
        kb_overrides,
        chunk_text=chunk.text,
        pre_extraction_json=json.dumps(pre.model_dump(), ensure_ascii=False),
    )
    req = ChatRequest(
        messages=[
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=user),
        ],
        role=LLMRole.EXTRACT,
        temperature=0.0,
        json_mode=True,
        kb_id=chunk.kb_id,
        trace_id=trace_id,
    )
    try:
        return await llm.structured(req, schema=ExtractionResult)
    except Exception as exc:
        logger.warning("hybrid augment failed, falling back to pre-extract: %s", exc)
        return pre
