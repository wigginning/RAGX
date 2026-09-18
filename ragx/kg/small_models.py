"""Small-model entity / relation extraction (05-kg.md §5.2.5, RX-KG-01).

The production deployment uses GLiNER for zero-shot NER and REBEL for
relation extraction. Both models are heavy (~500 MB combined) and require
GPU/CPU resources that are not always available in CI or the lite profile.

This module ships **two paths**:

1. **Real path** — when ``gliner`` and ``transformers`` are installed, the
   ``extract_small`` function delegates to them. Output schema is identical
   to the rule-based path so the rest of the pipeline is model-agnostic.

2. **Rule-based fallback** — when the heavy deps are missing, a deterministic
   heuristic extracts entities (capitalised noun phrases, quoted terms,
   domain keywords) and relations (``X 是 Y``, ``X 是 Y 的 Z`` patterns).
   This is good enough for tests and provides a sane default in lite
   profiles; users wanting real extraction install ``ragx[kg-small]``.
"""

from __future__ import annotations

import logging
import re

from ragx.core.models import Chunk
from ragx.core.settings import KGExtractionConfig
from ragx.kg.schemas import (
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
)

logger = logging.getLogger("ragx.kg.small_models")

CONFIDENCE = 0.55  # conservative; rule-based path is less reliable than GLiNER


def _gliner_available() -> bool:
    try:
        import gliner  # type: ignore[import-not-found]

        return bool(gliner)
    except ImportError:
        return False


def _rebel_available() -> bool:
    try:
        import transformers  # type: ignore[import-not-found]

        return bool(transformers)
    except ImportError:
        return False


# -------------------------------------------------------------------
# Rule-based entity extraction
# -------------------------------------------------------------------
_CN_PHRASE = re.compile(
    r"[\u4e00-\u9fff]{2,6}(?:[\u4e00-\u9fff、， ]+[\u4e00-\u9fff]{1,6})?"
)
_EN_PROPER_NOUN = re.compile(
    r"(?<![A-Za-z])[A-Z][A-Za-z]{2,}(?:\s+[A-Z][A-Za-z]{2,})*\b"
)
_QUOTED = re.compile(r"[""「]([^""」]{2,30})[""」]")


def _extract_entities_rule(text: str, types: list[str]) -> list[ExtractedEntity]:
    """A very simple extractor: any capitalised noun phrase / quoted term /
    short Chinese phrase becomes an entity. Type defaults to the first
    configured type or ``"concept"``."""
    primary = types[0] if types else "concept"
    seen: set[str] = set()
    out: list[ExtractedEntity] = []
    for pattern in (_QUOTED, _EN_PROPER_NOUN, _CN_PHRASE):
        for match in pattern.finditer(text):
            name = match.group(0).strip(" ""「」")
            if not name or name in seen:
                continue
            seen.add(name)
            out.append(
                ExtractedEntity(
                    name=name,
                    type=primary,
                    description=f"{name}是一个{primary}",
                    confidence=CONFIDENCE,
                )
            )
    return out


_RELATION_PATTERNS = [
    # "X 是 Y", "X 是 Y 的 Z"
    re.compile(r"([\u4e00-\u9fffA-Za-z]{2,20})\s*是\s*([\u4e00-\u9fffA-Za-z]{2,20})"),
    # "X uses Y", "X contains Y"
    re.compile(r"([A-Z][a-zA-Z]+)\s+(uses|contains|includes|has)\s+([A-Z][a-zA-Z]+)"),
    # "X 的 Y 是 Z"  (X's Y is Z) -> X relates_to Z via Y
    re.compile(
        r"([\u4e00-\u9fff]{2,10})\s*的\s*([\u4e00-\u9fff]{2,10})\s*是\s*([\u4e00-\u9fff]{2,20})"
    ),
]


def _extract_relations_rule(
    text: str, entities: list[ExtractedEntity], confidence: float
) -> list[ExtractedRelation]:
    names = {e.name for e in entities}
    out: list[ExtractedRelation] = []
    for pattern in _RELATION_PATTERNS:
        for match in pattern.finditer(text):
            groups = [g for g in match.groups() if g]
            if len(groups) < 2:
                continue
            head, tail = groups[0], groups[1]
            if head in names and tail in names and head != tail:
                rel_type = "related_to" if pattern is _RELATION_PATTERNS[2] else "is_a"
                out.append(
                    ExtractedRelation(
                        head=head,
                        tail=tail,
                        type=rel_type,
                        description=match.group(0),
                        weight=confidence,
                    )
                )
    return out


async def _rule_based_extract(
    chunk: Chunk, cfg: KGExtractionConfig
) -> ExtractionResult:
    entities = _extract_entities_rule(chunk.text, list(cfg.entity_types))
    relations = _extract_relations_rule(chunk.text, entities, CONFIDENCE)
    return ExtractionResult(entities=entities, relations=relations)


# -------------------------------------------------------------------
# Real GLiNER + REBEL path
# -------------------------------------------------------------------
async def _gliner_rebel_extract(
    chunk: Chunk, cfg: KGExtractionConfig
) -> ExtractionResult:
    """Load GLiNER + REBEL once, cache them on the function object."""
    import gliner  # type: ignore[import-not-found]
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore[import-not-found]

    model = getattr(_gliner_rebel_extract, "_ner", None)
    if model is None:
        model = gliner.GLiNER.from_pretrained(cfg.small_ner_model)
        _gliner_rebel_extract._ner = model  # type: ignore[attr-defined]
    tokenizer = getattr(_gliner_rebel_extract, "_re_tokenizer", None)
    re_model = getattr(_gliner_rebel_extract, "_re_model", None)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(cfg.small_re_model)
        re_model = AutoModelForSeq2SeqLM.from_pretrained(cfg.small_re_model)
        _gliner_rebel_extract._re_tokenizer = tokenizer  # type: ignore[attr-defined]
        _gliner_rebel_extract._re_model = re_model  # type: ignore[attr-defined]

    # GLiNER entity extraction
    types = list(cfg.entity_types) or ["entity"]
    ner_results = model.predict_entities(chunk.text, types)
    entities = [
        ExtractedEntity(
            name=r["text"],
            type=r["label"],
            description=f"{r['text']}是一个{r['label']}",
            confidence=float(r.get("score", CONFIDENCE)),
        )
        for r in ner_results
    ]

    # REBEL relation extraction (returns triplets as decoded text)
    relations: list[ExtractedRelation] = []
    try:
        import torch  # local import — optional

        inputs = tokenizer(
            chunk.text[:512],
            return_tensors="pt",
            truncation=True,
        )
        assert re_model is not None  # REBEL model loaded/assigned above
        with torch.no_grad():
            outputs = re_model.generate(**inputs, max_length=256)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        # REBEL emits "<triplet> X <subj> Y <obj>" tokens; parse them.
        for triplet in _rebel_parse(decoded):
            relations.append(
                ExtractedRelation(
                    head=triplet["head"],
                    tail=triplet["tail"],
                    type=triplet.get("type", "related_to"),
                    description=decoded,
                    weight=CONFIDENCE + 0.1,
                )
            )
    except Exception as exc:  # noqa: BLE001 - degrade silently
        logger.debug("REBEL relation extraction failed: %s", exc)

    return ExtractionResult(entities=entities, relations=relations)


_REBEL_TRIPLET = re.compile(
    r"<triplet>\s*(?P<head>[^<]+)\s*<subj>\s*(?P<tail>[^<]+)\s*<obj>"
)


def _rebel_parse(text: str) -> list[dict[str, str]]:
    """Parse REBEL's structured output into ``{head, tail, type}`` dicts."""
    return [
        {"head": m.group("head").strip(), "tail": m.group("tail").strip(), "type": "related_to"}
        for m in _REBEL_TRIPLET.finditer(text)
    ]


# -------------------------------------------------------------------
# Public dispatch
# -------------------------------------------------------------------
async def extract_small(
    chunk: Chunk,
    cfg: KGExtractionConfig,
    *,
    trace_id: str | None = None,
) -> ExtractionResult | None:
    """Small-model extraction — picks the heavy ML path or the rule fallback.

    Returns ``None`` when both paths fail (callers treat this as a graceful
    skip; KG-01 §5.2.3).
    """
    if _gliner_available() and _rebel_available():
        try:
            return await _gliner_rebel_extract(chunk, cfg)
        except Exception as exc:  # noqa: BLE001 - degrade to rules
            logger.debug("GLiNER/REBEL failed, falling back to rules: %s", exc)
    return await _rule_based_extract(chunk, cfg)


__all__ = ["extract_small"]
