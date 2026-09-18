"""Content fingerprints: the basis of deduplication and description caching
(02-core.md §2.2).

| fingerprint        | algorithm                                   | used for
|--------------------|---------------------------------------------|---------------------------------
| ``doc_hash``       | ``sha256(normalize(bytes))``                | document-level dedup (2004)
| ``atom.content_hash`` | TEXT: ``sha256(text.strip())``; others: ``sha256(payload)`` | description cache key
| semantic cache key | ``kb_id + normalize(query)`` → vector near  | query-level cache (§8.5)
"""

from __future__ import annotations

import hashlib

# Unicode normalisation + BOM strip + CRLF/CR -> LF + trailing whitespace trim.
_BOMS: tuple[bytes, ...] = (
    b"\xef\xbb\xbf",  # UTF-8
    b"\xff\xfe",      # UTF-16 LE
    b"\xfe\xff",      # UTF-16 BE
)


def normalize_bytes(data: bytes) -> bytes:
    """Normalise raw document bytes: drop BOM, unify newlines, trim tail blanks."""
    out = data
    for bom in _BOMS:
        if out.startswith(bom):
            out = out[len(bom) :]
            break
    out = out.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return out.rstrip()


def normalize_text(text: str) -> str:
    """Normalise text the same way as :func:`normalize_bytes` (str convenience)."""
    return normalize_bytes(text.encode("utf-8")).decode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compute_doc_hash(content: bytes | str) -> str:
    """``doc_hash`` = sha256 of normalised bytes (02-core.md §2.2)."""
    raw = content.encode("utf-8") if isinstance(content, str) else content
    return sha256_hex(normalize_bytes(raw))


def compute_atom_content_hash(atom_text: str | None, payload: bytes | None = None) -> str:
    """``atom.content_hash`` (02-core.md §2.2).

    * TEXT/TABLE/FORMULA: hash of ``text.strip()`` (callers pass ``payload=None``)
    * IMAGE (and any binary atom): hash of the payload bytes
    """
    if payload is not None:
        return sha256_hex(payload)
    return sha256_hex((atom_text or "").strip().encode("utf-8"))


def normalize_query(query: str) -> str:
    """Query normalisation for semantic-cache keys (08-llm.md §8.5.1).

    lower-case → strip punctuation → drop stopwords → collapse whitespace.
    Deliberately lightweight (no external NLP dependency in the lite profile).
    """
    _STOP = {
        "的", "了", "是", "在", "和", "与", "及", "或", "也", "都", "很", "有", "被", "把",
        "对", "于", "以", "为", "等", "吗", "呢", "啊", "吧", "这", "那", "如何", "怎么",
        "what", "how", "why", "when", "where", "who", "which", "the", "a", "an", "is",
        "are", "of", "to", "in", "on", "for", "and", "or", "with", "by", "from", "that",
        "this", "it", "its", "as", "at", "do", "does", "did", "can", "could", "would",
    }
    lowered = query.casefold()
    chars: list[str] = []
    for ch in lowered:
        if ch.isalnum() or ch.isspace():
            chars.append(ch)
        elif ch in "-_/":
            chars.append(" ")
    words = [w for w in "".join(chars).split() if w not in _STOP and len(w) > 1]
    return " ".join(words)


def cosine_similarity(a: list[float] | tuple[float, ...], b: list[float] | tuple[float, ...]) -> float:
    """Pure-python cosine similarity; returns 0.0 for degenerate vectors."""
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} vs {len(b)}")
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / ((na**0.5) * (nb**0.5))
