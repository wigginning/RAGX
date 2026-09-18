"""HashEmbedder - zero-dependency deterministic embedder (lite profile).

This plugin exists because the ``lite`` profile must run with no external model
downloads (DESIGN.md §1.2: 零重型依赖). It is *not* a semantic model: it maps
text to a fixed-size vector through signed character n-gram hashing, so that
shared n-grams raise cosine similarity. That is enough for:

* retrieval smoke tests and the full critical path end-to-end
* deterministic unit/contract tests (no network, no model cache)

For production recall quality switch to the ``embed-st`` extra (e.g.
``BAAI/bge-m3``) or an API embeder - ``KBConfig.embedder`` alone, no code
change (11-plugins-builtin.md §11.5.7 path ③).
"""

from __future__ import annotations

from typing import Any

_NAME = "hash"
_DEFAULT_NGRAMS = (2, 3, 4)


def _tokenize(text: str, ngrams: tuple[int, ...]) -> list[str]:
    """Character n-grams: language-agnostic, works for CJK without a tokenizer."""
    text = text.strip()
    if not text:
        return []
    grams: list[str] = []
    for n in ngrams:
        if len(text) < n:
            grams.append(text)
            continue
        for i in range(len(text) - n + 1):
            grams.append(text[i : i + n])
    return grams


def hash_embed(
    text: str, dimension: int = 256, ngrams: tuple[int, ...] = _DEFAULT_NGRAMS
) -> list[float]:
    """Signed feature-hashing embedding, L2 normalised."""
    vec = [0.0] * dimension
    for gram in _tokenize(text, ngrams):
        digest = 0
        for ch in gram:
            digest = (digest * 131 + ord(ch)) & 0xFFFFFFFF
        idx = digest % dimension
        sign = 1.0 if (digest >> 16) & 1 else -1.0
        vec[idx] += sign
    norm = sum(x * x for x in vec) ** 0.5
    if norm == 0.0:
        vec[0] = 1.0
        return vec
    return [x / norm for x in vec]


class HashEmbedder:
    """SPI ``Embedder`` backed by deterministic hashing."""

    name: str = _NAME

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.dimension: int = int(cfg.get("dim", 256))
        self.max_batch_size: int = int(cfg.get("max_batch_size", 32))
        n = cfg.get("ngrams")
        self._ngrams: tuple[int, ...] = tuple(int(x) for x in n) if n else _DEFAULT_NGRAMS

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), self.max_batch_size):
            for text in texts[start : start + self.max_batch_size]:
                out.append(hash_embed(text or "", self.dimension, self._ngrams))
        return out

    async def startup(self) -> None:  # pragma: no cover - trivial hook
        return None

    async def shutdown(self) -> None:  # pragma: no cover - trivial hook
        return None
