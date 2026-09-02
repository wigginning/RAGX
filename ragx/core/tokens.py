"""Token counting (04-chunking.md §4.7).

Lives in ``core`` because both domain services (``chunking``, ``retrieval``) and
plugins need it, and plugins must not import domain-service modules
(00-overview.md §0.1 rule 3).

Default is ``tiktoken cl100k_base`` when the optional dependency is installed;
otherwise a deterministic heuristic counter is used so the ``lite`` profile
still runs with zero extra packages. Heuristic scale: ``1 token ~= 4 chars``
for ASCII and ``1 token ~= 1.6 chars`` for CJK, which tracks cl100k closely
enough for budget decisions.
"""

from __future__ import annotations

import re
from typing import Protocol

_ASCII_WORD = re.compile(r"[A-Za-z0-9_`]+")
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]")


class TokenCounter(Protocol):
    name: str

    def count(self, text: str) -> int: ...


class TiktokenCounter:
    """Exact cl100k_base counting via the optional ``tiktoken`` package."""

    name = "tiktoken-cl100k"

    def __init__(self, encoding: str = "cl100k_base") -> None:
        import tiktoken  # optional dependency

        self._encoding = tiktoken.get_encoding(encoding)

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._encoding.encode(text))


class HeuristicCounter:
    """Dependency-free counter used when ``tiktoken`` is unavailable."""

    name = "heuristic-cl100k"

    def count(self, text: str) -> int:
        if not text:
            return 0
        ascii_tokens = len(_ASCII_WORD.findall(text))
        # short ascii runs count as one token; the remainder is roughly 1/4 char
        consumed = sum(len(m.group(0)) for m in _ASCII_WORD.finditer(text))
        residual_ascii = max(0, len(text) - consumed)
        cjk_chars = len(_CJK.findall(text))
        return max(
            1,
            ascii_tokens + (residual_ascii - cjk_chars) // 4 + cjk_chars // 1,
        )


_counter_cache: dict[str, TokenCounter] = {}


def get_counter(name: str = "tiktoken-cl100k") -> TokenCounter:
    """Resolve a counter by name, falling back to the heuristic one."""
    if name in _counter_cache:
        return _counter_cache[name]
    counter: TokenCounter
    if name.startswith("tiktoken"):
        try:
            counter = TiktokenCounter()
        except Exception:
            counter = HeuristicCounter()
    else:
        counter = HeuristicCounter()
    _counter_cache[name] = counter
    return counter


def count_tokens(text: str, counter_name: str = "tiktoken-cl100k") -> int:
    """Convenience wrapper used across ingestion, chunking and retrieval."""
    return get_counter(counter_name).count(text)


def active_counter_name() -> str:
    """Report which counter is in effect (observability / startup logs)."""
    return get_counter().name
