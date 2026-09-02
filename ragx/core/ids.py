"""ULID identifier helpers (00-overview.md §0.4).

IDs are prefixed: ``doc_`` / ``chk_`` / ``ent_`` / ``rel_`` / ``task_`` / ``kb_``
plus ``tpc_`` (topics, 05-kg.md) and ``cache_`` (semantic cache entries,
08-llm.md §8.5.2). A ULID is 26 Crockford base-32 characters, lexicographically
sortable by time (ms resolution), URL safe.
"""

from __future__ import annotations

import os
import time

# Crockford base32 alphabet (01-spi/00-overview: exclude I, L, O, U).
_ALPHABET: str = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_MASK: int = 0x1F

#: Prefixes allowed by 00-overview.md §0.4 (+ tpc_ for topics).
ID_PREFIXES: tuple[str, ...] = (
    "doc_",
    "chk_",
    "ent_",
    "rel_",
    "task_",
    "kb_",
    "tpc_",
    "cache_",
    "key_",
    "trace_",
)


def ulid() -> str:
    """Return a fresh 26-char Crockford base-32 ULID.

    Layout: 48-bit millisecond timestamp (sortable) | 80-bit random.
    """
    timestamp = int(time.time() * 1000)
    random_bits = int.from_bytes(os.urandom(10), "big")
    value = (timestamp << 80) | random_bits
    chars: list[str] = []
    for index in range(25, -1, -1):
        chars.append(_ALPHABET[(value >> (index * 5)) & _MASK])
    return "".join(chars)


def new_id(prefix: str) -> str:
    """Return ``<prefix><ulid>``; raises ``ValueError`` for unknown prefixes."""
    if not prefix.endswith("_"):
        raise ValueError(f"id prefix must end with '_', got {prefix!r}")
    return f"{prefix}{ulid()}"


def ulid_timestamp_ms(value: str) -> int | None:
    """Best-effort decode of the leading 48-bit ms timestamp (observability aid)."""
    try:
        bits: int = 0
        for ch in value[:10]:
            bits = (bits << 5) | _ALPHABET.index(ch.upper())
        # only the low 48 bits hold the timestamp
        return bits & ((1 << 48) - 1)
    except (ValueError, TypeError):
        return None
