"""QueryRouter (06-retrieval.md §6.5).

Three-tier routing decision: fast (semantic-cache hit) -> agentic -> standard.
The agentic path is only reachable when the feature flag is enabled; an
explicit ``agentic`` override without the flag degrades to standard.
"""

from __future__ import annotations

from typing import Any, Literal

from ragx.core.models import RequestOverride
from ragx.core.settings import KBConfig


class _Heuristics:
    entity_connectors = ["和", "与", "以及", "and", "比较", "对比", "关系"]
    entity_marker = "的"
    length_threshold = 40
    confidence_threshold = 0.3


class QueryRouter:
    def __init__(self, cache: Any | None = None, heuristics: _Heuristics | None = None) -> None:
        self.cache = cache
        self.heuristics = heuristics or _Heuristics()

    async def route(
        self, query: str, kb_cfg: KBConfig, override: RequestOverride
    ) -> Literal["fast", "standard", "agentic"]:
        # ① fast: semantic-cache hit (cache is deferred in the critical path)
        if kb_cfg.flags.cache_enabled and override.mode != "agentic" and self.cache is not None:
            hit = await self.cache.lookup(query)
            if hit is not None:
                return "fast"

        # explicit override
        if override.mode == "agentic":
            if not kb_cfg.flags.agentic_enabled:
                return "standard"
            return "agentic"
        if override.mode == "standard":
            return "standard"

        # ② heuristic
        if kb_cfg.flags.agentic_enabled and self._heuristic_agentic(query):
            return "agentic"

        # ③ fallback
        return "standard"

    def _heuristic_agentic(self, query: str) -> bool:
        h = self.heuristics
        if any(c in query for c in h.entity_connectors):
            if query.count(h.entity_marker) >= 2:
                return True
        if len(query) > h.length_threshold:
            return True
        return False
