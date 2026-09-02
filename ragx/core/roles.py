"""Cross-cutting LLM role enum.

08-llm.md §8.2.1 places this in ``ragx/llm/roles.py``; the canonical
definition lives here because ``core.settings.Settings`` (roles config) and
``ragx.spi.interfaces.ChatRequest`` (SPI contract) both need the enum, and
``core`` must not depend on ``spi``/``llm`` (00-overview.md §0.1).
``ragx.llm.roles`` re-exports this symbol so the documented import path works.
"""

from __future__ import annotations

from enum import StrEnum


class LLMRole(StrEnum):
    """Roles map to role-scoped provider/model config (08-llm.md §8.2.1)."""

    EXTRACT = "extract"        # entity/relation extraction - cheap model
    REWRITE = "rewrite"        # query rewriting / seed retry
    PLAN = "plan"              # agentic planning - cheap model
    SYNTHESIZE = "synthesize"  # agentic synthesis - flagship model
    GENERATE = "generate"      # Standard final answer / agentic sub-answer
    DESCRIBE = "describe"      # multimodal atom description (VLM)
    EMBED = "embed"            # vectorisation
    RERANK = "rerank"          # reranking
