"""Cost Ledger (08-llm.md §8.6).

Computes per-call USD cost from the pricing table and records a
:class:`CostRecord`. Missing unit prices degrade to ``cost_usd=0`` with a
warning (防漏配单价), never an error.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from pydantic import BaseModel

from ragx.core.models import TokenUsage
from ragx.core.roles import LLMRole
from ragx.core.settings import LLMRouterConfig, ProviderTarget

logger = logging.getLogger("ragx.llm.ledger")


class CostRecord(BaseModel):
    """One metered LLM call (08-llm.md §8.6.1)."""

    trace_id: str
    ts: str                      # UTC ISO-8601
    kb_id: str
    tenant_id: str
    role: str                    # LLMRole value
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    unit_price: dict[str, float] = {}
    cost_usd: float = 0.0
    cached: bool = False


def compute_cost(
    usage: TokenUsage, target: ProviderTarget, config: LLMRouterConfig
) -> tuple[float, dict[str, float]]:
    """Return ``(cost_usd, unit_price)`` for one call.

    ``cost_usd = prompt/1000 * prompt_per_1k + completion/1000 * completion_per_1k``.
    A missing price entry yields ``cost_usd=0`` plus a warning (08-llm.md §8.6.2).
    """
    price = config.unit_price(target)
    if not price:
        logger.warning(
            "no unit price configured for %s; cost_usd=0", target.key
        )
        return 0.0, {}
    prompt_per_1k = float(price.get("prompt_per_1k", 0.0))
    completion_per_1k = float(price.get("completion_per_1k", 0.0))
    cost = (
        usage.prompt_tokens / 1000.0 * prompt_per_1k
        + usage.completion_tokens / 1000.0 * completion_per_1k
    )
    return cost, price


def make_record(
    *,
    trace_id: str,
    kb_id: str,
    tenant_id: str,
    role: LLMRole,
    target: ProviderTarget,
    usage: TokenUsage,
    unit_price: dict[str, float],
    cost_usd: float,
    cached: bool = False,
) -> CostRecord:
    """Build a :class:`CostRecord` for persistence / observability."""
    return CostRecord(
        trace_id=trace_id,
        ts=datetime.now(UTC).isoformat(),
        kb_id=kb_id,
        tenant_id=tenant_id,
        role=role.value,
        provider=target.provider,
        model=target.model,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        unit_price=unit_price,
        cost_usd=cost_usd,
        cached=cached,
    )
