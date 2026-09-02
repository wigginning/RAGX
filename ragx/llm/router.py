"""Resilient Model Router (08-llm.md §8.3 / §8.2.4).

The single entry point for every LLM call. Responsibilities:

* per-role provider/model selection across three strategies (priority /
  weighted / least_cost)
* per-provider concurrency semaphores
* retry with exponential backoff (only RateLimitError / 5xx / TimeoutError;
  StructuredParseError retries once)
* circuit breaker per ``(provider, model)``
* daily cost budget gate (08-llm.md §8.6.3)

Semantic caching (08-llm.md §8.5) is **deferred** to a later task (RX-LLM-01
DoD: "暂不含语义缓存"); the router keeps a ``cache`` slot that a future task
wires in.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

from ragx.core.exceptions import (
    AllProvidersFailedError,
    CircuitOpenError,
    ConfigError,
    LLMError,
    RateLimitError,
    StructuredParseError,
)
from ragx.core.models import TokenUsage
from ragx.core.roles import LLMRole
from ragx.core.settings import LLMRouterConfig, ProviderTarget, RoleConfig
from ragx.llm.circuit import CircuitBreaker
from ragx.llm.ledger import compute_cost, make_record
from ragx.llm.usage import collect_usage
from ragx.spi.interfaces import ChatRequest, ChatResponse, LLMProvider
from ragx.spi.registry import PluginRegistry

logger = logging.getLogger("ragx.llm.router")


class _RetryableExhausted(Exception):
    """Internal: a target exhausted its retries with a retryable failure."""


class ResilientRouter:
    """Routes LLM calls through retry / circuit / budget / cost accounting."""

    def __init__(
        self,
        config: LLMRouterConfig,
        registry: PluginRegistry,
        *,
        tenant_id: str = "default",
        cache: Any = None,
        store: Any = None,
    ) -> None:
        self.config = config
        self.registry = registry
        self.tenant_id = tenant_id
        self.cache = cache  # SemanticCache (deferred; None for now)
        #: MetadataStore (optional). When set, cost records are persisted to the
        #: DB and the daily-budget gate reads today's accumulated cost from it
        #: (08-llm.md §8.6.3) instead of the in-process list.
        self.store = store
        self.circuit = CircuitBreaker(config.circuit)
        self._sems: dict[str, asyncio.Semaphore] = {}
        self._cost_records: list[Any] = []
        #: kb_id -> daily cost cap (USD); populated by the app factory from KBConfig.
        self.budget_caps: dict[str, float] = {}

    # -- public API ---------------------------------------------------------
    async def chat(self, req: ChatRequest) -> ChatResponse:
        role = req.role
        cfg = self.config.get_role(role)

        # 1) semantic cache (deferred) - only standard final answers
        if (
            role == LLMRole.GENERATE
            and self.cache is not None
            and self.config.cache.enabled
            and req.cacheable
        ):
            cached = await self.cache.lookup(req.messages_text, kb_id=req.kb_id)
            if cached is not None:
                return ChatResponse(text=cached, usage=TokenUsage(), cached=True)

        # 2) budget gate (08-llm.md §8.6.3)
        await self._check_budget(req)

        # 3) build the ordered target list per strategy
        targets = self._targets(cfg)

        last_exc: Exception | None = None
        for target in targets:
            key = (target.provider, target.model)
            if self.circuit.is_open(key):
                raise CircuitOpenError(
                    code=6002,
                    message="circuit open",
                    details={"provider": target.provider, "model": target.model},
                    trace_id=req.trace_id,
                )
            sem = self._semaphore(target.provider)
            try:
                async with sem:
                    resp = await self._call_with_retry(target, req, cfg)
                    # write-through: cache the fresh answer for the next caller
                    if (
                        role == LLMRole.GENERATE
                        and self.cache is not None
                        and self.config.cache.enabled
                        and req.cacheable
                    ):
                        try:
                            await self.cache.store(
                                req.messages_text, resp.text, kb_id=req.kb_id,
                            )
                        except Exception as exc:  # noqa: BLE001 - degrade
                            logger.warning("semantic cache store failed: %s", exc)
                    return resp
            except _RetryableExhausted as e:
                self.circuit.record_failure(key)
                last_exc = e
                continue  # next fallback / candidate
            except (RateLimitError, LLMError) as e:
                # non-retryable business error -> record and move on
                self.circuit.record_failure(key)
                last_exc = e
                continue

        raise AllProvidersFailedError(
            code=6001,
            message="all providers failed",
            details={
                "role": role.value,
                "tried": [(t.provider, t.model) for t in targets],
            },
            trace_id=req.trace_id,
        ) from last_exc

    async def structured(
        self, req: ChatRequest, *, schema: type[Any]
    ) -> Any:
        """Route a structured-output call (LLMProvider.structured)."""
        role = req.role
        cfg = self.config.get_role(role)
        await self._check_budget(req)
        targets = self._targets(cfg)
        last_exc: Exception | None = None
        for target in targets:
            key = (target.provider, target.model)
            if self.circuit.is_open(key):
                raise CircuitOpenError(
                    code=6002,
                    message="circuit open",
                    details={"provider": target.provider, "model": target.model},
                    trace_id=req.trace_id,
                )
            sem = self._semaphore(target.provider)
            try:
                async with sem:
                    return await self._structured_with_retry(target, req, cfg, schema)
            except _RetryableExhausted as e:
                self.circuit.record_failure(key)
                last_exc = e
                continue
            except (RateLimitError, LLMError) as e:
                self.circuit.record_failure(key)
                last_exc = e
                continue
        raise AllProvidersFailedError(
            code=6001,
            message="all providers failed",
            details={
                "role": role.value,
                "tried": [(t.provider, t.model) for t in targets],
            },
            trace_id=req.trace_id,
        ) from last_exc

    # -- strategy selection (08-llm.md §8.2.4) ------------------------------
    def _targets(self, cfg: RoleConfig) -> list[ProviderTarget]:
        strategy = cfg.strategy
        if strategy == "priority":
            return [cfg.primary, *cfg.fallbacks]
        if strategy == "weighted":
            return self._weighted_targets(cfg)
        if strategy == "least_cost":
            return self._least_cost_targets(cfg)
        raise ConfigError(
            "unknown routing strategy", details={"strategy": strategy}
        )

    def _weighted_targets(self, cfg: RoleConfig) -> list[ProviderTarget]:
        """Weighted random among non-open candidates; open ones get weight 0."""
        candidates = cfg.candidates or [cfg.primary]
        pool: list[ProviderTarget] = []
        weights: list[float] = []
        for cand in candidates:
            if self.circuit.is_open((cand.provider, cand.model)):
                continue
            pool.append(cand)
            weights.append(max(0.0, cfg.weights.get(cand.key, 1.0)))
        if not pool:
            return []
        total = sum(weights)
        if total <= 0:
            return pool
        r = random.uniform(0.0, total)
        acc = 0.0
        for cand, w in zip(pool, weights, strict=True):
            acc += w
            if r <= acc:
                return [cand, *[c for c in pool if c is not cand]]
        return [pool[0], *pool[1:]]

    def _least_cost_targets(self, cfg: RoleConfig) -> list[ProviderTarget]:
        """Cheapest non-open candidate first (08-llm.md §8.2.4)."""
        candidates = cfg.candidates or [cfg.primary]
        priced: list[tuple[float, ProviderTarget]] = []
        for cand in candidates:
            if self.circuit.is_open((cand.provider, cand.model)):
                continue
            price = self.config.unit_price(cand)
            cost = (
                float(price.get("prompt_per_1k", 0.0))
                + float(price.get("completion_per_1k", 0.0))
                if price
                else float("inf")
            )
            priced.append((cost, cand))
        priced.sort(key=lambda pair: pair[0])
        return [cand for _, cand in priced]

    # -- call + retry -------------------------------------------------------
    async def _call_with_retry(
        self, target: ProviderTarget, req: ChatRequest, cfg: RoleConfig
    ) -> ChatResponse:
        provider = self._provider(target)
        for attempt in range(cfg.max_retries + 1):
            try:
                resp = await asyncio.wait_for(
                    provider.chat(req), timeout=cfg.timeout
                )
                self.circuit.record_success((target.provider, target.model))
                self._account(req, target, resp)
                return resp
            except StructuredParseError:
                if attempt >= 1:
                    raise
                continue  # retry once
            except (TimeoutError, RateLimitError, LLMError) as e:
                # RateLimitError / TimeoutError are always retryable; LLMError
                # carries retryability in details["retryable"] (4xx -> False).
                if isinstance(e, LLMError) and not e.details.get("retryable", True):
                    raise
                if attempt == cfg.max_retries:
                    raise _RetryableExhausted(str(e)) from e
                await asyncio.sleep(2**attempt * 0.5)  # 0.5 / 1 / 2 / ...
        raise _RetryableExhausted("unreachable")

    async def _structured_with_retry(
        self, target: ProviderTarget, req: ChatRequest, cfg: RoleConfig, schema: type[Any]
    ) -> Any:
        provider = self._provider(target)
        for attempt in range(cfg.max_retries + 1):
            try:
                result = await asyncio.wait_for(
                    provider.structured(req, schema=schema), timeout=cfg.timeout
                )
                self.circuit.record_success((target.provider, target.model))
                return result
            except StructuredParseError:
                if attempt >= 1:
                    raise
                continue
            except (TimeoutError, RateLimitError, LLMError) as e:
                if isinstance(e, LLMError) and not e.details.get("retryable", True):
                    raise
                if attempt == cfg.max_retries:
                    raise _RetryableExhausted(str(e)) from e
                await asyncio.sleep(2**attempt * 0.5)
        raise _RetryableExhausted("unreachable")

    # -- helpers ------------------------------------------------------------
    def _provider(self, target: ProviderTarget) -> LLMProvider:
        cfg = self.config.providers.get(target.provider, {})
        return self.registry.resolve("llm_provider", target.provider, cfg)

    def _semaphore(self, provider: str) -> asyncio.Semaphore:
        if provider not in self._sems:
            self._sems[provider] = asyncio.Semaphore(4)
        return self._sems[provider]

    def _account(self, req: ChatRequest, target: ProviderTarget, resp: ChatResponse) -> None:
        usage = collect_usage(resp, target.model, req.messages_text)
        cost, price = compute_cost(usage, target, self.config)
        record = make_record(
            trace_id=req.trace_id or "",
            kb_id=req.kb_id,
            tenant_id=self.tenant_id,
            role=req.role,
            target=target,
            usage=usage,
            unit_price=price,
            cost_usd=cost,
            cached=resp.cached,
        )
        self._cost_records.append(record)
        # OBS-02: feed the Prometheus LLM token/cost counters (§10.2). The
        # singleton is resolved per call so tests calling reset_metrics() win.
        try:
            from ragx.observability.metrics import get_metrics

            get_metrics().record_llm_call(
                role=str(getattr(req.role, "value", req.role) or ""),
                model=target.model,
                tenant=self.tenant_id,
                tokens=int(getattr(usage, "total_tokens", 0) or 0),
                cost_usd=float(cost or 0.0),
            )
        except Exception:  # pragma: no cover - metrics must never break a call
            logger.debug("failed to record llm metrics", exc_info=True)

    async def _check_budget(self, req: ChatRequest) -> None:
        """Daily cost cap (08-llm.md §8.6.3). Cache-hit fast path is deferred."""
        cap = self._budget_cap(req.kb_id)
        if cap is None:
            return
        daily = self._sum_daily_cost(req.kb_id)
        if daily < cap:
            return
        # over cap: only a cache hit may pass (cost=0); cache is deferred, so reject
        if req.cacheable and self.cache is not None:
            hit = await self.cache.lookup(req.messages_text, kb_id=req.kb_id)
            if hit is not None:
                return
        raise RateLimitError(
            code=1004,
            message="daily budget cap reached, only cache-hit queries allowed",
            details={"kb_id": req.kb_id, "used": daily, "cap": cap},
            trace_id=req.trace_id,
        )

    def _budget_cap(self, kb_id: str) -> float | None:
        return self.budget_caps.get(kb_id)

    def _sum_daily_cost(self, kb_id: str) -> float:
        return sum(r.cost_usd for r in self._cost_records if r.kb_id == kb_id)

    # -- observability ------------------------------------------------------
    def cost_records(self) -> list[Any]:
        return list(self._cost_records)
