"""QueryService (06-retrieval.md §6.8).

Composes the Standard path: route -> retrieve -> assemble -> generate via the
``generate_standard.v1`` prompt (12-prompts.md P12). Empty recall produces an
honest answer (4002 metric) rather than an exception.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ragx.core.models import (
    FilterExpr,
    QueryResult,
    RequestOverride,
    TokenUsage,
)
from ragx.core.settings import KBConfig
from ragx.llm.prompts import PromptRegistry
from ragx.retrieval.assembler import ContextAssembler
from ragx.retrieval.hybrid import HybridRetriever
from ragx.retrieval.models import RetrievalConfig
from ragx.retrieval.router import QueryRouter

logger = logging.getLogger("ragx.retrieval.pipeline")

_EMPTY_ANSWER = "知识库中未检索到相关内容。"


class QueryService:
    def __init__(
        self,
        retriever: HybridRetriever,
        assembler: ContextAssembler,
        router: QueryRouter,
        prompts: PromptRegistry,
        llm: Any,
        *,
        kb_cfg: KBConfig,
        retrieval_cfg: RetrievalConfig,
        agentic: Any | None = None,
        metrics: Any | None = None,
    ) -> None:
        self.retriever = retriever
        self.assembler = assembler
        self.router = router
        self.prompts = prompts
        self.llm = llm
        self.kb_cfg = kb_cfg
        self.retrieval_cfg = retrieval_cfg
        #: AgenticOrchestrator (07-agentic.md §7.9). ``None`` → the agentic
        #: route degrades to standard (critical-path behaviour).
        self.agentic = agentic
        #: Optional injected Metrics; when None the process-wide singleton is
        #: resolved lazily so tests calling ``reset_metrics()`` stay correct.
        self._metrics = metrics

    # -- observability helpers (10-observability.md §10.2) -----------------
    def _metrics_obj(self) -> Any:
        if self._metrics is not None:
            return self._metrics
        from ragx.observability.metrics import get_metrics

        return get_metrics()

    def _stage(self, kb: str, mode: str, stage: str, seconds: float) -> None:
        """Record one stage latency. Metrics must never break the query path."""
        try:
            self._metrics_obj().record_stage_latency(kb, mode, stage, seconds)
        except Exception:  # pragma: no cover - defensive
            logger.debug("failed to record stage latency", exc_info=True)

    def _record_query(
        self, kb: str, mode: str, status: str, total_s: float
    ) -> None:
        """Record query_total + total latency (10-observability.md §10.2)."""
        try:
            self._metrics_obj().record_query(kb, mode, status, total_s)
        except Exception:  # pragma: no cover - defensive
            logger.debug("failed to record query metric", exc_info=True)

    async def query(
        self,
        query: str,
        qvec: list[float],
        *,
        kb_id: str,
        trace_id: str,
        override: RequestOverride | None = None,
        filter_expr: FilterExpr | None = None,
    ) -> QueryResult:
        override = override or RequestOverride()
        started = time.perf_counter()
        mode = await self.router.route(query, self.kb_cfg, override)

        if mode == "fast":
            self._record_query(kb_id, "fast", "ok", time.perf_counter() - started)
            return QueryResult(
                answer="", citations=[], mode="fast", trace_id=trace_id,
                usage=TokenUsage(), cost_usd=0.0,
            )

        if mode == "agentic":
            if self.agentic is not None:
                result = await self.agentic.run_agentic(query, kb_id, qvec)
                self._record_query(
                    kb_id, getattr(result, "mode", "agentic"), "ok",
                    time.perf_counter() - started,
                )
                return result
            # No orchestrator wired in → degrade to standard (critical path).
            mode = "standard"

        retrieve_started = time.perf_counter()
        hits = await self.retriever.retrieve(query, qvec, filter_expr)
        self._stage(kb_id, mode, "retrieve", time.perf_counter() - retrieve_started)

        if not hits:
            logger.info("empty recall (4002) for kb=%s", kb_id)
            try:
                self._metrics_obj().recall_empty.labels(kb=kb_id).inc()
            except Exception:  # pragma: no cover - defensive
                logger.debug("failed to record empty-recall metric", exc_info=True)
            self._record_query(
                kb_id, mode, "empty", time.perf_counter() - started
            )
            return QueryResult(
                answer=_EMPTY_ANSWER, citations=[], mode="standard",
                trace_id=trace_id, usage=TokenUsage(), cost_usd=0.0,
                details={"empty": True, "reason": "4002"},
            )

        context, citations = self.assembler.assemble(hits)
        generate_started = time.perf_counter()
        answer = await self._generate(query, context, kb_id, trace_id)
        self._stage(kb_id, mode, "llm", time.perf_counter() - generate_started)
        self._record_query(kb_id, mode, "ok", time.perf_counter() - started)
        return QueryResult(
            answer=answer,
            citations=citations,
            mode="standard",
            trace_id=trace_id,
            usage=TokenUsage(),
            cost_usd=0.0,
        )

    async def _generate(self, query: str, context: str, kb_id: str, trace_id: str) -> str:
        system, user = self.prompts.render_for_kb(
            "generate_standard",
            self.kb_cfg.prompt_overrides,
            query=query,
            assembled_context=context,
        )
        from ragx.core.roles import LLMRole
        from ragx.spi.interfaces import ChatMessage, ChatRequest

        req = ChatRequest(
            messages=[
                ChatMessage(role="system", content=system),
                ChatMessage(role="user", content=user),
            ],
            role=LLMRole.GENERATE,
            kb_id=kb_id,
            trace_id=trace_id,
        )
        resp = await self.llm.chat(req)
        return resp.text
