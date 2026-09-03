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
from ragx.observability.rag_trace import RAGTraceCollector, RAGTraceStore, TraceLLMCall
from ragx.retrieval.assembler import ContextAssembler
from ragx.retrieval.hybrid import GraphChunkResolver, HybridRetriever
from ragx.retrieval.models import RetrievalConfig
from ragx.retrieval.router import QueryRouter

logger = logging.getLogger("ragx.retrieval.pipeline")

#: Synthetic doc_id written by the semantic cache's vector tier
#: (08-llm.md §8.5). Retrieved results must never surface these.
CACHE_DOC_ID = "__semantic_cache__"

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
        trace_store: RAGTraceStore | None = None,
        registry: Any | None = None,
    ) -> None:
        self.retriever = retriever
        self.assembler = assembler
        self.router = router
        self.prompts = prompts
        self.llm = llm
        self.kb_cfg = kb_cfg
        self.retrieval_cfg = retrieval_cfg
        #: Plugin registry used to resolve a kb-scoped retriever at query time
        #: (see ``_retriever_for``). ``None`` keeps the constructor-supplied
        #: ``retriever`` (used by unit tests with fakes).
        self.registry = registry
        #: Per-kb retriever cache (one HybridRetriever per kb_id).
        self._retriever_cache: dict[str, HybridRetriever] = {}
        #: AgenticOrchestrator (07-agentic.md §7.9). ``None`` → the agentic
        #: route degrades to standard (critical-path behaviour).
        self.agentic = agentic
        #: Optional injected Metrics; when None the process-wide singleton is
        #: resolved lazily so tests calling ``reset_metrics()`` stay correct.
        self._metrics = metrics
        #: Optional RAG Trace store (OBS-04 §10.5). When set, every query
        #: produces a replayable trace. ``None`` → no trace capture.
        self._trace_store = trace_store

    # -- observability helpers (10-observability.md §10.2) -----------------
    def _retriever_for(self, kb_id: str) -> HybridRetriever:
        """Return a retriever scoped to ``kb_id``.

        The plugin registry caches each plugin instance per
        ``(interface, name, kb_id)``, so the ingest pipeline and the query
        path must resolve the *same* kb-scoped vector/graph store — otherwise
        a lite in-memory store (per-kb instance) ingested under ``kb_e2e`` is
        invisible to a retriever built from the ``"default"`` store. The
        search route already resolves per kb; the query path now does too.

        When no registry is injected we fall back to the constructor-supplied
        ``retriever`` (unit tests with fakes).
        """
        if self.registry is None:
            return self.retriever
        cached = self._retriever_cache.get(kb_id)
        if cached is not None:
            return cached
        store = self.registry.resolve_from_kb("vector_store", self.kb_cfg, kb_id=kb_id)
        graph_store = (
            self.registry.resolve_from_kb("graph_store", self.kb_cfg, kb_id=kb_id)
            if self.kb_cfg.graph_store else None
        )
        resolver = GraphChunkResolver(store) if graph_store is not None else None
        retr = HybridRetriever(
            store, graph_store, resolver, self.retrieval_cfg,
            kg_enabled=self.kb_cfg.flags.kg_enabled,
        )
        self._retriever_cache[kb_id] = retr
        return retr

    def _metrics_obj(self) -> Any:
        if self._metrics is not None:
            return self._metrics
        from ragx.observability.metrics import get_metrics

        return get_metrics()

    @staticmethod
    def _exclude_cache(filter_expr: FilterExpr | None) -> FilterExpr:
        """Merge a ``doc_id != CACHE_DOC_ID`` exclusion into the retrieval filter.

        The semantic cache's vector tier stores synthetic chunks under
        ``__semantic_cache__``; they must never appear in query results
        (08-llm.md §8.5). The exclusion is appended to the ``and`` group so it
        always applies regardless of any caller-supplied filter.
        """
        exclusion = {"field": "doc_id", "op": "ne", "value": CACHE_DOC_ID}
        # FilterExpr declares `and_`/`or_` with pydantic aliases "and"/"or" and
        # populate_by_name=True; mypy's pydantic plugin mis-reads keyword-aliased
        # fields here, so the constructor calls below are runtime-valid.
        if filter_expr is None:
            return FilterExpr(and_=[exclusion])  # type: ignore[call-arg]
        if filter_expr.and_:
            return FilterExpr(  # type: ignore[call-arg]
                and_=list(filter_expr.and_) + [exclusion], or_=filter_expr.or_
            )
        # Caller supplied only an ``or`` group → keep it, AND the exclusion.
        return FilterExpr(  # type: ignore[call-arg]
            and_=[exclusion], or_=filter_expr.or_
        )

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

    async def _flush_trace(
        self,
        collector: RAGTraceCollector | None,
        result: QueryResult,
        *,
        citations: list[Any],
        context: str,
    ) -> None:
        """Persist a RAG Trace for the finished query (OBS-04 §10.5).

        Never raises — trace capture must never break the query path.
        """
        if collector is None or self._trace_store is None:
            return
        try:
            collector.set_assembled(
                len(context or ""), list(citations), self._trace_budget()
            )
            collector.set_degraded(bool(getattr(result, "degraded", False)))
            collector.set_usage(result.usage)
            await self._trace_store.save(collector.build())
        except Exception:  # pragma: no cover - defensive
            logger.debug("failed to persist RAG trace", exc_info=True)

    def _trace_budget(self) -> int:
        """Token budget for the assembled context (best-effort, OBS-04)."""
        return int(getattr(self.retrieval_cfg, "token_budget", 0) or 0)

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
        # RAG Trace collector (OBS-04 §10.5); None when no store is wired.
        collector = (
            RAGTraceCollector(trace_id, kb_id, query)
            if self._trace_store is not None else None
        )

        mode = await self.router.route(query, self.kb_cfg, override)
        if collector is not None:
            collector.set_mode(mode)

        if mode == "fast":
            self._record_query(kb_id, "fast", "ok", time.perf_counter() - started)
            result = QueryResult(
                answer="", citations=[], mode="fast", trace_id=trace_id,
                usage=TokenUsage(), cost_usd=0.0,
            )
            await self._flush_trace(collector, result, citations=[], context="")
            return result

        if mode == "agentic":
            if self.agentic is not None:
                result = await self.agentic.run_agentic(query, kb_id, qvec)
                if collector is not None:
                    collector.set_mode(getattr(result, "mode", "agentic"))
                self._record_query(
                    kb_id, getattr(result, "mode", "agentic"), "ok",
                    time.perf_counter() - started,
                )
                await self._flush_trace(
                    collector, result, citations=result.citations, context=""
                )
                return result
            # No orchestrator wired in → degrade to standard (critical path).
            mode = "standard"

        retrieve_started = time.perf_counter()
        hits = await self._retriever_for(kb_id).retrieve(query, qvec, self._exclude_cache(filter_expr))
        self._stage(kb_id, mode, "retrieve", time.perf_counter() - retrieve_started)
        if collector is not None:
            collector.add_dense_hits(hits)

        if not hits:
            logger.info("empty recall (4002) for kb=%s", kb_id)
            try:
                self._metrics_obj().recall_empty.labels(kb=kb_id).inc()
            except Exception:  # pragma: no cover - defensive
                logger.debug("failed to record empty-recall metric", exc_info=True)
            self._record_query(
                kb_id, mode, "empty", time.perf_counter() - started
            )
            result = QueryResult(
                answer=_EMPTY_ANSWER, citations=[], mode="standard",
                trace_id=trace_id, usage=TokenUsage(), cost_usd=0.0,
                details={"empty": True, "reason": "4002"},
            )
            await self._flush_trace(collector, result, citations=[], context="")
            return result

        context, citations = self.assembler.assemble(hits)
        generate_started = time.perf_counter()
        answer = await self._generate(
            query, context, kb_id, trace_id, collector=collector
        )
        self._stage(kb_id, mode, "llm", time.perf_counter() - generate_started)
        self._record_query(kb_id, mode, "ok", time.perf_counter() - started)
        result = QueryResult(
            answer=answer,
            citations=citations,
            mode="standard",
            trace_id=trace_id,
            usage=TokenUsage(),
            cost_usd=0.0,
        )
        await self._flush_trace(collector, result, citations=citations, context=context)
        return result

    async def _generate(
        self,
        query: str,
        context: str,
        kb_id: str,
        trace_id: str,
        collector: RAGTraceCollector | None = None,
    ) -> str:
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
        if collector is not None:
            usage = getattr(resp, "usage", None)
            collector.add_llm_call(TraceLLMCall(
                role="generate",
                model=getattr(resp, "model", "") or "",
                tokens=usage if isinstance(usage, TokenUsage) else TokenUsage(),
                cost_usd=float(getattr(resp, "cost_usd", 0.0) or 0.0),
                prompt_name="generate_standard",
            ))
        return resp.text
