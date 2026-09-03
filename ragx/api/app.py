"""FastAPI app factory (09-api.md).

Wires the registry, metadata store, queue, pipeline, retriever and query
service into a FastAPI app with the documented routes and error handling.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from ragx.api.errors import register_exception_handlers
from ragx.api.middleware import (
    AuditMiddleware,
    AuthMiddleware,
    InMemoryAuditStore,
    InMemoryQuotaStore,
    MetadataAuditStore,
    MetadataQuotaStore,
    QuotaMiddleware,
    RateLimiter,
    RateLimitMiddleware,
    TraceMiddleware,
)
from ragx.api.routes import audit, chat, health, ingest, metrics, search
from ragx.api.routes.traces import router as traces_router
from ragx.core.settings import KBConfig, Settings
from ragx.ingestion.pipeline import IngestionPipeline
from ragx.ingestion.queue_redis import make_queue
from ragx.ingestion.store import MetadataStore
from ragx.llm.prompts import PromptRegistry
from ragx.llm.semantic_cache import SemanticCache
from ragx.observability.metrics import get_metrics as _get_metrics
from ragx.observability.rag_trace import RAGTraceStore
from ragx.plugins import register_builtins
from ragx.plugins.cache_memory import InMemoryExactCache
from ragx.plugins.cache_redis import RedisCache
from ragx.retrieval.assembler import ContextAssembler
from ragx.retrieval.hybrid import GraphChunkResolver, HybridRetriever
from ragx.retrieval.models import RetrievalConfig
from ragx.retrieval.pipeline import QueryService
from ragx.retrieval.router import QueryRouter
from ragx.spi.registry import PluginRegistry


def _build_llm_router(
    settings: Settings,
    registry: PluginRegistry,
    kb_cfg: KBConfig,
    *,
    cache: Any | None = None,
) -> Any | None:
    """Build the ResilientModelRouter from ``settings.llm`` (08-llm.md §8.3).

    ``cache`` is the wired :class:`SemanticCache` (LLM-02); when provided the
    router consults/stores it for ``GENERATE`` requests.

    Returns ``None`` when no LLM roles are configured — the service then runs
    in retrieval-only mode (chat/agentic degrade; search still works).
    """
    if not settings.llm.roles:
        return None
    from ragx.llm.router import ResilientRouter

    router = ResilientRouter(
        settings.llm, registry, tenant_id="default", cache=cache
    )
    router.budget_caps = {"default": kb_cfg.budgets.daily_cost_limit_usd}
    return router


def _build_agentic_service(
    standard_service: QueryService,
    retriever: HybridRetriever,
    assembler: ContextAssembler,
    embedder: Any,
    graph_store: Any,
    prompts: PromptRegistry,
    llm: Any,
    kb_cfg: KBConfig,
    trace_store: Any | None = None,
    registry: Any | None = None,
) -> QueryService:
    """Wire the AgenticOrchestrator into a QueryService (07-agentic.md §7.9).

    The orchestrator's ``standard_fallback`` is the agentic-free
    ``standard_service`` — so a degrade path can never recurse back into the
    agentic route.
    """
    from ragx.agentic.executor import ParallelExecutor
    from ragx.agentic.orchestrator import AgenticOrchestrator
    from ragx.agentic.planner import Planner
    from ragx.agentic.synthesis import Verifier

    planner = Planner(llm, kb_cfg, prompts, graph_store)
    executor = ParallelExecutor(llm, retriever, assembler, embedder)
    verifier = Verifier(llm)
    orchestrator = AgenticOrchestrator(
        planner,
        executor,
        verifier,
        llm,
        standard_service,
        kb_cfg,
        trace_id="",
        budget_cap=kb_cfg.budgets.agentic_token_budget,
        embedder=embedder,
    )
    return QueryService(
        retriever, assembler, standard_service.router, prompts, llm,
        kb_cfg=kb_cfg, retrieval_cfg=standard_service.retrieval_cfg,
        agentic=orchestrator, trace_store=trace_store, registry=registry,
    )


def create_app(
    *,
    settings: Settings | None = None,
    kb_cfg: KBConfig | None = None,
    db: MetadataStore | None = None,
    registry: PluginRegistry | None = None,
    llm: Any | None = None,
) -> FastAPI:
    settings = settings or Settings()
    kb_cfg = kb_cfg or KBConfig()
    registry = registry or PluginRegistry(settings)
    register_builtins(registry)
    db = db or MetadataStore(":memory:")
    queue = make_queue(settings.queue)
    prompts = PromptRegistry()
    # Per-key token bucket (lite=in-process, full=Redis when a URL is set).
    rate_limiter = RateLimiter.create(
        settings.security, redis_url=settings.queue.redis_url
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await db.connect()
        await trace_store.connect()
        await registry.startup_all()
        if settings.queue.auto_consume:
            # lite compose smoke: consume the in-process queue so uploads are
            # processed without an external worker (03-ingestion.md §3.7).
            async def _consume(task_id: str) -> None:
                task = await db.get_task(task_id)
                if task is not None:
                    await pipeline.run(task)

            await queue.consume(_consume)
        yield
        await queue.shutdown()
        await rate_limiter.shutdown()
        await registry.shutdown_all()
        await db.close()
        await trace_store.close()

    app = FastAPI(title="RAGX", version="0.1.0", lifespan=lifespan)
    # Order (outermost -> innermost): audit -> trace -> auth -> rate-limit -> routes.
    # FastAPI runs middlewares in REVERSE registration order, so to make
    # audit outermost we register it last. Audit captures every request,
    # including ones rejected by auth or rate-limit.
    # Default to in-memory audit (lite profile); the full profile wires the
    # persistent MetadataAuditStore via env (RAGX_AUDIT_BACKEND=metadata).
    audit_store: Any = InMemoryAuditStore()
    if getattr(settings, "audit_backend", None) == "metadata":
        audit_store = MetadataAuditStore(db)
    quota_store: Any = InMemoryQuotaStore()
    if getattr(settings, "audit_backend", None) == "metadata":
        quota_store = MetadataQuotaStore(db)
    # Middleware execution order (outermost -> innermost):
    #   Audit -> Auth -> Quota -> RateLimit -> Trace
    # Auth MUST run before Quota and RateLimit: both key their buckets by the
    # auth context (request.state.tenant_id / request.state.auth.key_id).
    # With them outside Auth the context is never resolved in time, every key
    # shares one "anonymous" bucket and every tenant one "default" quota — one
    # tenant could exhaust the shared bucket and 429/quota-block everyone else.
    app.add_middleware(TraceMiddleware)
    app.add_middleware(RateLimitMiddleware, limiter=rate_limiter)
    app.add_middleware(QuotaMiddleware, config=settings.security)
    app.add_middleware(AuthMiddleware, security=settings.security, store=db)
    app.add_middleware(AuditMiddleware)
    register_exception_handlers(app)

    # resolve the lite plugins
    embedder = registry.resolve_from_kb("embedder", kb_cfg, kb_id="default")
    vector_store = registry.resolve_from_kb("vector_store", kb_cfg, kb_id="default")
    graph_store = registry.resolve_from_kb("graph_store", kb_cfg, kb_id="default") \
        if kb_cfg.graph_store else None

    # Semantic cache (LLM-02): exact tier (Redis for the full profile,
    # in-memory for lite) + vector tier that reuses the retrieval store. The
    # vector tier writes synthetic ``__semantic_cache__`` docs; the query path
    # excludes them so retrieval is never polluted (see QueryService.query).
    cache_cfg = settings.llm.cache
    redis_url = getattr(settings.queue, "redis_url", None)
    if redis_url:
        exact_cache: Any = RedisCache(
            {"url": redis_url, "namespace": cache_cfg.namespace, "ttl_s": cache_cfg.ttl_s}
        )
    else:
        exact_cache = InMemoryExactCache(
            {"namespace": cache_cfg.namespace, "ttl_s": cache_cfg.ttl_s}
        )
    semantic_cache = SemanticCache(
        embedder, vector_store, exact_cache=exact_cache, config=cache_cfg
    )

    # RAG Trace store (OBS-04 §10.5). Lite uses in-memory; a production
    # deployment can point this at the metadata SQLite path to persist traces.
    trace_store = RAGTraceStore(":memory:")

    retrieval_cfg = RetrievalConfig()
    resolver = GraphChunkResolver(vector_store) if graph_store is not None else None
    retriever = HybridRetriever(
        vector_store, graph_store, resolver, retrieval_cfg,
        kg_enabled=kb_cfg.flags.kg_enabled,
    )
    assembler = ContextAssembler(retrieval_cfg)
    router = QueryRouter()

    # Resolve the LLM (explicit param wins; otherwise build from settings.llm
    # so chat/agentic generation works in the deployed lite/full compose).
    if llm is None:
        llm = _build_llm_router(settings, registry, kb_cfg, cache=semantic_cache)

    # Standard-only service (agentic=None) — used both as the default query
    # path and as the AgenticOrchestrator's degradation fallback (§7.6).
    standard_service = QueryService(
        retriever, assembler, router, prompts, llm,
        kb_cfg=kb_cfg, retrieval_cfg=retrieval_cfg,
        trace_store=trace_store, registry=registry,
    )

    query_service = standard_service
    if kb_cfg.flags.agentic_enabled and llm is not None:
        query_service = _build_agentic_service(
            standard_service, retriever, assembler, embedder, graph_store,
            prompts, llm, kb_cfg, trace_store=trace_store, registry=registry,
        )

    pipeline = IngestionPipeline(registry, kb_cfg, db, llm=llm, prompts=prompts)

    app.state.settings = settings
    app.state.kb_cfg = kb_cfg
    app.state.registry = registry
    app.state.db = db
    app.state.queue = queue
    app.state.rate_limiter = rate_limiter
    app.state.prompts = prompts
    app.state.embedder = embedder
    app.state.vector_store = vector_store
    app.state.graph_store = graph_store
    app.state.retriever = retriever
    app.state.query_service = query_service
    app.state.pipeline = pipeline
    app.state.audit_store = audit_store
    app.state.quota_store = quota_store
    app.state.semantic_cache = semantic_cache
    app.state.trace_store = trace_store

    app.include_router(search.router, prefix="/v1")
    app.include_router(ingest.router, prefix="/v1")
    app.include_router(chat.router, prefix="/v1")
    app.include_router(health.router, prefix="/v1")
    app.include_router(metrics.router, prefix="/v1")
    app.include_router(audit.router, prefix="/v1")
    app.include_router(traces_router, prefix="/v1")

    # MCP SSE transport (09-api.md §9.7.2) — only when explicitly enabled.
    if settings.mcp.enabled:
        from ragx.api.routes.mcp import router as mcp_router

        app.include_router(mcp_router, prefix="/v1")

    app.state.metrics = _get_metrics()
    return app
