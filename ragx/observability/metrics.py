"""Prometheus metrics registry (10-observability.md §10.2).

Implements the full metric table from §10.2. Exposed via ``GET /metrics``
(OBS-02 wires the route). All metrics are process-scoped and use the label
convention from §10.2's "label consistency" note.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, generate_latest
from prometheus_client import Counter as PromCounter
from prometheus_client import Gauge as PromGauge
from prometheus_client import Histogram as PromHistogram

#: Histogram buckets for latency (seconds) — tuned for RAG sub-second to multi-second.
_LATENCY_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
)


class Metrics:
    """The §10.2 metric catalogue, all in one registry.

    Label consistency (§10.2):
    * ``mode`` = ``QueryResult.mode`` (fast/standard/agentic), never the request-side ``auto``.
    * ``stage`` matches the span-name suffix (retrieve/rerank/assemble/llm).
    * ``tenant`` comes from the API key; lite profile uses ``"default"``.
    """

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        self._init_metrics()

    def _init_metrics(self) -> None:
        r = self.registry

        self.query_total = PromCounter(
            "ragx_query_total", "Query count", ["kb", "mode", "status"],
            registry=r,
        )
        self.query_latency = PromHistogram(
            "ragx_query_latency_seconds", "Query latency by stage",
            ["kb", "mode", "stage"], buckets=_LATENCY_BUCKETS, registry=r,
        )
        self.llm_tokens = PromCounter(
            "ragx_llm_tokens_total", "LLM token usage", ["role", "model", "tenant"],
            registry=r,
        )
        self.llm_cost = PromCounter(
            "ragx_llm_cost_usd_total", "LLM cost in USD", ["role", "model", "tenant"],
            registry=r,
        )
        self.cache_hit_ratio = PromGauge(
            "ragx_cache_hit_ratio", "Semantic cache hit ratio", ["kb"], registry=r,
        )
        self.ingest_task_total = PromCounter(
            "ragx_ingest_task_total", "Ingest task count", ["kb", "status"], registry=r,
        )
        self.ingest_duration = PromHistogram(
            "ragx_ingest_task_duration_seconds", "Ingest stage duration",
            ["stage"], buckets=_LATENCY_BUCKETS, registry=r,
        )
        self.recall_empty = PromCounter(
            "ragx_retrieval_recall_empty_total",
            "Empty-recall occurrences (4002)", ["kb"], registry=r,
        )
        self.llm_retry = PromCounter(
            "ragx_llm_retry_total", "LLM retry count", ["role", "model"], registry=r,
        )
        self.circuit_state = PromGauge(
            "ragx_circuit_state",
            "Circuit state 0=closed,1=open,2=half_open", ["model"], registry=r,
        )
        self.agentic_task_total = PromCounter(
            "ragx_agentic_task_total", "Agentic task outcomes",
            ["kb", "outcome"], registry=r,
        )
        self.budget_exceeded = PromCounter(
            "ragx_budget_exceeded_total", "Token budget exceeded (6003)",
            ["kb"], registry=r,
        )
        self.ingest_parse = PromCounter(
            "ragx_ingest_parse_total", "Parse result count",
            ["parser", "result"], registry=r,
        )
        self.context_overflow = PromCounter(
            "ragx_context_overflow_chunks",
            "Chunks truncated at assembly (06 §6.4)", ["kb"], registry=r,
        )
        self.kg_extraction_skipped = PromCounter(
            "ragx_kg_extraction_skipped_total",
            "KG extraction skipped (5003 degrade)", ["kb"], registry=r,
        )
        self.cache_invalidation = PromCounter(
            "ragx_cache_invalidation_total",
            "Semantic cache invalidation events", ["kb", "reason"], registry=r,
        )
        self.agentic_degrade = PromCounter(
            "ragx_agentic_degrade_total", "Agentic degrade count", ["reason"], registry=r,
        )
        self.agentic_budget_used = PromGauge(
            "ragx_agentic_budget_used",
            "Agentic token consumption distribution", registry=r,
        )

    # -- convenience recorders -------------------------------------------
    def record_query(self, kb: str, mode: str, status: str, total_s: float) -> None:
        """Record one query: counter + latency histogram (stage=total)."""
        self.query_total.labels(kb=kb, mode=mode, status=status).inc()
        self.query_latency.labels(kb=kb, mode=mode, stage="total").observe(total_s)

    def record_stage_latency(
        self, kb: str, mode: str, stage: str, seconds: float
    ) -> None:
        self.query_latency.labels(kb=kb, mode=mode, stage=stage).observe(seconds)

    def record_llm_call(
        self,
        role: str,
        model: str,
        tenant: str,
        tokens: int,
        cost_usd: float,
    ) -> None:
        self.llm_tokens.labels(role=role, model=model, tenant=tenant).inc(tokens)
        if cost_usd:
            self.llm_cost.labels(role=role, model=model, tenant=tenant).inc(cost_usd)

    def record_circuit(self, model: str, state: str) -> None:
        """``state`` is the breaker's string state (closed/open/half_open)."""
        mapping = {"closed": 0, "open": 1, "half_open": 2}
        self.circuit_state.labels(model=model).set(mapping.get(state, 0))

    def record_agentic_outcome(self, kb: str, outcome: str) -> None:
        self.agentic_task_total.labels(kb=kb, outcome=outcome).inc()

    def render(self) -> bytes:
        """Prometheus exposition format for ``GET /metrics``."""
        return generate_latest(self.registry)


_metrics: Metrics | None = None


def get_metrics(registry: CollectorRegistry | None = None) -> Metrics:
    """Process-wide metrics singleton."""
    global _metrics
    if _metrics is None:
        _metrics = Metrics(registry)
    return _metrics


def reset_metrics() -> None:
    """Reset the global metrics (test helper)."""
    global _metrics
    _metrics = None
