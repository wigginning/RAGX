"""Observability layer (10-observability.md): OTel spans, Prometheus metrics,
RAG Trace collection, and the evaluation framework.

Design principles (§10.0):
1. Telemetry lives in the abstraction/SPI layer - plugin authors get spans for free.
2. Cost attribution: every LLM call records model/tokens/cost/role/tenant.
3. Replayable: a query's full retrieval path lands in RAG Trace (7-day TTL).
4. Regression baseline: eval set + baseline.json, nightly CI gate.
"""

from ragx.observability.metrics import Metrics, get_metrics
from ragx.observability.rag_trace import RAGTrace, RAGTraceStore, TraceLLMCall
from ragx.observability.tracing import InMemorySpanExporter, Span, Tracer, get_tracer

__all__ = [
    "InMemorySpanExporter",
    "Metrics",
    "RAGTrace",
    "RAGTraceStore",
    "Span",
    "TraceLLMCall",
    "Tracer",
    "get_metrics",
    "get_tracer",
]
