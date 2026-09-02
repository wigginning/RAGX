"""Prometheus metrics tests (10-observability.md §10.2)."""

from __future__ import annotations

import pytest
from prometheus_client import CollectorRegistry

from ragx.observability.metrics import Metrics


@pytest.fixture()
def metrics() -> Metrics:
    return Metrics(CollectorRegistry())


class TestMetricCatalogue:
    def test_all_section2_metrics_registered(self, metrics: Metrics) -> None:
        """Every §10.2 metric appears in the Prometheus exposition output.

        We increment/set each metric once so that it produces at least one
        sample, then check the rendered text for the metric name.
        """
        # Increment / set every metric so it appears in the rendered output.
        metrics.record_query("kb", "standard", "ok", 0.1)
        metrics.record_stage_latency("kb", "standard", "retrieve", 0.05)
        metrics.record_llm_call("generate", "model_a", "default", 100, 0.01)
        metrics.cache_hit_ratio.labels(kb="kb").set(0.5)
        metrics.ingest_task_total.labels(kb="kb", status="done").inc()
        metrics.ingest_duration.labels(stage="parse").observe(0.1)
        metrics.recall_empty.labels(kb="kb").inc()
        metrics.llm_retry.labels(role="generate", model="model_a").inc()
        metrics.record_circuit("model_a", "closed")
        metrics.record_agentic_outcome("kb", "ok")
        metrics.budget_exceeded.labels(kb="kb").inc()
        metrics.ingest_parse.labels(parser="text", result="ok").inc()
        metrics.context_overflow.labels(kb="kb").inc()
        metrics.kg_extraction_skipped.labels(kb="kb").inc()
        metrics.cache_invalidation.labels(kb="kb", reason="delete").inc()
        metrics.agentic_degrade.labels(reason="7001").inc()
        metrics.agentic_budget_used.set(50000)

        output = metrics.render().decode()
        expected_names = [
            "ragx_query_total",
            "ragx_query_latency_seconds",
            "ragx_llm_tokens_total",
            "ragx_llm_cost_usd_total",
            "ragx_cache_hit_ratio",
            "ragx_ingest_task_total",
            "ragx_ingest_task_duration_seconds",
            "ragx_retrieval_recall_empty_total",
            "ragx_llm_retry_total",
            "ragx_circuit_state",
            "ragx_agentic_task_total",
            "ragx_budget_exceeded_total",
            "ragx_ingest_parse_total",
            "ragx_context_overflow_chunks",
            "ragx_kg_extraction_skipped_total",
            "ragx_cache_invalidation_total",
        ]
        for name in expected_names:
            assert name in output, f"metric {name} not in rendered output"

    def test_record_query(self, metrics: Metrics) -> None:
        metrics.record_query("kb_1", "standard", "ok", 0.5)
        output = metrics.render().decode()
        assert 'ragx_query_total{kb="kb_1",mode="standard",status="ok"}' in output
        assert "ragx_query_latency_seconds" in output

    def test_record_llm_call(self, metrics: Metrics) -> None:
        metrics.record_llm_call("generate", "deepseek-v3", "default", 1530, 0.0042)
        output = metrics.render().decode()
        assert (
            'ragx_llm_tokens_total{model="deepseek-v3",role="generate",tenant="default"} 1530'
            in output
        )
        assert "0.0042" in output

    def test_record_circuit_state(self, metrics: Metrics) -> None:
        metrics.record_circuit("model_a", "open")
        output = metrics.render().decode()
        assert 'ragx_circuit_state{model="model_a"} 1.0' in output

    def test_record_agentic_outcome(self, metrics: Metrics) -> None:
        metrics.record_agentic_outcome("kb_1", "ok")
        metrics.record_agentic_outcome("kb_1", "all_failed")
        output = metrics.render().decode()
        assert 'ragx_agentic_task_total{kb="kb_1",outcome="ok"}' in output
        assert 'ragx_agentic_task_total{kb="kb_1",outcome="all_failed"}' in output

    def test_stage_latency(self, metrics: Metrics) -> None:
        metrics.record_stage_latency("kb_1", "standard", "rerank", 0.05)
        output = metrics.render().decode()
        assert "ragx_query_latency_seconds" in output
        assert 'stage="rerank"' in output

    def test_budget_exceeded(self, metrics: Metrics) -> None:
        metrics.budget_exceeded.labels(kb="kb_1").inc()
        output = metrics.render().decode()
        assert 'ragx_budget_exceeded_total{kb="kb_1"}' in output

    def test_kg_extraction_skipped(self, metrics: Metrics) -> None:
        metrics.kg_extraction_skipped.labels(kb="kb_1").inc()
        output = metrics.render().decode()
        assert 'ragx_kg_extraction_skipped_total{kb="kb_1"}' in output

    def test_cache_invalidation(self, metrics: Metrics) -> None:
        metrics.cache_invalidation.labels(kb="kb_1", reason="chunk_edit").inc()
        output = metrics.render().decode()
        assert 'ragx_cache_invalidation_total{kb="kb_1",reason="chunk_edit"}' in output
