"""Span tree structure tests (OBS-01 DoD: in-memory exporter asserts span tree).

Covers 10-observability.md §10.1: the query-side span tree and the
attribute dictionary (§10.1.3).
"""

from __future__ import annotations

import pytest

from ragx.observability.tracing import InMemorySpanExporter, Tracer


@pytest.fixture()
def tracer() -> Tracer:
    """Fresh tracer with a dedicated in-memory exporter per test."""
    exporter = InMemorySpanExporter()
    return Tracer(exporter)


def _build_query_span_tree(tracer: Tracer, trace_id: str = "trace_001") -> None:
    """Reproduce the §10.1.1 query-side span tree."""
    with tracer.start_as_current_span(
        "ragx.query", trace_id=trace_id, attributes={"ragx.kb_id": "kb_1", "ragx.mode": "standard"}
    ) as root:
        root.set_attribute("trace_id", trace_id)
        with tracer.start_as_current_span("ragx.cache_lookup") as c:
            c.set_attribute("ragx.cache_hit", False)
        with tracer.start_as_current_span("ragx.retrieve.dense") as d:
            d.set_attributes({"ragx.top_k": 20, "ragx.returned": 8, "ragx.latency_ms": 12})
        with tracer.start_as_current_span("ragx.retrieve.bm25") as b:
            b.set_attribute("ragx.supported", True)
        with tracer.start_as_current_span("ragx.assemble") as a:
            a.set_attributes({"ragx.token_budget": 4096, "ragx.assembled_tokens": 1800})
        with tracer.start_as_current_span("ragx.llm.generate") as llm:
            llm.set_attributes({
                "ragx.role": "generate",
                "ragx.model": "deepseek-v3",
                "ragx.cost_usd": 0.0041,
                "ragx.retry_count": 0,
            })


class TestSpanTree:
    def test_root_has_no_parent(self, tracer: Tracer) -> None:
        with tracer.start_as_current_span("ragx.query", trace_id="t1"):
            pass
        root = tracer.exporter.find("ragx.query")
        assert root is not None
        assert root.parent_id is None

    def test_child_links_to_parent(self, tracer: Tracer) -> None:
        _build_query_span_tree(tracer, "t2")
        root = tracer.exporter.find("ragx.query")
        assert root is not None
        children = tracer.exporter.children(root.span_id)
        child_names = {c.name for c in children}
        assert "ragx.cache_lookup" in child_names
        assert "ragx.retrieve.dense" in child_names
        assert "ragx.retrieve.bm25" in child_names
        assert "ragx.assemble" in child_names
        assert "ragx.llm.generate" in child_names
        for child in children:
            assert child.parent_id == root.span_id

    def test_all_spans_share_trace_id(self, tracer: Tracer) -> None:
        _build_query_span_tree(tracer, "t3")
        for span in tracer.exporter.spans:
            assert span.trace_id == "t3"

    def test_span_attributes_dict(self, tracer: Tracer) -> None:
        _build_query_span_tree(tracer, "t4")
        dense = tracer.exporter.find("ragx.retrieve.dense")
        assert dense is not None
        assert dense.attributes["ragx.top_k"] == 20
        assert dense.attributes["ragx.returned"] == 8
        llm = tracer.exporter.find("ragx.llm.generate")
        assert llm is not None
        assert llm.attributes["ragx.role"] == "generate"
        assert llm.attributes["ragx.model"] == "deepseek-v3"
        assert llm.attributes["ragx.retry_count"] == 0

    def test_nested_tree_structure(self, tracer: Tracer) -> None:
        _build_query_span_tree(tracer, "t5")
        tree = tracer.exporter.tree()
        assert tree["name"] == "ragx.query"
        child_names = [c["name"] for c in tree["children"]]
        assert "ragx.llm.generate" in child_names
        # children of children are empty in this tree
        for c in tree["children"]:
            assert c["children"] == []

    def test_exception_marks_span_error(self, tracer: Tracer) -> None:
        with pytest.raises(ValueError, match="boom"):
            with tracer.start_as_current_span("ragx.query", trace_id="t6") as span:
                span.set_attribute("ragx.kb_id", "kb_1")
                raise ValueError("boom")
        root = tracer.exporter.find("ragx.query")
        assert root is not None
        assert root.status == "ERROR"
        exc_events = [e for e in root.events if e.name == "exception"]
        assert len(exc_events) == 1
        assert exc_events[0].attributes["exception.type"] == "ValueError"


class TestIngestSpanTree:
    """§10.1.2 ingestion-side span tree."""

    def test_ingest_span_tree(self, tracer: Tracer) -> None:
        with tracer.start_as_current_span(
            "ragx.ingest", trace_id="ing_1", attributes={"ragx.doc_id": "doc_1"}
        ) as root:
            for stage in ("ragx.ingest.parse", "ragx.ingest.process", "ragx.ingest.chunk", "ragx.ingest.embed"):
                with tracer.start_as_current_span(stage) as s:
                    s.set_attribute("ragx.doc_id", "doc_1")
        root = tracer.exporter.find("ragx.ingest")
        assert root is not None
        children = tracer.exporter.children(root.span_id)
        assert len(children) == 4
        for c in children:
            assert c.parent_id == root.span_id
            assert c.attributes.get("ragx.doc_id") == "doc_1"


class TestSpanLifecycle:
    def test_duration_recorded(self, tracer: Tracer) -> None:
        import time

        with tracer.start_as_current_span("ragx.query", trace_id="t7") as span:
            time.sleep(0.01)
        assert span.end_time is not None
        assert span.duration_ms >= 8.0  # allow scheduling jitter

    def test_disabled_tracer_no_export(self) -> None:
        exporter = InMemorySpanExporter()
        tracer = Tracer(exporter, enabled=False)
        with tracer.start_as_current_span("ragx.query", trace_id="t8"):
            pass
        assert exporter.spans == []

    def test_add_event(self, tracer: Tracer) -> None:
        with tracer.start_as_current_span("ragx.query", trace_id="t9") as span:
            span.add_event("checkpoint", {"stage": "rerank"})
        root = tracer.exporter.find("ragx.query")
        assert root is not None
        assert len(root.events) == 1
        assert root.events[0].name == "checkpoint"
        assert root.events[0].attributes["stage"] == "rerank"
