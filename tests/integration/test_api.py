"""API integration tests (RX-API-01 DoD).

Covers: OpenAPI generation, error-code mapping, health deep check, and an
end-to-end upload -> search flow.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ragx.api.app import create_app
from ragx.core.settings import KBConfig


@pytest.fixture
def client():
    app = create_app(kb_cfg=KBConfig())
    with TestClient(app) as c:
        yield c


def test_openapi_generated(client) -> None:
    spec = client.get("/openapi.json").json()
    assert spec["info"]["title"] == "RAGX"
    paths = spec["paths"]
    assert "/v1/search" in paths
    assert "/v1/documents" in paths
    assert "/v1/tasks/{task_id}" in paths
    assert "/v1/health" in paths
    assert "/v1/chat/completions" in paths


def test_health_shallow(client) -> None:
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_health_deep(client) -> None:
    resp = client.get("/v1/health?deep=true")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in ("ok", "degraded")
    assert "vector_store" in body["checks"]


def test_trace_id_header(client) -> None:
    resp = client.get("/v1/health")
    assert resp.headers.get("X-Trace-Id")
    assert len(resp.headers["X-Trace-Id"]) == 26


def test_task_not_found_maps_to_404(client) -> None:
    resp = client.get("/v1/tasks/task_nonexistent")
    assert resp.status_code == 404
    body = resp.json()
    assert body["error"]["code"] == 2003
    assert "trace_id" in body["error"]


def test_upload_document_and_search(client) -> None:
    content = "# 测试\n\nRAGX 是一个检索增强生成平台，使用向量检索。".encode()
    resp = client.post(
        "/v1/documents",
        files={"file": ("test.md", content, "text/markdown")},
        data={"kb_id": "kb_t", "metadata": "{}"},
    )
    assert resp.status_code == 202
    body = resp.json()
    assert body["doc_id"].startswith("doc_")
    assert body["task_id"].startswith("task_")

    # run the pipeline synchronously so search has data
    import asyncio


    async def _run():
        task = await client.app.state.db.get_task(body["task_id"])
        await client.app.state.pipeline.run(task)

    asyncio.run(_run())

    search = client.post(
        "/v1/search",
        json={"kb_id": "kb_t", "query": "向量检索", "top_k": 5},
    )
    assert search.status_code == 200
    results = search.json()["results"]
    assert results, "search must return the ingested chunk"


def test_metrics_endpoint_exposes_key_series(client) -> None:
    """OBS-02 DoD: GET /v1/metrics exposes the §10.2 catalogue."""
    resp = client.get("/v1/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]

    body = resp.text
    for name in (
        "ragx_query_total",
        "ragx_query_latency_seconds",
        "ragx_llm_tokens_total",
        "ragx_ingest_task_total",
        "ragx_ingest_task_duration_seconds",
        "ragx_retrieval_recall_empty_total",
    ):
        assert name in body, f"metric {name} missing from /v1/metrics"


def test_duplicate_upload_returns_2004(client) -> None:
    content = "# 重复\n\n相同内容。".encode()
    for _ in range(2):
        resp = client.post(
            "/v1/documents",
            files={"file": ("dup.md", content, "text/markdown")},
            data={"kb_id": "kb_t", "metadata": "{}"},
        )
    assert resp.status_code == 202
    assert resp.json()["duplicate"] is True
