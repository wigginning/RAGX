"""Audit log tests (RX-API-03)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ragx.api.app import create_app
from ragx.api.middleware import InMemoryAuditStore
from ragx.core.settings import SecurityConfig


@pytest.fixture
def client() -> TestClient:
    app = create_app(settings=SecurityConfig().__class__())
    return TestClient(app)


def _collect_audit(client: TestClient) -> list:
    r = client.get("/v1/audit")
    assert r.status_code == 200, r.text
    return r.json()["entries"]


def test_audit_middleware_records_search_request() -> None:
    app = create_app()
    store = InMemoryAuditStore()
    # Replace the default audit store (the app already has one).
    app.state.audit_store = store

    client = TestClient(app)
    # Hit a known route; the middleware must record the entry.
    r = client.get("/v1/health")
    assert r.status_code == 200

    entries = store.entries()
    assert any(e.path == "/v1/health" for e in entries)
    health_entry = next(e for e in entries if e.path == "/v1/health")
    assert health_entry.method == "GET"
    assert health_entry.resource == "ops"
    assert health_entry.action == "read"
    assert health_entry.status_code == 200


def test_audit_records_kb_id_from_query_string() -> None:
    app = create_app()
    store = InMemoryAuditStore()
    app.state.audit_store = store
    client = TestClient(app)
    # /v1/health is a no-dep GET; we add an arbitrary ?kb_id=… and verify
    # the middleware captures it.
    client.get("/v1/health?kb_id=kb-7")
    entries = store.entries()
    health_entries = [e for e in entries if e.path == "/v1/health"]
    assert health_entries, entries
    assert health_entries[0].details.get("kb_id") == "kb-7"


def test_audit_store_failure_does_not_break_response() -> None:
    class _BrokenStore:
        async def append(self, entry):  # noqa: ARG002
            raise RuntimeError("store down")

    app = create_app()
    app.state.audit_store = _BrokenStore()
    client = TestClient(app)
    r = client.get("/v1/health")
    assert r.status_code == 200  # response still served


def test_metadata_audit_store_persists_to_db() -> None:
    """Persistent store path: the metadata DB has the audit rows."""
    import asyncio

    from ragx.api.middleware import MetadataAuditStore
    from ragx.ingestion.store import MetadataStore

    db = MetadataStore(":memory:")

    async def go() -> None:
        await db.connect()
        store = MetadataAuditStore(db)
        from ragx.api.middleware.audit import AuditEntry

        await store.append(
            AuditEntry(
                timestamp="2026-01-01T00:00:00Z",
                tenant_id="default",
                key_id="anonymous",
                method="GET",
                path="/v1/health",
                status_code=200,
                trace_id="t1",
                action="read",
                resource="ops",
                details={"kb_id": "kb-1"},
            )
        )
        rows = await db.list_audit_entries(tenant_id="default")
        assert any(r["path"] == "/v1/health" for r in rows)
        await db.close()

    asyncio.run(go())


def test_classify_method_resource_matrix() -> None:
    from ragx.api.middleware.audit import _classify

    assert _classify("GET", "/v1/search") == ("read", "query")
    assert _classify("POST", "/v1/documents") == ("write", "document")
    assert _classify("PUT", "/v1/chunks/x") == ("write", "chunk")
    assert _classify("PATCH", "/v1/chunks/x") == ("write", "chunk")
    assert _classify("DELETE", "/v1/chunks/x") == ("delete", "chunk")
    assert _classify("GET", "/v1/chat/completions") == ("read", "chat")
    assert _classify("GET", "/v1/tasks/x") == ("read", "task")
    assert _classify("GET", "/v1/audit") == ("read", "ops")
    assert _classify("WEIRD", "/foo") == ("admin", "settings")
