"""End-to-end journey tests for RAGX v1.0 (RX-v1.0 E2E DoD).

Boots the real app in-process (see conftest) and exercises the complete
user path plus the three API-03 tenant-safety guarantees:

* full journey:  health -> upload -> in-process ingest -> search -> chat
  with citations (citations are assembled from retrieved chunks, not the LLM).
* audit log is recorded for every request (audit middleware, outermost).
* quota gate rejects oversized uploads with 429 / 1004.
* cross-tenant isolation (API-03 fix): a key authorised only for one kb
  cannot read/mutate another tenant's objects by id (403 / 1003).
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime

from ragx.core.models import Chunk, IngestTask

# Mirror conftest.E2E_KEY / E2E_KB (kept in sync; the authorised key's ACL
# covers only kb_e2e, so "kb_other" objects must be rejected cross-tenant).
E2E_KEY = "sk-e2e-0001"
E2E_KB = "kb_e2e"

SAMPLE = """# RAGX 冒烟测试

RAGX 是一个分层插件化的检索增强生成平台，使用向量检索与知识图谱提升问答质量。

## 背景

它通过 dense + BM25 混合检索、RRF 融合与引用标注，输出带 [n] 引用的答案。
"""

AUTH = {"Authorization": f"Bearer {E2E_KEY}"}


def _wait_done(client, task_id: str, timeout_s: float = 20.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = client.get(f"/v1/tasks/{task_id}", headers=AUTH)
        assert resp.status_code == 200, resp.text
        status = resp.json()["status"]
        if status == "done":
            return
        if status == "failed":
            raise AssertionError(f"task {task_id} failed: {resp.json().get('error')}")
        time.sleep(0.3)
    raise AssertionError(f"task {task_id} did not reach done within {timeout_s}s")


def test_health(client) -> None:
    # Auth is enabled in this suite (api_keys configured), so health needs a key.
    resp = client.get("/v1/health", headers=AUTH)
    assert resp.status_code == 200


def test_full_journey_upload_search_chat(client) -> None:
    # 1) upload (202 Accepted)
    resp = client.post(
        "/v1/documents",
        files={"file": ("smoke.md", SAMPLE.encode("utf-8"), "text/markdown")},
        data={"kb_id": E2E_KB, "metadata": "{}"},
        headers=AUTH,
    )
    assert resp.status_code == 202, resp.text
    body = resp.json()
    doc_id = body["doc_id"]
    task_id = body["task_id"]
    assert doc_id and task_id

    # 2) in-process worker finishes ingest
    _wait_done(client, task_id)

    # 3) search returns the ingested chunk
    sresp = client.post(
        "/v1/search",
        json={"kb_id": E2E_KB, "query": "向量检索", "top_k": 5},
        headers=AUTH,
    )
    assert sresp.status_code == 200, sresp.text
    results = sresp.json().get("results") or []
    assert results, "search must return the ingested chunk"
    assert doc_id in {r.get("doc_id") for r in results}

    # 4) chat returns citations referencing the uploaded document
    cresp = client.post(
        "/v1/chat/completions",
        json={
            "model": "ragx-default",
            "stream": False,
            "messages": [{"role": "user", "content": "什么是 RAGX?"}],
            "ragx": {"kb_id": E2E_KB, "mode": "auto"},
        },
        headers=AUTH,
    )
    assert cresp.status_code == 200, cresp.text
    data = cresp.json()
    citations = (data.get("ragx") or {}).get("citations") or []
    assert citations, "chat answer must carry citations"
    assert doc_id in {c.get("doc_id") for c in citations}


def test_audit_log_recorded(client) -> None:
    client.get("/v1/health", headers=AUTH)
    resp = client.get("/v1/audit", headers=AUTH)
    assert resp.status_code == 200
    entries = resp.json().get("entries") or []
    assert len(entries) > 0, "audit middleware must record every request"


def test_upload_quota_exceeded_returns_429(quota_client) -> None:
    resp = quota_client.post(
        "/v1/documents",
        files={"file": ("smoke.md", SAMPLE.encode("utf-8"), "text/markdown")},
        data={"kb_id": E2E_KB, "metadata": "{}"},
        headers=AUTH,
    )
    assert resp.status_code == 429, resp.text
    assert resp.json()["error"]["code"] == 1004


# --- API-03 cross-tenant isolation, full middleware stack ----------------
def _seed_chunk(client, *, kb_id: str, chunk_id: str = "cx1", doc_id: str = "dx1") -> None:
    chunk = Chunk(
        chunk_id=chunk_id, doc_id=doc_id, kb_id=kb_id,
        text="secret-content", token_count=1, atom_ids=[],
    )
    asyncio.run(client.app.state.db.save_chunks([chunk]))


def _seed_task(client, *, kb_id: str, task_id: str = "tx1", doc_id: str = "dx1") -> None:
    task = IngestTask(
        task_id=task_id, doc_id=doc_id, kb_id=kb_id,
        created_at=datetime.now(UTC), updated_at=datetime.now(UTC),
    )
    asyncio.run(client.app.state.db.save_task(task))


def test_cross_tenant_task_returns_403(client) -> None:
    _seed_task(client, kb_id="kb_other")
    resp = client.get("/v1/tasks/tx1", headers=AUTH)
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == 1003


def test_cross_tenant_list_chunks_returns_403(client) -> None:
    _seed_chunk(client, kb_id="kb_other")
    resp = client.get("/v1/documents/dx1/chunks", headers=AUTH)
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == 1003


def test_cross_tenant_put_chunk_returns_403(client) -> None:
    _seed_chunk(client, kb_id="kb_other")
    resp = client.put(
        "/v1/chunks/cx1",
        json={"text": "hacked", "version": 1, "metadata": {}},
        headers=AUTH,
    )
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == 1003


def test_cross_tenant_patch_chunk_returns_403(client) -> None:
    _seed_chunk(client, kb_id="kb_other")
    resp = client.patch(
        "/v1/chunks/cx1",
        json={"metadata": {"x": 1}, "version": 1},
        headers=AUTH,
    )
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == 1003
