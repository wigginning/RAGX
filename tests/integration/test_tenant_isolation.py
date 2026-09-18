"""Cross-tenant isolation on read endpoints (v1.0 security review).

Regression tests for the leaks found in the security review:

* ``GET /v1/traces?kb_id=`` returned another tenant's traces (raw query +
  retrieved context) to any key with a non-empty ACL.
* ``GET /v1/traces/{id}`` returned any trace by id with no ownership check.
* ``GET /v1/audit?tenant_id=`` let any key read any tenant's audit trail.
* The 403 error body echoed the key's full ``kb_acl`` (kb enumeration).

Two keys in one app: ``a`` (tenant ``t_a``, kb ``kb_a``) and ``b`` (tenant
``t_b``, kb ``kb_b``). ``b`` must never see ``a``'s data through these
endpoints, and the 403 body must not enumerate ``a``'s allowed kbs.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest
from fastapi.testclient import TestClient

from ragx.api.app import create_app
from ragx.core.models import TokenUsage
from ragx.core.settings import QueueConfig, SecurityConfig, Settings

KEY_A = "sk-aaa-0001"
KEY_B = "sk-bbb-0002"
KB_A = "kb_a"
KB_B = "kb_b"

DOC = """# 机密文档 A

Alpha 项目的并购底价是 4.2 亿人民币，仅限 A 租户知悉。
"""

AUTH_A = {"Authorization": f"Bearer {KEY_A}"}
AUTH_B = {"Authorization": f"Bearer {KEY_B}"}


class MockLLM:
    async def chat(self, req):
        class R:
            pass

        r = R()
        r.text = "机密回答。"
        r.model = "mock"
        r.usage = TokenUsage(prompt_tokens=1, completion_tokens=1, total=2)
        r.cost_usd = 0.0
        return r

    async def chat_stream(self, req):  # pragma: no cover
        return None

    async def structured(self, req, schema):  # pragma: no cover
        raise NotImplementedError

    async def startup(self):
        return None

    async def shutdown(self):
        return None


def _make_client() -> TestClient:
    settings = Settings(
        queue=QueueConfig(auto_consume=True),
        security=SecurityConfig(
            api_keys={
                "a": {"key": KEY_A, "kb_acl": [KB_A], "tenant_id": "t_a", "enabled": True},
                "b": {"key": KEY_B, "kb_acl": [KB_B], "tenant_id": "t_b", "enabled": True},
            },
            rate_limit_rps=1000.0,
            rate_limit_burst=10000,
        ),
    )
    return TestClient(create_app(settings=settings, llm=MockLLM()))


@pytest.fixture
def client():
    with _make_client() as c:
        yield c


def _seed_a(client) -> str:
    """Tenant A ingests a secret doc and queries it.

    Returns the trace_id produced by A's query (must be unreadable to B).
    """
    asyncio.run(client.app.state.db.save_kb(KB_A, {"name": KB_A}))
    r = client.post(
        "/v1/documents",
        files={"file": ("d.md", DOC.encode("utf-8"), "text/markdown")},
        data={"kb_id": KB_A, "metadata": "{}"},
        headers=AUTH_A,
    )
    assert r.status_code == 202, r.text
    tid = r.json()["task_id"]
    for _ in range(60):
        t = client.get(f"/v1/tasks/{tid}", headers=AUTH_A)
        if t.json()["status"] in ("done", "failed"):
            break
        time.sleep(0.3)
    assert t.json()["status"] == "done"

    chat = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "并购底价是多少?"}],
            "stream": False,
            "ragx": {"kb_id": KB_A, "mode": "auto"},
        },
        headers=AUTH_A,
    )
    assert chat.status_code == 200, chat.text
    traces = client.get(f"/v1/traces?kb_id={KB_A}", headers=AUTH_A).json()["traces"]
    assert traces, "A's query must produce a trace"
    return traces[0]["trace_id"]


def _assert_1003(resp) -> None:
    assert resp.status_code == 403, resp.text
    assert resp.json()["error"]["code"] == 1003


def test_list_traces_cross_tenant_returns_403(client) -> None:
    _seed_a(client)
    _assert_1003(client.get(f"/v1/traces?kb_id={KB_A}", headers=AUTH_B))


def test_get_trace_cross_tenant_returns_403(client) -> None:
    trace_id = _seed_a(client)
    _assert_1003(client.get(f"/v1/traces/{trace_id}", headers=AUTH_B))


def test_audit_cross_tenant_returns_403(client) -> None:
    _seed_a(client)
    _assert_1003(client.get("/v1/audit?tenant_id=t_a", headers=AUTH_B))


def test_own_tenant_reads_still_work(client) -> None:
    """Guard against over-blocking: A reads its own traces and audit."""
    trace_id = _seed_a(client)

    own_traces = client.get(f"/v1/traces?kb_id={KB_A}", headers=AUTH_A)
    assert own_traces.status_code == 200
    assert own_traces.json()["traces"]

    own_trace = client.get(f"/v1/traces/{trace_id}", headers=AUTH_A)
    assert own_trace.status_code == 200

    own_audit = client.get("/v1/audit", headers=AUTH_A)  # no tenant_id -> own
    assert own_audit.status_code == 200
    assert all(e["tenant_id"] == "t_a" for e in own_audit.json()["entries"])

    own_audit_explicit = client.get("/v1/audit?tenant_id=t_a", headers=AUTH_A)
    assert own_audit_explicit.status_code == 200


def test_403_body_does_not_echo_kb_acl(client) -> None:
    """The 403 payload must not enumerate the key's allowed kbs."""
    _seed_a(client)
    resp = client.get(f"/v1/traces?kb_id={KB_A}", headers=AUTH_B)
    _assert_1003(resp)
    body = json.dumps(resp.json(), ensure_ascii=False)
    assert "kb_acl" not in body, "403 must not leak the caller's kb allow-list"
    # The requested kb_id appears legitimately; B's own kb (kb_b) must not.
    assert "kb_b" not in body


def test_rate_limit_is_per_key(client) -> None:
    """One key exhausting its token bucket must not 429 other tenants.

    Regression: RateLimitMiddleware used to run BEFORE AuthMiddleware, so
    request.state.auth was never resolved and every key shared a single
    "anonymous" bucket — tenant A could rate-limit the whole deployment.
    """
    settings = Settings(
        queue=QueueConfig(auto_consume=True),
        security=SecurityConfig(
            api_keys={
                "a": {"key": KEY_A, "kb_acl": [], "tenant_id": "t_a", "enabled": True},
                "b": {"key": KEY_B, "kb_acl": [], "tenant_id": "t_b", "enabled": True},
            },
            rate_limit_rps=5.0,
            rate_limit_burst=5,
        ),
    )
    with TestClient(create_app(settings=settings, llm=MockLLM())) as c:
        h_a = {"Authorization": f"Bearer {KEY_A}"}
        h_b = {"Authorization": f"Bearer {KEY_B}"}
        # A exhausts its own bucket (rps=5, burst=5 -> 6th request is 429)
        codes_a = [c.get("/v1/health", headers=h_a).status_code for _ in range(6)]
        assert codes_a == [200, 200, 200, 200, 200, 429], codes_a
        # B must be untouched
        assert c.get("/v1/health", headers=h_b).status_code == 200
        # A stays limited (its bucket is still empty)
        assert c.get("/v1/health", headers=h_a).status_code == 429
