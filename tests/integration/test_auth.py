"""Auth integration tests (RX-API-02 DoD).

Covers: missing key -> 401 (1002), kb outside key ACL -> 403 (1003), rate
limit -> 429 (1004) with Retry-After, and the optional HS256 JWT path
(09-api.md §9.2.2).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from datetime import UTC

import pytest
from fastapi.testclient import TestClient

from ragx.api.app import create_app
from ragx.api.middleware.auth import verify_jwt
from ragx.core.exceptions import AuthError
from ragx.core.settings import SecurityConfig, Settings

_KEY = "sk-test-0001"


def _settings(*, rate_limit_rps: float = 100.0, burst: int = 1000) -> Settings:
    return Settings(
        security=SecurityConfig(
            api_keys={
                "k1": {
                    "key": _KEY,
                    "kb_acl": ["kb_a"],
                    "tenant_id": "t1",
                    "enabled": True,
                }
            },
            rate_limit_rps=rate_limit_rps,
            rate_limit_burst=burst,
        )
    )


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _jwt(secret: str, *, kb_acl: list[str], tenant: str = "t1") -> str:
    header = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload = _b64url(json.dumps({
        "sub": "u1", "tenant": tenant, "kb_acl": kb_acl,
        "exp": time.time() + 60,
    }).encode())
    signing = f"{header}.{payload}".encode()
    sig = _b64url(hmac.new(secret.encode(), signing, hashlib.sha256).digest())
    return f"{header}.{payload}.{sig}"


@pytest.fixture
def client():
    app = create_app(settings=_settings())
    with TestClient(app) as c:
        yield c


def _search(c: TestClient, kb_id: str, *, key: str | None = None):
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    return c.post("/v1/search", json={"kb_id": kb_id, "query": "向量检索"}, headers=headers)


def test_missing_key_returns_401(client) -> None:
    resp = _search(client, "kb_a")
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == 1002


def test_invalid_key_returns_401(client) -> None:
    resp = _search(client, "kb_a", key="sk-not-a-key")
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == 1002


def test_authorized_kb_passes(client) -> None:
    # kb_a is in the key's ACL; an empty store returns 200 with no results.
    resp = _search(client, "kb_a", key=_KEY)
    assert resp.status_code == 200
    assert resp.json()["results"] == []


def test_kb_outside_acl_returns_403(client) -> None:
    resp = _search(client, "kb_b", key=_KEY)
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == 1003


def test_rate_limit_returns_429_with_retry_after() -> None:
    app = create_app(settings=_settings(rate_limit_rps=0.5, burst=1))
    with TestClient(app) as c:
        first = _search(c, "kb_a", key=_KEY)
        second = _search(c, "kb_a", key=_KEY)
    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["error"]["code"] == 1004
    assert "Retry-After" in second.headers


def test_jwt_auth_path() -> None:
    secret = "test-secret"
    token = _jwt(secret, kb_acl=["kb_a"])
    claims = verify_jwt(token, secret)
    assert claims["tenant"] == "t1"
    assert claims["kb_acl"] == ["kb_a"]

    # a tampered signature must be rejected
    header, payload, _ = token.split(".")
    bad_sig = _b64url(hmac.new(b"wrong", f"{header}.{payload}".encode(),
                               hashlib.sha256).digest())
    with pytest.raises(AuthError) as ei:
        verify_jwt(f"{header}.{payload}.{bad_sig}", secret)
    assert ei.value.code == 1002


def test_jwt_through_middleware() -> None:
    secret = "test-secret"
    app = create_app(settings=Settings(
        security=SecurityConfig(jwt_enabled=True, jwt_secret=secret)
    ))
    token = _jwt(secret, kb_acl=["kb_a"])
    with TestClient(app) as c:
        resp = c.post(
            "/v1/search",
            json={"kb_id": "kb_a", "query": "向量检索"},
            headers={"Authorization": f"Bearer {token}"},
        )
    assert resp.status_code == 200


def _seed_chunk(client, *, kb_id: str, chunk_id: str = "cx1", doc_id: str = "dx1") -> None:
    import asyncio

    from ragx.core.models import Chunk
    chunk = Chunk(chunk_id=chunk_id, doc_id=doc_id, kb_id=kb_id,
                  text="secret-content", token_count=1, atom_ids=[])
    asyncio.run(client.app.state.db.save_chunks([chunk]))


def _seed_task(client, *, kb_id: str, task_id: str = "tx1", doc_id: str = "dx1") -> None:
    import asyncio
    from datetime import datetime

    from ragx.core.models import IngestTask
    task = IngestTask(
        task_id=task_id, doc_id=doc_id, kb_id=kb_id,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    asyncio.run(client.app.state.db.save_task(task))


def test_get_task_cross_tenant_returns_403(client) -> None:
    # key k1 is authorised only for kb_a; a task belonging to kb_b must 403.
    _seed_task(client, kb_id="kb_b")
    resp = client.get("/v1/tasks/tx1", headers={"Authorization": f"Bearer {_KEY}"})
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == 1003


def test_list_chunks_cross_tenant_returns_403(client) -> None:
    _seed_chunk(client, kb_id="kb_b")
    resp = client.get(
        "/v1/documents/dx1/chunks", headers={"Authorization": f"Bearer {_KEY}"}
    )
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == 1003


def test_put_chunk_cross_tenant_returns_403(client) -> None:
    _seed_chunk(client, kb_id="kb_b")
    resp = client.put(
        "/v1/chunks/cx1",
        json={"text": "hacked", "version": 1, "metadata": {}},
        headers={"Authorization": f"Bearer {_KEY}"},
    )
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == 1003


def test_patch_chunk_cross_tenant_returns_403(client) -> None:
    _seed_chunk(client, kb_id="kb_b")
    resp = client.patch(
        "/v1/chunks/cx1",
        json={"metadata": {"x": 1}, "version": 1},
        headers={"Authorization": f"Bearer {_KEY}"},
    )
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == 1003
