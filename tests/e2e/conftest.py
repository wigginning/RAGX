"""Shared fixtures for the in-process E2E suite (RX-v1.0 E2E).

These tests boot the real RAGX app via ``create_app`` with an in-memory
SQLite metadata store, the lite plugin set (in-memory embedder + vector
store), and a no-network ``MockLLM``. Because ``queue.auto_consume`` is
enabled, an upload is ingested by the in-process worker with no external
dependencies — so the full journey (upload -> ingest -> search -> chat
with citations) runs in CI without Docker Compose.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ragx.api.app import create_app
from ragx.core.models import TokenUsage
from ragx.core.settings import QueueConfig, SecurityConfig, Settings


class _Capabilities:
    supports_json_mode = True
    supports_stream = False
    name = "e2e-mock-llm"


class MockLLM:
    """Minimal LLMProvider for in-process E2E (no network, deterministic)."""

    def __init__(self, text: str = "RAGX 是一个分层检索增强生成平台。", model: str = "e2e-mock") -> None:
        self._text = text
        self._model = model
        self.capabilities = _Capabilities()
        self.name = model
        self.call_count = 0

    async def chat(self, req):
        self.call_count += 1

        class _Resp:
            pass

        r = _Resp()
        r.text = self._text
        r.model = self._model
        r.usage = TokenUsage(prompt_tokens=12, completion_tokens=6, total=18)
        r.cost_usd = 0.0
        return r

    async def chat_stream(self, req):  # pragma: no cover - not exercised (stream:false)
        return None

    async def structured(self, req, schema):  # pragma: no cover - not exercised
        raise NotImplementedError("structured() not used in e2e mock")

    async def startup(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None


# Single authorised key for the journey; the cross-tenant cases seed objects
# under a different kb ("kb_other") that this key's ACL does not include.
E2E_KEY = "sk-e2e-0001"
E2E_KB = "kb_e2e"
E2E_TENANT = "t_e2e"


def _security(*, upload_quota: int = 0, token_quota: int = 0) -> SecurityConfig:
    return SecurityConfig(
        api_keys={
            "e2e": {
                "key": E2E_KEY,
                "kb_acl": [E2E_KB],
                "tenant_id": E2E_TENANT,
                "enabled": True,
            }
        },
        # Generous per-request limits so the E2E flow is never rate-limited.
        rate_limit_rps=1000.0,
        rate_limit_burst=10000,
        monthly_upload_quota_bytes=upload_quota,
        monthly_token_quota=token_quota,
    )


def _make_client(*, upload_quota: int = 0, token_quota: int = 0) -> TestClient:
    settings = Settings(
        queue=QueueConfig(auto_consume=True),
        security=_security(upload_quota=upload_quota, token_quota=token_quota),
    )
    app = create_app(settings=settings, llm=MockLLM())
    return TestClient(app)


@pytest.fixture
def client():
    with _make_client() as c:
        yield c


@pytest.fixture
def quota_client():
    # Tiny upload cap (10 bytes) so any real document upload is rejected (1004).
    with _make_client(upload_quota=10) as c:
        yield c
