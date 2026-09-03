"""MCP integration tests (RX-v1.0 MCP 完善 DoD).

Verifies the MCP server is actually wired to the real RAGX backend (not the
default stubs): tools/list, ragx_search, ragx_generate and ragx_list_kbs all
return live data, tenant scoping works, and the SSE transport routes are
registered and functional.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest
from fastapi.testclient import TestClient

from ragx.api.app import create_app
from ragx.core.models import TokenUsage
from ragx.core.settings import MCPConfig, QueueConfig, SecurityConfig, Settings
from ragx.mcp.protocol import JSONRPCRequest
from ragx.mcp.server import build_mcp_server
from ragx.mcp.transport import SSETransport

SAMPLE = """# RAGX 冒烟测试

RAGX 是一个分层插件化的检索增强生成平台，使用向量检索与知识图谱提升问答质量。
"""

E2E_KEY = "sk-e2e-0001"
E2E_KB = "kb_e2e"
AUTH = {"Authorization": f"Bearer {E2E_KEY}"}


class MockLLM:
    async def chat(self, req):
        class R:
            pass

        r = R()
        r.text = "RAGX 是分层检索增强生成平台。"
        r.model = "e2e-mock"
        r.usage = TokenUsage(prompt_tokens=12, completion_tokens=6, total=18)
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


def _make_client():
    settings = Settings(
        queue=QueueConfig(auto_consume=True),
        mcp=MCPConfig(enabled=True, transport="sse", sse_idle_timeout=1.0),
        security=SecurityConfig(
            api_keys={"e2e": {
                "key": E2E_KEY, "kb_acl": [E2E_KB], "tenant_id": "t_e2e", "enabled": True,
            }},
            rate_limit_rps=1000.0, rate_limit_burst=10000,
        ),
    )
    return TestClient(create_app(settings=settings, llm=MockLLM()))


def _seed_kb_and_ingest(client) -> None:
    asyncio.run(client.app.state.db.save_kb(E2E_KB, {"name": "E2E KB"}))
    resp = client.post(
        "/v1/documents",
        files={"file": ("smoke.md", SAMPLE.encode("utf-8"), "text/markdown")},
        data={"kb_id": E2E_KB, "metadata": "{}"},
        headers=AUTH,
    )
    assert resp.status_code == 202, resp.text
    task_id = resp.json()["task_id"]
    deadline = time.time() + 20.0
    while time.time() < deadline:
        t = client.get(f"/v1/tasks/{task_id}", headers=AUTH)
        if t.json()["status"] in ("done", "failed"):
            break
        time.sleep(0.3)
    assert t.json()["status"] == "done"


@pytest.fixture
def client():
    with _make_client() as c:
        _seed_kb_and_ingest(c)
        yield c


def _rpc(method, params=None, id=1):
    return JSONRPCRequest(jsonrpc="2.0", id=id, method=method, params=params or {})


def _call(server, rpc):
    """handle_request is async; run it to completion in a fresh loop."""
    return asyncio.run(server.handle_request(rpc))


def _payload(rpc_resp):
    """Extract the parsed JSON dict from a tool-call JSON-RPC result."""
    return json.loads(rpc_resp.result["content"][0]["text"])


def test_mcp_tools_list_and_search_generate(client) -> None:
    server = build_mcp_server(client.app)
    init = _call(server, _rpc("initialize"))
    assert init.result["serverInfo"]["name"] == "ragx-mcp"

    tl = _call(server, _rpc("tools/list"))
    names = {t["name"] for t in tl.result["tools"]}
    assert names == {"ragx_search", "ragx_generate", "ragx_list_kbs"}

    lk = _call(server, _rpc("tools/call", {"name": "ragx_list_kbs", "arguments": {}}))
    kb_ids = {k["kb_id"] for k in _payload(lk)}
    assert E2E_KB in kb_ids

    sr = _call(server, _rpc("tools/call", {"name": "ragx_search", "arguments": {
        "kb_id": E2E_KB, "query": "向量检索", "top_k": 5}}))
    assert _payload(sr)["results"], "ragx_search must return the ingested chunk"

    gr = _call(server, _rpc("tools/call", {"name": "ragx_generate", "arguments": {
        "kb_id": E2E_KB, "query": "什么是 RAGX?", "mode": "auto"}}))
    gp = _payload(gr)
    assert gp["answer"]
    assert gp["citations"], "ragx_generate must carry citations"


def test_mcp_kb_acl_scoping(client) -> None:
    server = build_mcp_server(client.app, kb_acl=["some_other_kb"])
    _call(server, _rpc("initialize"))

    sr = _call(server, _rpc("tools/call", {"name": "ragx_search", "arguments": {
        "kb_id": E2E_KB, "query": "向量检索"}}))
    assert sr.result["is_error"] is True

    lk = _call(server, _rpc("tools/call", {"name": "ragx_list_kbs", "arguments": {}}))
    kbs = _payload(lk)
    assert all(k["kb_id"] in {"some_other_kb"} for k in kbs)


def test_mcp_sse_routes_registered_and_handshake(client) -> None:
    with client.stream("GET", "/v1/mcp/sse", headers=AUTH) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        lines = []
        for line in resp.iter_lines():
            lines.append(line)
            if "sessionId" in line:
                break
        assert any("sessionId" in ln for ln in lines), "SSE stream must emit a sessionId event"

    r = client.post(
        "/v1/mcp/messages?sessionId=nope",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        headers=AUTH,
    )
    assert r.status_code == 404


def test_mcp_sse_transport_roundtrip(client) -> None:
    server = build_mcp_server(client.app)
    _call(server, _rpc("initialize"))
    transport = SSETransport(server)
    sid, queue = transport.new_session()
    asyncio.run(transport.deliver_message(sid, {
        "jsonrpc": "2.0", "id": 7, "method": "tools/call",
        "params": {"name": "ragx_list_kbs", "arguments": {}},
    }))
    item = queue.get_nowait()
    # NB: substring checks on the raw text are wrong — `is_error` contains
    # "error". Parse and assert on the top-level JSON-RPC fields instead.
    payload = json.loads(item)
    assert "result" in payload and "error" not in payload, f"unexpected SSE payload: {item}"
