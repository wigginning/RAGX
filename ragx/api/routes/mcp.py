"""MCP SSE transport routes (09-api.md §9.7.2).

Exposes:
* ``GET /v1/mcp/sse`` — open an SSE stream; emits a ``sessionId`` event, then
  JSON-RPC responses for messages delivered via ``POST /v1/mcp/messages``.
* ``POST /v1/mcp/messages`` — deliver a JSON-RPC request for an open session.

Registered by ``create_app`` only when ``Settings.mcp.enabled`` is true. The
server for each session is scoped to the connecting API key's kb ACL, so an
MCP client can only touch the kbs its key is authorised for (tenant
isolation, consistent with the REST API).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ragx.mcp.server import build_mcp_server
from ragx.mcp.transport import SSETransport

router = APIRouter()


def _session_registry(app: Any) -> dict[str, SSETransport]:
    registry = getattr(app.state, "mcp_sessions", None)
    if registry is None:
        registry = {}
        app.state.mcp_sessions = registry
    return registry


def _kb_acl_of(request: Request) -> list[str] | None:
    auth = getattr(request.state, "auth", None)
    if auth is None:
        return None
    acl = getattr(auth, "kb_acl", None)
    if acl in (None, [], ()):
        return None
    return acl


@router.get("/mcp/sse")
async def mcp_sse(request: Request):
    app = request.app
    kb_acl = _kb_acl_of(request)
    server = build_mcp_server(app, kb_acl=kb_acl)
    transport = SSETransport(server)
    sid, _ = transport.new_session()
    _session_registry(app)[sid] = transport
    settings = getattr(app.state, "settings", None)
    idle_timeout = settings.mcp.sse_idle_timeout if settings is not None else None

    async def _stream():
        async for chunk in transport.sse_stream(sid, request, idle_timeout=idle_timeout):
            yield chunk

    return StreamingResponse(_stream(), media_type="text/event-stream")


@router.post("/mcp/messages")
async def mcp_messages(request: Request):
    app = request.app
    sid = request.query_params.get("sessionId") or request.query_params.get("session_id")
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not sid:
        sid = body.get("sessionId") or body.get("session_id")
    transport = _session_registry(app).get(sid)
    if transport is None:
        return JSONResponse({"error": "unknown or expired session"}, status_code=404)
    try:
        await transport.deliver_message(sid, body)
    except Exception as exc:  # pragma: no cover - defensive
        return JSONResponse({"error": str(exc)}, status_code=400)
    return JSONResponse({"status": "accepted"}, status_code=202)
