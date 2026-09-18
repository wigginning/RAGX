"""MCP transports (09-api.md §9.7.2).

* **stdio** — JSON-RPC over stdin/stdout (for local Agent / CLI integration)
* **SSE**   — SSE long-connection for remote Agent / service integration
  (``GET /v1/mcp/sse`` + ``POST /v1/mcp/messages``)

The stdio transport reads newline-delimited JSON-RPC from stdin and
writes responses to stdout. The SSE transport is designed to be wired
into the FastAPI app.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any

from ragx.mcp.protocol import JSONRPCRequest, JSONRPCResponse
from ragx.mcp.server import MCPServer

logger = logging.getLogger("ragx.mcp.transport")


class StdioTransport:
    """stdio JSON-RPC transport (§9.7.2).

    Reads newline-delimited JSON-RPC requests from stdin and writes
    responses to stdout. Designed for local CLI / Agent integration.

    Usage::

        server = MCPServer(search_fn=..., generate_fn=..., list_kbs_fn=...)
        transport = StdioTransport(server)
        await transport.run()
    """

    def __init__(self, server: MCPServer) -> None:
        self.server = server

    async def run(self) -> None:
        """Main loop: read stdin → handle → write stdout."""
        loop = asyncio.get_running_loop()
        while True:
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line:
                # EOF — shutdown
                break
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                req = JSONRPCRequest.model_validate(data)
            except (json.JSONDecodeError, Exception) as exc:
                error_resp = JSONRPCResponse.failure(
                    None, -32700, f"Parse error: {exc}"
                )
                await self._write(error_resp)
                continue

            try:
                resp = await self.server.handle_request(req)
            except Exception as exc:
                resp = JSONRPCResponse.failure(
                    req.id, -32603, f"Internal error: {exc}"
                )

            await self._write(resp)

    async def _write(self, resp: JSONRPCResponse) -> None:
        """Write a JSON-RPC response to stdout."""
        loop = asyncio.get_running_loop()
        text = resp.model_dump_json(exclude_none=True)
        await loop.run_in_executor(None, lambda: (
            sys.stdout.write(text + "\n"),
            sys.stdout.flush(),
        ))


class SSETransport:
    """SSE transport (§9.7.2).

    Designed to be wired into the FastAPI app. Provides two endpoints:
    * ``GET /v1/mcp/sse`` — establish an SSE long-connection
    * ``POST /v1/mcp/messages`` — deliver a JSON-RPC request

    The SSE connection sends a ``sessionId`` event on connect, then
    JSON-RPC responses are pushed to the client via the SSE stream.
    """

    def __init__(self, server: MCPServer) -> None:
        self.server = server
        self._connections: dict[str, asyncio.Queue[str]] = {}

    def new_session(self) -> tuple[str, asyncio.Queue[str]]:
        """Create a new SSE session. Returns (session_id, event_queue)."""
        import uuid

        sid = uuid.uuid4().hex
        queue: asyncio.Queue[str] = asyncio.Queue()
        self._connections[sid] = queue
        return sid, queue

    def close_session(self, session_id: str) -> None:
        """Close an SSE session."""
        self._connections.pop(session_id, None)

    async def deliver_message(
        self, session_id: str, data: dict[str, Any]
    ) -> None:
        """Deliver a JSON-RPC request from POST /v1/mcp/messages.

        Handles the request and pushes the response to the SSE queue.
        """
        queue = self._connections.get(session_id)
        if queue is None:
            raise ValueError(f"Unknown session: {session_id}")

        try:
            req = JSONRPCRequest.model_validate(data)
        except Exception as exc:
            error_resp = JSONRPCResponse.failure(
                None, -32700, f"Parse error: {exc}"
            )
            await queue.put(error_resp.model_dump_json(exclude_none=True))
            return

        try:
            resp = await self.server.handle_request(req)
        except Exception as exc:
            resp = JSONRPCResponse.failure(
                req.id, -32603, f"Internal error: {exc}"
            )
        await queue.put(resp.model_dump_json(exclude_none=True))

    def sse_stream(self, session_id: str, request=None, idle_timeout: float | None = None):
        """Generator that yields SSE-formatted events for a session.

        After emitting the ``sessionId`` event it pushes any message delivered
        via :meth:`deliver_message`. The loop terminates when:

        * the client disconnects (best-effort, via ``request.is_disconnected``),
        * or the connection has been idle (no pushed message) for longer than
          ``idle_timeout`` seconds (``<= 0`` disables this safety net).

        The idle guard lets the server reap hung/dead SSE connections instead of
        blocking forever on an empty queue (which would also block test
        teardown under TestClient, where ``http.disconnect`` is not delivered).
        """
        queue = self._connections.get(session_id)
        if queue is None:
            return

        import time

        async def _iter():
            # Send the session ID event
            yield f"event: sessionId\ndata: {session_id}\n\n"
            last_activity = time.monotonic()
            try:
                while True:
                    if request is not None and await request.is_disconnected():
                        break
                    try:
                        data = await asyncio.wait_for(queue.get(), timeout=1.0)
                    except TimeoutError:
                        if idle_timeout is not None and idle_timeout > 0:
                            if (time.monotonic() - last_activity) >= idle_timeout:
                                break
                        continue
                    last_activity = time.monotonic()
                    yield f"data: {data}\n\n"
            finally:
                self.close_session(session_id)

        return _iter()
