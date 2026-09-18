"""MCP stdio transport tests (09-api.md §9.7.2).

Covers: the stdio handshake, JSON-RPC message framing, and the
SSE transport's session management.
"""

from __future__ import annotations

import asyncio
import io
import json
from unittest.mock import patch

import pytest

from ragx.mcp.protocol import JSONRPCRequest
from ragx.mcp.server import MCPServer
from ragx.mcp.transport import SSETransport, StdioTransport


class _MockQueryService:
    async def search(self, **kwargs):
        return {"results": [], "trace_id": "t1"}

    async def generate(self, **kwargs):
        return {"answer": "answer", "mode": "standard"}

    async def list_kbs(self):
        return [{"kb_id": "kb_1", "name": "KB1"}]


def _make_server() -> MCPServer:
    svc = _MockQueryService()
    return MCPServer(
        search_fn=svc.search, generate_fn=svc.generate, list_kbs_fn=svc.list_kbs
    )


class TestStdioTransport:
    async def test_initialize_via_stdio(self) -> None:
        """stdin→stdout: initialize handshake."""
        server = _make_server()
        transport = StdioTransport(server)

        # Simulate a JSON-RPC initialize request
        req = JSONRPCRequest(id=1, method="initialize", params={})
        data = req.model_dump_json()

        # Write to a fake stdin and capture stdout
        fake_stdin = io.StringIO(data + "\n")
        fake_stdout = io.StringIO()

        with patch("sys.stdin", fake_stdin), patch("sys.stdout", fake_stdout):
            # The transport reads one line and exits (EOF)
            await transport.run()

        output = fake_stdout.getvalue().strip()
        resp = json.loads(output)
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 1
        assert resp["result"]["serverInfo"]["name"] == "ragx-mcp"

    async def test_multiple_messages(self) -> None:
        """Two messages in sequence: initialize + tools/list."""
        server = _make_server()
        transport = StdioTransport(server)

        reqs = [
            JSONRPCRequest(id=1, method="initialize"),
            JSONRPCRequest(id=2, method="tools/list"),
        ]
        input_text = "\n".join(r.model_dump_json() for r in reqs) + "\n"

        fake_stdin = io.StringIO(input_text)
        fake_stdout = io.StringIO()

        with patch("sys.stdin", fake_stdin), patch("sys.stdout", fake_stdout):
            await transport.run()

        output_lines = fake_stdout.getvalue().strip().split("\n")
        assert len(output_lines) == 2
        resp1 = json.loads(output_lines[0])
        resp2 = json.loads(output_lines[1])
        assert resp1["id"] == 1
        assert resp2["id"] == 2
        assert "tools" in resp2["result"]


class TestSSETransport:
    def test_new_session(self) -> None:
        """new_session creates a session with a queue."""
        server = _make_server()
        transport = SSETransport(server)
        sid, queue = transport.new_session()
        assert len(sid) > 0
        assert isinstance(queue, asyncio.Queue)
        assert sid in transport._connections
        transport.close_session(sid)

    def test_close_session(self) -> None:
        """close_session removes the session."""
        server = _make_server()
        transport = SSETransport(server)
        sid, _ = transport.new_session()
        transport.close_session(sid)
        assert sid not in transport._connections

    async def test_deliver_message(self) -> None:
        """deliver_message handles a JSON-RPC request and enques the response."""
        server = _make_server()
        transport = SSETransport(server)
        sid, queue = transport.new_session()

        data = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        await transport.deliver_message(sid, data)

        assert not queue.empty()
        resp_text = await queue.get()
        resp = json.loads(resp_text)
        assert resp["result"]["protocolVersion"] is not None

    async def test_deliver_message_unknown_session(self) -> None:
        """deliver_message with unknown session raises ValueError."""
        server = _make_server()
        transport = SSETransport(server)
        with pytest.raises(ValueError):
            await transport.deliver_message("unknown", {})

    async def test_sse_stream_format(self) -> None:
        """sse_stream yields SSE-formatted events."""
        server = _make_server()
        transport = SSETransport(server)
        sid, queue = transport.new_session()

        # Pre-fill the queue with a response
        await queue.put('{"jsonrpc":"2.0","id":1,"result":{}}')

        stream = transport.sse_stream(sid)
        # stream is an async generator — iterate directly
        first = await stream.__anext__()
        assert "sessionId" in first
        assert sid in first
        second = await stream.__anext__()
        assert "data:" in second
