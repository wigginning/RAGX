"""MCP protocol and server tests (09-api.md §9.7).

Covers: JSON-RPC message types, tool registration, initialize handshake,
tools/list, tools/call dispatch, and error handling.
"""

from __future__ import annotations

import json

from ragx.mcp.protocol import (
    PROTOCOL_VERSION,
    ErrorCodes,
    JSONRPCRequest,
    JSONRPCResponse,
    MCPTool,
    ToolCallResult,
)
from ragx.mcp.server import MCPServer


class TestProtocol:
    def test_jsonrpc_request_parse(self) -> None:
        data = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        req = JSONRPCRequest.model_validate(data)
        assert req.method == "initialize"
        assert req.id == 1

    def test_jsonrpc_response_success(self) -> None:
        resp = JSONRPCResponse.ok(id=1, result={"ok": True})
        assert resp.result == {"ok": True}
        assert resp.error is None

    def test_jsonrpc_response_error(self) -> None:
        resp = JSONRPCResponse.failure(id=1, code=-32601, message="not found")
        assert resp.error is not None
        assert resp.error.code == -32601

    def test_tool_call_result_from_text(self) -> None:
        result = ToolCallResult.from_text("hello")
        assert result.content[0]["type"] == "text"
        assert result.content[0]["text"] == "hello"
        assert result.is_error is False

    def test_tool_call_result_from_json(self) -> None:
        result = ToolCallResult.from_json({"a": 1})
        assert result.content[0]["type"] == "text"
        data = json.loads(result.content[0]["text"])
        assert data == {"a": 1}

    def test_tool_call_result_error(self) -> None:
        result = ToolCallResult.from_text("boom", is_error=True)
        assert result.is_error is True

    def test_mcp_tool_schema(self) -> None:
        tool = MCPTool(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object", "properties": {"x": {"type": "integer"}}},
        )
        dumped = tool.model_dump()
        assert dumped["name"] == "test_tool"
        assert dumped["input_schema"]["type"] == "object"


class _MockQueryService:
    """Mock query/search functions for the MCP server."""

    async def search(self, **kwargs):
        return {
            "results": [
                {"chunk_id": "chk_1", "text": "result text", "score": 0.9},
            ],
            "trace_id": "trace_mcp",
        }

    async def generate(self, **kwargs):
        return {
            "answer": "This is the answer.",
            "mode": "standard",
            "citations": [{"chunk_id": "chk_1"}],
        }

    async def list_kbs(self):
        return [
            {"kb_id": "kb_1", "name": "Test KB"},
            {"kb_id": "kb_2", "name": "Production KB"},
        ]


def _make_server() -> MCPServer:
    svc = _MockQueryService()
    return MCPServer(
        search_fn=svc.search,
        generate_fn=svc.generate,
        list_kbs_fn=svc.list_kbs,
    )


class TestMCPServer:
    async def test_initialize_handshake(self) -> None:
        """MCP initialize → protocol version + capabilities + server info."""
        server = _make_server()
        req = JSONRPCRequest(id=1, method="initialize", params={})
        resp = await server.handle_request(req)

        assert resp.error is None
        assert resp.result["protocolVersion"] == PROTOCOL_VERSION
        assert "tools" in resp.result["capabilities"]
        assert resp.result["serverInfo"]["name"] == "ragx-mcp"

    async def test_tools_list(self) -> None:
        """tools/list returns all three tools."""
        server = _make_server()
        # Initialize first
        await server.handle_request(JSONRPCRequest(id=1, method="initialize"))

        resp = await server.handle_request(
            JSONRPCRequest(id=2, method="tools/list")
        )
        assert resp.error is None
        tools = resp.result["tools"]
        assert len(tools) == 3
        names = {t["name"] for t in tools}
        assert names == {"ragx_search", "ragx_generate", "ragx_list_kbs"}

    async def test_tools_list_before_init_fails(self) -> None:
        """tools/list before initialize → INVALID_REQUEST error."""
        server = _make_server()
        resp = await server.handle_request(
            JSONRPCRequest(id=1, method="tools/list")
        )
        assert resp.error is not None
        assert resp.error.code == ErrorCodes.INVALID_REQUEST

    async def test_search_tool_call(self) -> None:
        """ragx_search tool call returns results."""
        server = _make_server()
        await server.handle_request(JSONRPCRequest(id=1, method="initialize"))

        resp = await server.handle_request(JSONRPCRequest(
            id=2,
            method="tools/call",
            params={
                "name": "ragx_search",
                "arguments": {"kb_id": "kb_1", "query": "test query"},
            },
        ))
        assert resp.error is None
        result = resp.result
        assert result["content"][0]["type"] == "text"
        data = json.loads(result["content"][0]["text"])
        assert "results" in data
        assert data["results"][0]["chunk_id"] == "chk_1"

    async def test_generate_tool_call(self) -> None:
        """ragx_generate tool call returns an answer."""
        server = _make_server()
        await server.handle_request(JSONRPCRequest(id=1, method="initialize"))

        resp = await server.handle_request(JSONRPCRequest(
            id=2,
            method="tools/call",
            params={
                "name": "ragx_generate",
                "arguments": {"kb_id": "kb_1", "query": "What is RAG?"},
            },
        ))
        assert resp.error is None
        data = json.loads(resp.result["content"][0]["text"])
        assert "answer" in data
        assert data["answer"] == "This is the answer."

    async def test_list_kbs_tool_call(self) -> None:
        """ragx_list_kbs tool call returns KB list."""
        server = _make_server()
        await server.handle_request(JSONRPCRequest(id=1, method="initialize"))

        resp = await server.handle_request(JSONRPCRequest(
            id=2,
            method="tools/call",
            params={"name": "ragx_list_kbs", "arguments": {}},
        ))
        assert resp.error is None
        data = json.loads(resp.result["content"][0]["text"])
        assert len(data) == 2
        assert data[0]["kb_id"] == "kb_1"

    async def test_unknown_tool_error(self) -> None:
        """Unknown tool name → INVALID_PARAMS error."""
        server = _make_server()
        await server.handle_request(JSONRPCRequest(id=1, method="initialize"))

        resp = await server.handle_request(JSONRPCRequest(
            id=2,
            method="tools/call",
            params={"name": "unknown_tool", "arguments": {}},
        ))
        assert resp.error is not None
        assert resp.error.code == ErrorCodes.INVALID_PARAMS

    async def test_unknown_method_error(self) -> None:
        """Unknown method → METHOD_NOT_FOUND error."""
        server = _make_server()
        await server.handle_request(JSONRPCRequest(id=1, method="initialize"))

        resp = await server.handle_request(
            JSONRPCRequest(id=2, method="unknown/method")
        )
        assert resp.error is not None
        assert resp.error.code == ErrorCodes.METHOD_NOT_FOUND

    async def test_search_error_returns_error_result(self) -> None:
        """Search failure returns an error result (not a JSON-RPC error)."""
        async def failing_search(**kwargs):
            raise ValueError("connection refused")

        server = MCPServer(search_fn=failing_search)
        await server.handle_request(JSONRPCRequest(id=1, method="initialize"))

        resp = await server.handle_request(JSONRPCRequest(
            id=2,
            method="tools/call",
            params={
                "name": "ragx_search",
                "arguments": {"kb_id": "kb_1", "query": "test"},
            },
        ))
        # Tool execution errors are wrapped as error results, not JSON-RPC errors
        assert resp.error is None
        result = resp.result
        assert result["is_error"] is True
        assert "Search failed" in result["content"][0]["text"]
