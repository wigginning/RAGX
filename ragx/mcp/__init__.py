"""MCP (Model Context Protocol) server (09-api.md §9.7).

Exposes RAGX capabilities as standard MCP tools for any MCP client:

* ``ragx_search``    — POST /v1/search (pure retrieval)
* ``ragx_generate``  — POST /v1/chat/completions (stream=false)
* ``ragx_list_kbs``  — list knowledge bases

Transports (§9.7.2): stdio (JSON-RPC over stdin/stdout) and SSE
(``GET /v1/mcp/sse`` + ``POST /v1/mcp/messages``).
"""

from ragx.mcp.protocol import (
    JSONRPCRequest,
    JSONRPCResponse,
    MCPTool,
    ToolCallResult,
)
from ragx.mcp.server import MCPServer
from ragx.mcp.transport import StdioTransport

__all__ = [
    "JSONRPCRequest",
    "JSONRPCResponse",
    "MCPServer",
    "MCPTool",
    "StdioTransport",
    "ToolCallResult",
]
