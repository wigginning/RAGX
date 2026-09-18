"""MCP JSON-RPC protocol types (09-api.md §9.7).

The MCP protocol uses JSON-RPC 2.0. This module defines the typed
request/response models and the tool descriptor schema.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MCPTool(BaseModel):
    """MCP tool descriptor (§9.7.1)."""

    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)


class JSONRPCRequest(BaseModel):
    """JSON-RPC 2.0 request."""

    jsonrpc: str = "2.0"
    id: int | str | None = None
    method: str
    params: dict[str, Any] = Field(default_factory=dict)


class JSONRPCError(BaseModel):
    """JSON-RPC 2.0 error object."""

    code: int
    message: str
    data: Any | None = None


class JSONRPCResponse(BaseModel):
    """JSON-RPC 2.0 response."""

    jsonrpc: str = "2.0"
    id: int | str | None = None
    result: Any | None = None
    error: JSONRPCError | None = None

    @classmethod
    def ok(cls, id: int | str | None, result: Any) -> JSONRPCResponse:
        return cls(id=id, result=result)

    @classmethod
    def failure(
        cls,
        id: int | str | None,
        code: int,
        message: str,
        data: Any | None = None,
    ) -> JSONRPCResponse:
        return cls(
            id=id,
            error=JSONRPCError(code=code, message=message, data=data),
        )


class ToolCallResult(BaseModel):
    """Result of an MCP tool call (§9.7.3)."""

    content: list[dict[str, Any]] = Field(default_factory=list)
    is_error: bool = False

    @classmethod
    def from_text(cls, text: str, is_error: bool = False) -> ToolCallResult:
        return cls(
            content=[{"type": "text", "text": text}],
            is_error=is_error,
        )

    @classmethod
    def from_json(cls, data: Any, is_error: bool = False) -> ToolCallResult:
        import json

        return cls.from_text(json.dumps(data, ensure_ascii=False), is_error)


#: MCP protocol version.
PROTOCOL_VERSION = "2024-11-05"

#: Error codes (JSON-RPC 2.0 + MCP extensions).
class ErrorCodes:
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
