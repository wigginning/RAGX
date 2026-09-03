"""MCP Server (09-api.md §9.7.1).

Registers the three RAGX tools and dispatches JSON-RPC requests to
the appropriate handler.

Tools (§9.7.1):
* ``ragx_search``    — POST /v1/search (pure retrieval)
* ``ragx_generate``  — POST /v1/chat/completions (stream=false)
* ``ragx_list_kbs``  — list knowledge bases
"""

from __future__ import annotations

import logging
from typing import Any

from ragx.mcp.protocol import (
    PROTOCOL_VERSION,
    ErrorCodes,
    JSONRPCRequest,
    JSONRPCResponse,
    MCPTool,
    ToolCallResult,
)

logger = logging.getLogger("ragx.mcp.server")

#: Tool definitions with their JSON Schema input contracts (§9.7.1).
_TOOL_DEFS: list[MCPTool] = [
    MCPTool(
        name="ragx_search",
        description=(
            "Search the knowledge base. Returns relevant chunks with scores "
            "and sources (dense/bm25/graph_low/graph_high)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "kb_id": {"type": "string", "description": "Knowledge base ID"},
                "query": {"type": "string", "description": "Search query"},
                "top_k": {"type": "integer", "description": "Max results", "default": 8},
                "filter": {"type": "object", "description": "Optional metadata filter"},
                "mode": {"type": "string", "description": "Retrieval mode", "default": "standard"},
            },
            "required": ["kb_id", "query"],
        },
    ),
    MCPTool(
        name="ragx_generate",
        description=(
            "Generate an answer from the knowledge base using RAG. "
            "Returns the answer, citations, and metadata."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "kb_id": {"type": "string", "description": "Knowledge base ID"},
                "query": {"type": "string", "description": "User question"},
                "mode": {"type": "string", "description": "Mode: auto/fast/standard/agentic", "default": "auto"},
                "overrides": {"type": "object", "description": "Optional overrides"},
            },
            "required": ["kb_id", "query"],
        },
    ),
    MCPTool(
        name="ragx_list_kbs",
        description="List all available knowledge bases.",
        input_schema={
            "type": "object",
            "properties": {},
        },
    ),
]


class MCPServer:
    """MCP server that dispatches JSON-RPC requests to tool handlers.

    The server does not own the transport — it provides ``handle_request``
    which the transport layer calls for each JSON-RPC message.
    """

    def __init__(
        self,
        search_fn: Any | None = None,
        generate_fn: Any | None = None,
        list_kbs_fn: Any | None = None,
        kb_acl: list[str] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        search_fn:
            Async callable ``(kb_id, query, top_k, filter_expr, mode) -> dict``.
            Called by ``ragx_search``. Defaults to a stub.
        generate_fn:
            Async callable ``(kb_id, query, mode, overrides) -> dict``.
            Called by ``ragx_generate``. Defaults to a stub.
        list_kbs_fn:
            Async callable ``() -> list[dict]``.
            Called by ``ragx_list_kbs``. Defaults to returning an empty list.
        kb_acl:
            Optional allow-list of kb_ids this MCP session may touch. When
            set, ``ragx_search`` / ``ragx_generate`` reject other kbs and
            ``ragx_list_kbs`` is filtered to the allow-list (tenant scoping
            for the SSE transport, which builds the server per API key).
        """
        self._search_fn = search_fn or _stub_search
        self._generate_fn = generate_fn or _stub_generate
        self._list_kbs_fn = list_kbs_fn or _stub_list_kbs
        self._kb_acl = kb_acl
        self._initialized = False

    # -- JSON-RPC dispatch -------------------------------------------------
    async def handle_request(self, req: JSONRPCRequest) -> JSONRPCResponse:
        """Dispatch a JSON-RPC request to the appropriate handler."""
        if req.method == "initialize":
            return self._handle_initialize(req)

        if not self._initialized:
            return JSONRPCResponse.failure(
                req.id,
                ErrorCodes.INVALID_REQUEST,
                "Server not initialized — send 'initialize' first",
            )

        if req.method == "tools/list":
            return self._handle_tools_list(req)

        if req.method == "tools/call":
            return await self._handle_tools_call(req)

        return JSONRPCResponse.failure(
            req.id,
            ErrorCodes.METHOD_NOT_FOUND,
            f"Unknown method: {req.method}",
        )

    # -- MCP method handlers -----------------------------------------------
    def _handle_initialize(self, req: JSONRPCRequest) -> JSONRPCResponse:
        """Handle the MCP initialize handshake (§9.7.2)."""
        self._initialized = True
        return JSONRPCResponse.ok(req.id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {
                "tools": {"listChanged": False},
            },
            "serverInfo": {
                "name": "ragx-mcp",
                "version": "0.1.0",
            },
        })

    def _handle_tools_list(self, req: JSONRPCRequest) -> JSONRPCResponse:
        """Handle tools/list (§9.7.1)."""
        tools = [t.model_dump() for t in _TOOL_DEFS]
        return JSONRPCResponse.ok(req.id, {"tools": tools})

    async def _handle_tools_call(
        self, req: JSONRPCRequest
    ) -> JSONRPCResponse:
        """Handle tools/call (§9.7.3)."""
        params = req.params or {}
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        if tool_name == "ragx_search":
            result = await self._exec_search(arguments)
        elif tool_name == "ragx_generate":
            result = await self._exec_generate(arguments)
        elif tool_name == "ragx_list_kbs":
            result = await self._exec_list_kbs()
        else:
            return JSONRPCResponse.failure(
                req.id,
                ErrorCodes.INVALID_PARAMS,
                f"Unknown tool: {tool_name}",
            )

        if result is None:
            return JSONRPCResponse.failure(
                req.id,
                ErrorCodes.INTERNAL_ERROR,
                "Tool execution failed",
            )
        return JSONRPCResponse.ok(req.id, result.model_dump())

    # -- Tool implementations ----------------------------------------------
    async def _exec_search(self, args: dict[str, Any]) -> ToolCallResult | None:
        if self._kb_acl is not None and args.get("kb_id") not in self._kb_acl:
            return ToolCallResult.from_text(
                f"kb_id '{args.get('kb_id')}' is not authorised for this MCP session",
                is_error=True,
            )
        try:
            result = await self._search_fn(
                kb_id=args["kb_id"],
                query=args["query"],
                top_k=args.get("top_k", 8),
                filter_expr=args.get("filter"),
                mode=args.get("mode", "standard"),
            )
            return ToolCallResult.from_json(result)
        except Exception as exc:
            logger.error("ragx_search failed: %s", exc)
            return ToolCallResult.from_text(
                f"Search failed: {exc}", is_error=True
            )

    async def _exec_generate(self, args: dict[str, Any]) -> ToolCallResult | None:
        if self._kb_acl is not None and args.get("kb_id") not in self._kb_acl:
            return ToolCallResult.from_text(
                f"kb_id '{args.get('kb_id')}' is not authorised for this MCP session",
                is_error=True,
            )
        try:
            result = await self._generate_fn(
                kb_id=args["kb_id"],
                query=args["query"],
                mode=args.get("mode", "auto"),
                overrides=args.get("overrides"),
            )
            return ToolCallResult.from_json(result)
        except Exception as exc:
            logger.error("ragx_generate failed: %s", exc)
            return ToolCallResult.from_text(
                f"Generation failed: {exc}", is_error=True
            )

    async def _exec_list_kbs(self) -> ToolCallResult | None:
        try:
            kbs = await self._list_kbs_fn()
            return ToolCallResult.from_json(kbs)
        except Exception as exc:
            logger.error("ragx_list_kbs failed: %s", exc)
            return ToolCallResult.from_text(
                f"List KBs failed: {exc}", is_error=True
            )


# ── Default stub implementations (no dependencies) ─────────────────────


async def _stub_search(**kwargs) -> dict:
    return {
        "results": [],
        "message": "ragx_search stub — wire a real search_fn to MCPServer",
    }


async def _stub_generate(**kwargs) -> dict:
    return {
        "answer": "ragx_generate stub — wire a real generate_fn to MCPServer",
        "mode": "standard",
    }


async def _stub_list_kbs() -> list[dict]:
    return []


def build_mcp_server(app: Any, *, kb_acl: list[str] | None = None) -> MCPServer:
    """Wire an :class:`MCPServer` to the real RAGX backend (§9.7).

    Uses the live app's registry / ``query_service`` / metadata store so the
    three tools actually retrieve and generate instead of returning the
    default stubs. This is what makes the MCP server usable — without it the
    server ships inert.

    ``kb_acl`` scopes the session to an allow-list of kb_ids (tenant scoping
    for the SSE transport, which builds one server per authenticated API key).
    """
    from ragx.core.models import FilterExpr, RequestOverride
    from ragx.retrieval.filters import validate_filter
    from ragx.retrieval.hybrid import GraphChunkResolver, HybridRetriever
    from ragx.retrieval.models import RetrievalConfig

    db = app.state.db
    registry = app.state.registry
    kb_cfg = app.state.kb_cfg
    query_service = app.state.query_service
    retriever_cache: dict[str, HybridRetriever] = {}

    def _retriever_for(kb_id: str) -> HybridRetriever:
        cached = retriever_cache.get(kb_id)
        if cached is not None:
            return cached
        store = registry.resolve_from_kb("vector_store", kb_cfg, kb_id=kb_id)
        graph = (
            registry.resolve_from_kb("graph_store", kb_cfg, kb_id=kb_id)
            if kb_cfg.graph_store else None
        )
        resolver = GraphChunkResolver(store) if graph is not None else None
        retr = HybridRetriever(
            store, graph, resolver, RetrievalConfig(),
            kg_enabled=kb_cfg.flags.kg_enabled,
        )
        retriever_cache[kb_id] = retr
        return retr

    async def search_fn(kb_id, query, top_k=8, filter_expr=None, mode="standard"):
        embedder = registry.resolve_from_kb("embedder", kb_cfg, kb_id=kb_id)
        qvec = (await embedder.embed([query]))[0]
        parsed_filter = None
        if isinstance(filter_expr, dict):
            try:
                parsed_filter = FilterExpr.model_validate(filter_expr)
            except Exception:
                parsed_filter = None
        elif filter_expr is not None:
            parsed_filter = validate_filter(filter_expr)
        hits = await _retriever_for(kb_id).retrieve(query, qvec, parsed_filter)
        results = [
            {
                "chunk_id": h.chunk.chunk_id,
                "doc_id": h.chunk.doc_id,
                "text": h.chunk.text,
                "score": h.rrf_score if h.rrf_score is not None else (h.rerank_score or 0.0),
                "source": ",".join(h.sources),
            }
            for h in hits[:top_k]
        ]
        return {"kb_id": kb_id, "query": query, "results": results}

    async def generate_fn(kb_id, query, mode="auto", overrides=None):
        embedder = registry.resolve_from_kb("embedder", kb_cfg, kb_id=kb_id)
        qvec = (await embedder.embed([query]))[0]
        override = RequestOverride.from_dict(overrides) if overrides else None
        result = await query_service.query(
            query, qvec, kb_id=kb_id, trace_id=f"mcp-{kb_id}", override=override,
        )
        return {
            "answer": result.answer,
            "citations": [c.model_dump() for c in result.citations],
            "trace_id": result.trace_id,
            "mode": result.mode,
        }

    async def list_kbs_fn():
        kbs = await db.list_kbs()
        if kb_acl:
            allowed = set(kb_acl)
            kbs = [k for k in kbs if k.get("kb_id") in allowed]
        return kbs

    return MCPServer(
        search_fn=search_fn, generate_fn=generate_fn, list_kbs_fn=list_kbs_fn,
        kb_acl=kb_acl,
    )
