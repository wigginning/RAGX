"""stdio MCP server launcher (09-api.md §9.7.2).

Run with ``ragx-mcp`` (or ``python -m ragx.mcp``). Builds the RAGX app from
environment configuration and serves the MCP tools over stdin/stdout, so a
local Agent / CLI can call ``ragx_search`` / ``ragx_generate`` / ``ragx_list_kbs``.
"""

from __future__ import annotations

import asyncio

from ragx.api.app import create_app
from ragx.mcp.server import build_mcp_server
from ragx.mcp.transport import StdioTransport


def main() -> None:
    app = create_app()
    server = build_mcp_server(app)
    transport = StdioTransport(server)
    asyncio.run(transport.run())


if __name__ == "__main__":
    main()
