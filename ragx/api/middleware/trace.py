"""Trace middleware (09-api.md §9.1.1, implementation list §9.8.3).

Generates or passes through a ``trace_id`` (ULID, 26 chars) and writes it back
in the ``X-Trace-Id`` response header.
"""

from __future__ import annotations

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from ragx.core.ids import ulid


class TraceMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        incoming = request.headers.get("X-Trace-Id")
        if incoming and len(incoming) == 26:
            trace_id = incoming
        else:
            trace_id = ulid()
        request.state.trace_id = trace_id
        response = await call_next(request)
        response.headers["X-Trace-Id"] = trace_id
        return response
