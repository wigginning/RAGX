"""Unified error handling (09-api.md §9.1.2 / §9.6).

Maps domain exceptions to the ``{error: {code, message, trace_id}}`` envelope
and the HTTP status table. ``details`` is never leaked to clients.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ragx.core.exceptions import RAGXError

#: code -> HTTP status (09-api.md §9.6).
_HTTP_MAP: dict[int, int] = {
    1001: 400, 1002: 401, 1003: 403, 1004: 429,
    2001: 422, 2002: 415, 2003: 404, 2004: 200,
    3001: 404, 3002: 409,
    4001: 404, 4002: 200, 4003: 400,
    5001: 503, 5002: 503, 5003: 422,
    6001: 502, 6002: 503, 6003: 400, 6004: 503,
    7001: 500, 7002: 502, 7003: 200,
    9001: 503, 9002: 503, 9003: 500,
}


def http_status_for(code: int) -> int:
    return _HTTP_MAP.get(code, 500)


def error_body(exc: RAGXError, trace_id: str | None = None) -> dict:
    body: dict = {"code": exc.code, "message": exc.message}
    tid = exc.trace_id or trace_id
    if tid:
        body["trace_id"] = tid
    return {"error": body}


def error_response(exc: RAGXError, trace_id: str | None = None) -> JSONResponse:
    """Build the unified error JSONResponse (status + envelope + headers).

    Exposed separately from the exception handler so middleware that raises
    ``RAGXError`` (AuthMiddleware / RateLimitMiddleware) can convert it to the
    same wire format — the FastAPI ``ExceptionMiddleware`` only wraps route
    handlers, not middleware that sits outside it.
    """
    tid = exc.trace_id or trace_id
    headers: dict[str, str] = {"X-Trace-Id": tid or ""}
    # Rate limiting (1004) must surface a Retry-After header (§9.2.4).
    retry_after = exc.details.get("retry_after")
    if retry_after is not None:
        headers["Retry-After"] = str(retry_after)
    return JSONResponse(
        status_code=http_status_for(exc.code),
        content=error_body(exc, tid),
        headers=headers,
    )


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RAGXError)
    async def _ragx_error_handler(request: Request, exc: RAGXError) -> JSONResponse:
        tid = exc.trace_id or getattr(request.state, "trace_id", None)
        return error_response(exc, tid)

    @app.exception_handler(Exception)
    async def _unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        # never leak internals; 500 with a generic message
        return JSONResponse(
            status_code=500,
            content={"error": {"code": 5000, "message": "internal server error"}},
        )
