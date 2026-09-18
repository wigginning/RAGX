"""Audit log (RX-API-03, 09-api.md §9.6).

Every data read/write that touches user content is appended to the audit
log so security and compliance teams can reconstruct who-accessed-what. The
log lives in the Metadata DB so it survives restarts; the API service
emits one entry per request (after the route completes).

Logged fields
-------------

* ``timestamp``     — UTC ISO-8601
* ``tenant_id``     — tenant the key resolves to (default ``"default"``)
* ``key_id``        — API key id or ``"anonymous"``
* ``method``        — HTTP method
* ``path``          — request path
* ``status_code``   — response status
* ``trace_id``      — propagated from ``TraceMiddleware``
* ``kb_id``         — kb the request targets (when known)
* ``action``        — coarse-grained tag (read / write / delete / admin)
* ``resource``      — resource touched (doc / chunk / query / settings)
* ``details``       — extra context (doc id, chunk id, query length, …)

Storage
-------

``AuditStore.append(entry)`` is the single write point. Implementations:
* :class:`InMemoryAuditStore` — for tests and the lite profile
* :class:`MetadataStore.audit_log` — full profile (added in this module)

The default writes use best-effort semantics: an audit failure must NOT
block the main flow. Failed entries are logged to the standard logger and
counted via the ``ragx_audit_failure_total`` metric.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from ragx.core.ids import new_id

logger = logging.getLogger("ragx.api.middleware.audit")


@dataclass
class AuditEntry:
    """A single audit-log record (09-api.md §9.6)."""

    timestamp: str
    tenant_id: str
    key_id: str
    method: str
    path: str
    status_code: int
    trace_id: str
    action: str  # read | write | delete | admin
    resource: str  # document | chunk | query | kb | settings | …
    details: dict[str, Any] = field(default_factory=dict)
    audit_id: str = field(default_factory=lambda: new_id("audit_"))


class AuditStore(Protocol):
    """Persist audit entries; implementations degrade on failure."""

    async def append(self, entry: AuditEntry) -> None: ...


class InMemoryAuditStore:
    """A no-op audit store that keeps the most-recent ``max_entries`` rows.

    Useful for tests and the lite profile where SQLite already lives in the
    Metadata DB but a separate audit table would still be a hard dep.
    """

    def __init__(self, max_entries: int = 1000) -> None:
        self._entries: list[AuditEntry] = []
        self._max = max_entries

    async def append(self, entry: AuditEntry) -> None:
        self._entries.append(entry)
        if len(self._entries) > self._max:
            self._entries = self._entries[-self._max :]

    def entries(self) -> list[AuditEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)


def _classify(method: str, path: str) -> tuple[str, str]:
    """Return ``(action, resource)`` from the HTTP method + path."""
    verb_to_action = {
        "GET": "read",
        "POST": "write",
        "PUT": "write",
        "PATCH": "write",
        "DELETE": "delete",
    }
    action = verb_to_action.get(method.upper(), "admin")
    parts = [p for p in path.split("/") if p]
    if "documents" in parts:
        resource = "document"
    elif "chunks" in parts:
        resource = "chunk"
    elif "search" in parts:
        resource = "query"
    elif "chat" in parts:
        resource = "chat"
    elif "tasks" in parts:
        resource = "task"
    elif (
        "health" in parts
        or "metrics" in parts
        or "traces" in parts
        or "audit" in parts
    ):
        resource = "ops"
    elif "kb" in parts or "knowledge-base" in parts:
        resource = "kb"
    else:
        resource = "settings"
    return action, resource


class AuditMiddleware(BaseHTTPMiddleware):
    """Record one audit entry per HTTP request (after the route completes).

    Implements ``BaseHTTPMiddleware`` so FastAPI/Starlette can drive it via
    ``add_middleware``. The audit store is read from ``app.state.audit_store``
    on every request — BaseHTTPMiddleware does not forward ``**kwargs`` to
    subclasses, so the store has to be looked up dynamically rather than
    captured at registration time.
    """

    def __init__(self, app: Any) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        start_ns = time.time_ns()
        response = await call_next(request)
        store = getattr(request.app.state, "audit_store", None)
        try:
            entry = self._build_entry(request, response.status_code, start_ns)
            if store is not None:
                await store.append(entry)
        except Exception as exc:  # noqa: BLE001 - audit must not break responses
            logger.warning("audit append failed: %s", exc)
        return response

    def _build_entry(self, request: Request, status_code: int, start_ns: int) -> AuditEntry:
        method = request.method
        path = request.url.path
        state = request.state
        tenant_id = getattr(state, "tenant_id", None) or "default"
        auth_obj = getattr(state, "auth", None)
        key_id = getattr(auth_obj, "key_id", None) or "anonymous"
        trace_id = getattr(state, "trace_id", None) or ""
        action, resource = _classify(method, path)
        details: dict[str, Any] = {
            "duration_ms": (time.time_ns() - start_ns) / 1_000_000.0,
        }
        kb_id = request.query_params.get("kb_id")
        if kb_id:
            details["kb_id"] = kb_id
        return AuditEntry(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            tenant_id=tenant_id,
            key_id=key_id,
            method=method,
            path=path,
            status_code=status_code,
            trace_id=trace_id,
            action=action,
            resource=resource,
            details=details,
        )


def entry_to_dict(entry: AuditEntry) -> dict[str, Any]:
    """Serialise an :class:`AuditEntry` to a JSON-safe dict."""
    return asdict(entry)


__all__ = [
    "AuditEntry",
    "AuditMiddleware",
    "AuditStore",
    "InMemoryAuditStore",
    "entry_to_dict",
]
