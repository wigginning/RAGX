"""GET /v1/audit (09-api.md §9.6).

Read access to the audit log. The current tenant always scopes its own
entries; cross-tenant access requires admin scope (not yet modelled —
kept open for the v1.0 release).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from ragx.api.middleware import entry_to_dict

router = APIRouter()


class AuditEntryResponse(BaseModel):
    entries: list[dict[str, Any]] = Field(default_factory=list)
    count: int = 0
    trace_id: str


@router.get("/audit", response_model=AuditEntryResponse)
async def list_audit_entries(
    request: Request,
    tenant_id: str | None = None,
    limit: int = 100,
) -> AuditEntryResponse:
    """Return recent audit entries for the current tenant (or any tenant if admin)."""
    state = request.state
    auth = getattr(state, "auth", None)
    effective_tenant = tenant_id or (
        getattr(auth, "tenant_id", None) if auth else "default"
    )
    store = getattr(request.app.state, "audit_store", None)
    if store is None:
        return AuditEntryResponse(entries=[], count=0, trace_id=state.trace_id)

    # The in-memory store has no tenant filter; iterate directly.
    from ragx.api.middleware.audit import InMemoryAuditStore

    if isinstance(store, InMemoryAuditStore):
        rows = [
            e for e in store.entries()
            if effective_tenant is None or e.tenant_id == effective_tenant
        ]
        rows = list(reversed(rows))[:limit]
        return AuditEntryResponse(
            entries=[entry_to_dict(e) for e in rows],
            count=len(rows),
            trace_id=state.trace_id,
        )

    # Persistent store: ask the metadata DB.
    db = request.app.state.db
    rows = await db.list_audit_entries(tenant_id=effective_tenant, limit=limit)
    return AuditEntryResponse(
        entries=rows,
        count=len(rows),
        trace_id=state.trace_id,
    )
