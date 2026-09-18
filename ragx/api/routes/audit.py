"""GET /v1/audit (09-api.md §9.6).

Read access to the audit log. A caller is scoped to its own tenant; a
``tenant_id`` query param naming a foreign tenant is rejected with 403/1003
(admin cross-tenant scope is not modelled yet, so nobody can read another
tenant's trail through this endpoint).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from ragx.api.middleware import entry_to_dict, require_tenant_access

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
    """Return recent audit entries for the current tenant.

    A caller may only read its own tenant's entries. The ``tenant_id`` query
    param is honoured only when it equals the caller's tenant — passing a
    foreign ``tenant_id`` raises 403/1003 instead of leaking another tenant's
    audit trail (admin cross-tenant scope is not modelled yet).
    """
    state = request.state
    auth = getattr(state, "auth", None)
    caller_tenant = getattr(auth, "tenant_id", None) if auth else None
    effective_tenant = tenant_id or caller_tenant or "default"
    if tenant_id is not None:
        # Mirrors require_kb_access: unauthenticated (auth disabled) callers
        # are unrestricted; authenticated callers are pinned to their tenant.
        require_tenant_access(request, tenant_id)
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
