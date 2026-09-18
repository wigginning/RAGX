"""Bridge :class:`AuditStore` to the persistent :class:`MetadataStore`.

The audit middleware writes one record per HTTP request to an
:class:`AuditStore`. When the API runs against the full profile, the audit
records must land in the metadata DB so they survive restarts; this module
provides the adapter.
"""

from __future__ import annotations

from typing import Any

from ragx.api.middleware.audit import AuditEntry


class MetadataAuditStore:
    """AuditStore adapter that persists to a MetadataStore.

    Best-effort: a write failure is logged but never raised — the audit
    middleware already catches and logs at the call site.
    """

    def __init__(self, db: Any) -> None:
        self.db = db

    async def append(self, entry: AuditEntry) -> None:
        await self.db.save_audit_entry(
            audit_id=entry.audit_id,
            ts=entry.timestamp,
            tenant_id=entry.tenant_id,
            key_id=entry.key_id,
            method=entry.method,
            path=entry.path,
            status_code=entry.status_code,
            trace_id=entry.trace_id,
            action=entry.action,
            resource=entry.resource,
            details=entry.details,
        )


__all__ = ["MetadataAuditStore"]
