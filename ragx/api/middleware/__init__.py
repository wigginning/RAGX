"""API middleware package (09-api.md §9.8).

* ``trace.py`` — TraceMiddleware (trace_id generation/passthrough, §9.1.1)
* ``auth.py`` — AuthMiddleware (API Key sha256 + kb_acl + optional JWT, §9.2)
* ``ratelimit.py`` — per-key token-bucket rate limiter (§9.2.4)
* ``audit.py`` — AuditMiddleware (one record per HTTP request, §9.6)
* ``quota.py`` — per-tenant monthly quota gate (RX-API-03)
"""

from __future__ import annotations

from ragx.api.middleware.audit import (
    AuditEntry,
    AuditMiddleware,
    AuditStore,
    InMemoryAuditStore,
    entry_to_dict,
)
from ragx.api.middleware.audit_store import MetadataAuditStore
from ragx.api.middleware.auth import AuthContext, AuthMiddleware, require_kb_access
from ragx.api.middleware.quota import (
    InMemoryQuotaStore,
    MetadataQuotaStore,
    QuotaMiddleware,
    QuotaStore,
    QuotaUsage,
)
from ragx.api.middleware.ratelimit import (
    InProcessRateLimiter,
    RateLimiter,
    RateLimitMiddleware,
    RedisRateLimiter,
)
from ragx.api.middleware.trace import TraceMiddleware

__all__ = [
    "AuditEntry",
    "AuditMiddleware",
    "AuditStore",
    "AuthContext",
    "AuthMiddleware",
    "InMemoryAuditStore",
    "InMemoryQuotaStore",
    "MetadataAuditStore",
    "MetadataQuotaStore",
    "QuotaMiddleware",
    "QuotaStore",
    "QuotaUsage",
    "RateLimiter",
    "RateLimitMiddleware",
    "InProcessRateLimiter",
    "RedisRateLimiter",
    "TraceMiddleware",
    "entry_to_dict",
    "require_kb_access",
]
