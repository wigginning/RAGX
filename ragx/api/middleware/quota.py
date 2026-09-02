"""Per-tenant quota tracker (RX-API-03, 09-api.md §9.6.2).

Enforces two opt-in quotas on top of the per-request rate limiter:

* **Monthly LLM token quota** (``security.monthly_token_quota``) — sums of
  ``ragx_llm_tokens_total{tenant=…}`` resets on the 1st of every UTC month.
* **Monthly upload quota** (``security.monthly_upload_quota_bytes``) — sums of
  request bodies for ``POST /v1/documents``; reset on the 1st of every UTC month.

When a quota is exceeded the request is rejected with ``RateLimitError(1004)``
and the response carries ``Retry-After`` pointing at the next UTC midnight.

Two backends:

* :class:`InMemoryQuotaStore` — lite profile, single-process state.
* :class:`MetadataQuotaStore` — full profile; persists into the ``quotas`` table
  in the metadata DB so quotas survive restarts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from ragx.api.errors import error_response
from ragx.core.exceptions import RateLimitError
from ragx.core.settings import SecurityConfig

logger = logging.getLogger("ragx.api.middleware.quota")


@dataclass
class QuotaUsage:
    tokens: int = 0
    upload_bytes: int = 0
    period_start: str = ""


class QuotaStore(Protocol):
    """Persist the running monthly counters per tenant."""

    async def add_tokens(self, tenant_id: str, count: int) -> QuotaUsage: ...

    async def add_upload(self, tenant_id: str, count: int) -> QuotaUsage: ...

    async def usage(self, tenant_id: str) -> QuotaUsage: ...


class InMemoryQuotaStore:
    """Process-internal counter. No persistence across restarts."""

    def __init__(self) -> None:
        self._usage: dict[str, QuotaUsage] = {}

    @staticmethod
    def _period_start(now: datetime | None = None) -> str:
        now = now or datetime.now(UTC)
        return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()

    def _ensure(self, tenant_id: str) -> QuotaUsage:
        period = self._period_start()
        u = self._usage.get(tenant_id)
        if u is None or u.period_start != period:
            u = QuotaUsage(period_start=period)
            self._usage[tenant_id] = u
        # Return a snapshot so callers see consistent values even if the
        # canonical object mutates between calls.
        return QuotaUsage(
            tokens=u.tokens,
            upload_bytes=u.upload_bytes,
            period_start=u.period_start,
        )

    async def add_tokens(self, tenant_id: str, count: int) -> QuotaUsage:
        canonical = self._usage.get(tenant_id)
        if canonical is None or canonical.period_start != self._period_start():
            self._ensure(tenant_id)  # init for current period
            canonical = self._usage[tenant_id]
        canonical.tokens += count
        # snapshot for the caller
        return QuotaUsage(
            tokens=canonical.tokens,
            upload_bytes=canonical.upload_bytes,
            period_start=canonical.period_start,
        )

    async def add_upload(self, tenant_id: str, count: int) -> QuotaUsage:
        canonical = self._usage.get(tenant_id)
        if canonical is None or canonical.period_start != self._period_start():
            self._ensure(tenant_id)
            canonical = self._usage[tenant_id]
        canonical.upload_bytes += count
        return QuotaUsage(
            tokens=canonical.tokens,
            upload_bytes=canonical.upload_bytes,
            period_start=canonical.period_start,
        )

    async def usage(self, tenant_id: str) -> QuotaUsage:
        return self._ensure(tenant_id)


class MetadataQuotaStore:
    """Counter that persists into a dedicated ``quotas`` table."""

    def __init__(self, db: Any) -> None:
        self.db = db

    @staticmethod
    def _period_start(now: datetime | None = None) -> str:
        now = now or datetime.now(UTC)
        return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()

    @staticmethod
    def _next_period_seconds() -> int:
        now = datetime.now(UTC)
        if now.month == 12:
            nxt = now.replace(year=now.year + 1, month=1, day=1)
        else:
            nxt = now.replace(month=now.month + 1, day=1)
        return max(1, int((nxt - now).total_seconds()))

    async def add_tokens(self, tenant_id: str, count: int) -> QuotaUsage:
        return await self.db.bump_quota(tenant_id, "tokens", count, self._period_start())

    async def add_upload(self, tenant_id: str, count: int) -> QuotaUsage:
        return await self.db.bump_quota(tenant_id, "upload_bytes", count, self._period_start())

    async def usage(self, tenant_id: str) -> QuotaUsage:
        return await self.db.get_quota(tenant_id, self._period_start())


class QuotaMiddleware(BaseHTTPMiddleware):
    """Reject requests that breach monthly tenant quotas (09-api.md §9.6.2)."""

    def __init__(self, app: Any, *, config: SecurityConfig, store: QuotaStore | None = None) -> None:
        super().__init__(app)
        self.config = config
        # BaseHTTPMiddleware doesn't forward kwargs, but we read from app.state.
        self._store = store

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        store = self._store or getattr(request.app.state, "quota_store", None)
        tenant_id = getattr(request.state, "tenant_id", None) or "default"
        path = request.url.path
        method = request.method

        # Upload quota: only count POST /v1/documents requests with a body.
        if (
            method == "POST"
            and path.endswith("/v1/documents")
            and store is not None
            and self.config.monthly_upload_quota_bytes > 0
        ):
            cl = request.headers.get("content-length")
            try:
                size = int(cl) if cl else 0
            except ValueError:
                size = 0
            usage = await store.add_upload(tenant_id, size)
            if usage.upload_bytes > self.config.monthly_upload_quota_bytes:
                retry_after = MetadataQuotaStore._next_period_seconds()
                return error_response(
                    RateLimitError(
                        "monthly upload quota exceeded",
                        code=1004,
                        details={
                            "tenant_id": tenant_id,
                            "used": usage.upload_bytes,
                            "cap": self.config.monthly_upload_quota_bytes,
                            "retry_after": retry_after,
                        },
                    ),
                    getattr(request.state, "trace_id", None),
                )

        response = await call_next(request)

        # LLM token quota: back-fill from the response's usage block when present.
        if (
            store is not None
            and self.config.monthly_token_quota > 0
            and response.headers.get("content-type", "").startswith("application/json")
        ):
            try:
                # Read the response body once, count its tokens if it includes usage.
                body_chunks: list[bytes] = []
                async for chunk in response.body_iterator:  # type: ignore[attr-defined]
                    body_chunks.append(chunk if isinstance(chunk, bytes) else chunk.encode())
                body_bytes = b"".join(body_chunks)
                try:
                    payload = json.loads(body_bytes) if body_bytes else {}
                except Exception:
                    payload = {}
                usage_block = payload.get("usage") if isinstance(payload, dict) else None
                if isinstance(usage_block, dict):
                    total = int(usage_block.get("total_tokens", 0)) or 0
                    if total > 0:
                        updated = await store.add_tokens(tenant_id, total)
                        if updated.tokens > self.config.monthly_token_quota:
                            logger.warning(
                                "tenant %s exceeded monthly token quota (%s > %s)",
                                tenant_id,
                                updated.tokens,
                                self.config.monthly_token_quota,
                            )
                # Rebuild the response so the body is consumable downstream.
                from starlette.responses import Response as StarletteResponse

                response = StarletteResponse(
                    content=body_bytes,
                    status_code=response.status_code,
                    headers={k: v for k, v in response.headers.items() if k.lower() != "content-length"},
                    media_type=response.headers.get("content-type"),
                )
            except Exception as exc:  # noqa: BLE001 - quota must not break responses
                logger.warning("quota middleware failed: %s", exc)
        return response


__all__ = [
    "QuotaUsage",
    "QuotaStore",
    "InMemoryQuotaStore",
    "MetadataQuotaStore",
    "QuotaMiddleware",
]
