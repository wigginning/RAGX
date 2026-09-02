"""GET /v1/health (09-api.md §9.4.10)."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

router = APIRouter()


class Check(BaseModel):
    ok: bool
    latency_ms: float = 0.0
    error: str | None = None
    code: int | None = None


class HealthReport(BaseModel):
    status: str
    profile: str = "lite"
    checks: dict[str, Check] = Field(default_factory=dict)


@router.get("/health", response_model=HealthReport)
async def health(request: Request, deep: bool = False) -> HealthReport:
    if not deep:
        return HealthReport(status="ok")
    checks: dict[str, Check] = {}
    try:
        store = request.app.state.vector_store
        await store.count()
        checks["vector_store"] = Check(ok=True)
    except Exception as e:  # noqa: BLE001
        checks["vector_store"] = Check(ok=False, error=str(e))
    try:
        await request.app.state.db.get_task("__probe__")
        checks["metadata_db"] = Check(ok=True)
    except Exception as e:  # noqa: BLE001
        checks["metadata_db"] = Check(ok=False, error=str(e))
    degraded = any(not c.ok for c in checks.values())
    return HealthReport(status="degraded" if degraded else "ok", checks=checks)
