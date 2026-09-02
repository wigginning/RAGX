"""GET /v1/metrics (10-observability.md §10.2).

Prometheus exposition endpoint. Renders the process-wide :class:`Metrics`
registry owned by :mod:`ragx.observability.metrics`. The endpoint is mounted
unauthenticated on the same prefix as other API routes so the standard
Prometheus scrape config can target it directly.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import Response

router = APIRouter()


@router.get("/metrics")
async def metrics(request: Request) -> Response:
    """Render the Prometheus exposition format.

    The renderer is pulled lazily so the metric module is not imported until
    the endpoint is hit (the lite profile may run without prometheus_client
    installed if metrics are not consumed; we still expose the route so that
    ``prometheus_client`` is the only hard dep when this endpoint is wired).
    """
    from ragx.observability.metrics import get_metrics

    metrics_singleton = getattr(request.app.state, "metrics", None) or get_metrics()
    request.app.state.metrics = metrics_singleton
    body = metrics_singleton.render()
    return Response(content=body, media_type="text/plain; version=0.0.4; charset=utf-8")
