"""OpenTelemetry-style tracing (10-observability.md §10.1).

The real ``opentelemetry`` SDK is an optional dependency. When it is absent
(the lite profile default), this module provides a lightweight, in-process
``Tracer``/``Span`` pair whose API surface mirrors the OTel ``Tracer`` so
domain code never changes. An :class:`InMemorySpanExporter` captures every
span for unit-test assertions on the span tree structure (OBS-01 DoD).

Span attribute dictionary: §10.1.3 — every span carries a ``name``, a
``parent`` id (or ``None`` for the root), typed ``attributes``, optional
``events`` (exceptions), and ``status`` (``OK`` / ``ERROR``).
"""

from __future__ import annotations

import contextvars
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

# Context var holding the *currently active* span (so child spans auto-link).
_current_span: contextvars.ContextVar[Span | None] = contextvars.ContextVar(
    "ragx_current_span", default=None
)


@dataclass
class SpanEvent:
    """An event recorded on a span (exception, checkpoint, …)."""

    name: str
    timestamp: float
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    """A single span — the unit of the trace tree (§10.1).

    Mirrors ``opentelemetry.trace.Span`` closely enough that domain code is
    identical whether the real SDK or this lightweight fallback is active.
    """

    name: str
    trace_id: str
    span_id: str
    parent_id: str | None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[SpanEvent] = field(default_factory=list)
    status: str = "OK"                       # OK | ERROR
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    _token: contextvars.Token | None = field(default=None, repr=False)

    # -- attribute / event API --------------------------------------------
    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def set_attributes(self, attrs: dict[str, Any]) -> None:
        self.attributes.update(attrs)

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.events.append(
            SpanEvent(name=name, timestamp=time.time(), attributes=dict(attributes or {}))
        )

    def record_exception(self, exc: BaseException) -> None:
        """Mark the span ERROR and record the exception as an event (§10.1.3)."""
        self.status = "ERROR"
        self.add_event(
            "exception",
            {
                "exception.type": type(exc).__name__,
                "exception.message": str(exc),
            },
        )
        if hasattr(exc, "code"):
            self.attributes["code"] = exc.code          # domain error code

    # -- lifecycle ---------------------------------------------------------
    def end(self) -> None:
        if self.end_time is None:
            self.end_time = time.time()
        if self._token is not None:
            _current_span.reset(self._token)
            self._token = None

    @property
    def duration_ms(self) -> float:
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000.0

    def __enter__(self) -> Span:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_val is not None:
            self.record_exception(exc_val)
        self.end()


class InMemorySpanExporter:
    """Captures finished spans for test assertions (OBS-01 DoD).

    Thread-safe; spans are stored in completion order.
    """

    def __init__(self) -> None:
        self._spans: list[Span] = []
        self._lock = threading.Lock()

    def export(self, span: Span) -> None:
        with self._lock:
            self._spans.append(span)

    @property
    def spans(self) -> list[Span]:
        with self._lock:
            return list(self._spans)

    def clear(self) -> None:
        with self._lock:
            self._spans.clear()

    # -- tree helpers (for assertions) ------------------------------------
    def children(self, parent_id: str) -> list[Span]:
        return [s for s in self.spans if s.parent_id == parent_id]

    def find(self, name: str) -> Span | None:
        for s in reversed(self.spans):
            if s.name == name:
                return s
        return None

    def tree(self) -> dict[str, Any]:
        """Build a nested ``{name, span, children}`` view for the last root."""
        roots = [s for s in self.spans if s.parent_id is None]
        if not roots:
            return {}
        root = roots[-1]

        def build(span: Span) -> dict[str, Any]:
            return {
                "name": span.name,
                "span": span,
                "children": [build(c) for c in self.children(span.span_id)],
            }

        return build(root)


_span_counter: int = 0


def _new_span_id() -> str:
    """16-hex pseudo span id (OTel span ids are 16 hex chars).

    Uses a monotonic counter + os.urandom so that spans created in the same
    nanosecond still get unique ids (avoids self-referential cycles in the
    span tree).
    """
    global _span_counter
    _span_counter += 1
    # 8 hex from counter + 8 hex from random = 16 unique hex chars.
    return f"{_span_counter:08x}{os.urandom(4).hex()}"


class Tracer:
    """Domain-facing tracer. Spans auto-link to the active span as children.

    Usage::

        with tracer.start_as_current_span("ragx.query", trace_id=tid) as root:
            root.set_attributes({"ragx.kb_id": kb_id, "ragx.mode": "standard"})
            ...  # child spans auto-parent to ``root``
    """

    def __init__(
        self,
        exporter: InMemorySpanExporter | None = None,
        *,
        service_name: str = "ragx",
        enabled: bool = True,
    ) -> None:
        self.exporter = exporter or InMemorySpanExporter()
        self.service_name = service_name
        self.enabled = enabled

    def start_span(
        self,
        name: str,
        *,
        trace_id: str | None = None,
        parent: Span | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """Start a span. If ``parent`` is None, links to the current span."""
        if not self.enabled:
            span = Span(
                name=name,
                trace_id=trace_id or "",
                span_id=_new_span_id(),
                parent_id=parent.span_id if parent else None,
            )
            return span
        active = parent or _current_span.get()
        tid = trace_id or (active.trace_id if active else f"{time.time_ns():032x}")
        span = Span(
            name=name,
            trace_id=tid,
            span_id=_new_span_id(),
            parent_id=active.span_id if active else None,
            attributes=dict(attributes or {}),
        )
        if active:
            span.attributes.setdefault("trace_id", tid)
        return span

    def start_as_current_span(
        self,
        name: str,
        *,
        trace_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """Start a span and activate it in the current context."""
        span = self.start_span(name, trace_id=trace_id, attributes=attributes)
        span._token = _current_span.set(span)
        if self.enabled:
            original_end = span.end

            def _exporting_end() -> None:
                original_end()
                self.exporter.export(span)

            span.end = _exporting_end  # type: ignore[method-assign]
        return span

    # -- span attribute helpers (§10.1.3) ---------------------------------
    @staticmethod
    def span_query_root(
        trace_id: str,
        kb_id: str,
        mode: str,
        query: str,
    ) -> dict[str, Any]:
        """Attributes for the ``ragx.query`` root span."""
        return {
            "trace_id": trace_id,
            "ragx.kb_id": kb_id,
            "ragx.mode": mode,
            "ragx.query_length": len(query),
        }

    @staticmethod
    def span_llm(role: str, model: str, usage: Any, cost: float, **extra: Any) -> dict[str, Any]:
        """Attributes for a ``ragx.llm.<role>`` span (§10.1.3)."""
        attrs: dict[str, Any] = {
            "ragx.role": role,
            "ragx.model": model,
            "ragx.tokens": usage,
            "ragx.cost_usd": cost,
        }
        attrs.update(extra)
        return attrs


_tracer: Tracer | None = None


def get_tracer(
    exporter: InMemorySpanExporter | None = None,
    *,
    service_name: str = "ragx",
    enabled: bool = True,
) -> Tracer:
    """Process-wide tracer singleton (created on first call)."""
    global _tracer
    if _tracer is None:
        _tracer = Tracer(exporter, service_name=service_name, enabled=enabled)
    return _tracer


def reset_tracer() -> None:
    """Reset the global tracer (test helper)."""
    global _tracer
    _tracer = None
