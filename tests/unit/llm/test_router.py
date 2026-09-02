"""ResilientRouter full-path tests (RX-LLM-01 DoD).

Covers: retry x3 -> fallback -> circuit open -> 6001 / 6002, plus the
non-retryable 4xx path and the budget gate.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ragx.core.exceptions import (
    AllProvidersFailedError,
    CircuitOpenError,
    LLMError,
    RateLimitError,
)
from ragx.core.models import TokenUsage
from ragx.core.roles import LLMRole
from ragx.core.settings import (
    CircuitConfig,
    LLMRouterConfig,
    ProviderTarget,
    RoleConfig,
)
from ragx.llm.router import ResilientRouter
from ragx.spi.interfaces import (
    ChatChunk,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    LLMCapabilities,
)
from ragx.spi.registry import PluginRegistry


class MockProvider:
    """A provider whose behaviour is scripted per instance."""

    name = "mock"
    capabilities = LLMCapabilities(supports_json_mode=True, supports_stream=True)

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.failures: list[Exception] = list(cfg.get("failures", []))
        self.fail_forever: Exception | None = cfg.get("fail_forever")
        self.calls = 0
        self.text = cfg.get("text", "mock answer")

    async def chat(self, req: ChatRequest) -> ChatResponse:
        self.calls += 1
        if self.fail_forever is not None:
            raise self.fail_forever
        if self.failures:
            exc = self.failures.pop(0)
            raise exc
        return ChatResponse(
            text=self.text,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total=15),
            model=req.model or self.name,
        )

    async def chat_stream(self, req: ChatRequest):
        yield ChatChunk(delta="x")

    async def structured(self, req: ChatRequest, *, schema):
        return schema()

    async def startup(self):
        return None

    async def shutdown(self):
        return None


def _req(role: LLMRole = LLMRole.GENERATE, **kw) -> ChatRequest:
    return ChatRequest(
        messages=[ChatMessage(role="user", content="测试问题")],
        role=role,
        model="m",
        kb_id="kb_t",
        trace_id="trace_t",
        **kw,
    )


def _registry(
    providers: dict[str, list[Exception]] | None = None,
    fail_forever: dict[str, Exception] | None = None,
) -> PluginRegistry:
    reg = PluginRegistry()
    for name, failures in (providers or {}).items():
        reg.register("llm_provider", name, lambda cfg, f=failures: MockProvider({"failures": list(f)}))
    for name, exc in (fail_forever or {}).items():
        reg.register("llm_provider", name, lambda cfg, e=exc: MockProvider({"fail_forever": e}))
    return reg


def _router(
    primary: ProviderTarget,
    fallbacks: list[ProviderTarget] | None = None,
    *,
    max_retries: int = 3,
    providers: dict[str, list[Exception]] | None = None,
    fail_forever: dict[str, Exception] | None = None,
    circuit: CircuitConfig | None = None,
) -> ResilientRouter:
    cfg = LLMRouterConfig(
        roles={
            LLMRole.GENERATE: RoleConfig(
                primary=primary,
                fallbacks=fallbacks or [],
                max_retries=max_retries,
            )
        },
        circuit=circuit or CircuitConfig(min_samples=2, failure_rate=0.5, open_s=0.05),
    )
    reg = _registry(providers or {}, fail_forever)
    return ResilientRouter(cfg, reg)


async def test_retry_then_success() -> None:
    """Two transient failures then success -> retries, no fallback, success."""
    router = _router(
        ProviderTarget(provider="a", model="m"),
        providers={"a": [RateLimitError("rl", code=1004), LLMError("5xx", details={"retryable": True})]},
    )
    resp = await router.chat(_req())
    assert resp.text == "mock answer"
    assert resp.usage.total == 15


async def test_retry_exhausted_then_fallback() -> None:
    """Primary exhausts retries -> fallback succeeds."""
    router = _router(
        ProviderTarget(provider="a", model="m"),
        fallbacks=[ProviderTarget(provider="b", model="m")],
        providers={
            "a": [RateLimitError("rl", code=1004)] * 4,  # max_retries=3 -> 4 attempts
            "b": [],
        },
    )
    resp = await router.chat(_req())
    assert resp.text == "mock answer"


async def test_all_providers_failed_6001() -> None:
    """Every candidate fails -> AllProvidersFailedError(6001)."""
    router = _router(
        ProviderTarget(provider="a", model="m"),
        fallbacks=[ProviderTarget(provider="b", model="m")],
        providers={
            "a": [RateLimitError("rl", code=1004)] * 4,
            "b": [LLMError("5xx", details={"retryable": True})] * 4,
        },
    )
    with pytest.raises(AllProvidersFailedError) as ei:
        await router.chat(_req())
    assert ei.value.code == 6001


async def test_circuit_open_6002() -> None:
    """After enough failures the circuit opens -> fast-fail 6002."""
    router = _router(
        ProviderTarget(provider="a", model="m"),
        fail_forever={"a": RateLimitError("rl", code=1004)},
        circuit=CircuitConfig(min_samples=2, failure_rate=0.5, open_s=60),
    )
    # two failing calls record two failure samples -> circuit opens
    with pytest.raises(AllProvidersFailedError):
        await router.chat(_req())
    with pytest.raises(AllProvidersFailedError):
        await router.chat(_req())
    # third call: circuit is open -> 6002
    with pytest.raises(CircuitOpenError) as ei:
        await router.chat(_req())
    assert ei.value.code == 6002


async def test_circuit_half_open_recovers() -> None:
    """After open_s elapses, a probe succeeds and the circuit closes."""
    router = _router(
        ProviderTarget(provider="a", model="m"),
        # 8 failures = 2 calls x 4 attempts; the 9th (probe) succeeds
        providers={"a": [RateLimitError("rl", code=1004)] * 8},
        circuit=CircuitConfig(min_samples=2, failure_rate=0.5, open_s=0.05),
    )
    with pytest.raises(AllProvidersFailedError):
        await router.chat(_req())
    with pytest.raises(AllProvidersFailedError):
        await router.chat(_req())
    with pytest.raises(CircuitOpenError):
        await router.chat(_req())
    await asyncio.sleep(0.06)  # let open_s elapse -> half_open probe
    resp = await router.chat(_req())
    assert resp.text == "mock answer"


async def test_non_retryable_4xx_goes_to_fallback() -> None:
    """A non-retryable 4xx is not retried; it moves to the fallback."""
    router = _router(
        ProviderTarget(provider="a", model="m"),
        fallbacks=[ProviderTarget(provider="b", model="m")],
        providers={
            "a": [LLMError("4xx", details={"retryable": False})],
            "b": [],
        },
    )
    resp = await router.chat(_req())
    assert resp.text == "mock answer"


async def test_budget_cap_rejects_over_budget() -> None:
    """Daily cap reached -> RateLimitError(1004) for a non-cacheable query."""
    router = _router(ProviderTarget(provider="a", model="m"), providers={"a": []})
    router.budget_caps["kb_t"] = 0.0  # cap already reached
    with pytest.raises(RateLimitError) as ei:
        await router.chat(_req())
    assert ei.value.code == 1004
