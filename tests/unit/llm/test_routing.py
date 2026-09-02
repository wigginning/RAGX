"""Multi-provider routing strategy tests (08-llm.md §8.2.4, RX-LLM-01 DoD)."""

from __future__ import annotations

from ragx.core.roles import LLMRole
from ragx.core.settings import (
    CircuitConfig,
    LLMRouterConfig,
    ProviderTarget,
    RoleConfig,
)
from ragx.llm.router import ResilientRouter
from ragx.spi.registry import PluginRegistry


def _router(role: RoleConfig, pricing: dict | None = None) -> ResilientRouter:
    cfg = LLMRouterConfig(
        roles={LLMRole.GENERATE: role},
        circuit=CircuitConfig(min_samples=2, failure_rate=0.5, open_s=60),
        pricing=pricing or {},
    )
    return ResilientRouter(cfg, PluginRegistry())


def test_priority_orders_primary_then_fallbacks() -> None:
    role = RoleConfig(
        primary=ProviderTarget(provider="a", model="m"),
        fallbacks=[ProviderTarget(provider="b", model="m")],
    )
    targets = _router(role)._targets(role)
    assert [t.provider for t in targets] == ["a", "b"]


def test_weighted_returns_only_candidates() -> None:
    role = RoleConfig(
        primary=ProviderTarget(provider="a", model="m"),
        strategy="weighted",
        candidates=[
            ProviderTarget(provider="a", model="m"),
            ProviderTarget(provider="b", model="m"),
        ],
        weights={"a/m": 0.7, "b/m": 0.3},
    )
    router = _router(role)
    for _ in range(50):
        targets = router._targets(role)
        assert targets, "weighted must return at least one candidate"
        assert targets[0].provider in {"a", "b"}


def test_weighted_excludes_open_candidates() -> None:
    role = RoleConfig(
        primary=ProviderTarget(provider="a", model="m"),
        strategy="weighted",
        candidates=[
            ProviderTarget(provider="a", model="m"),
            ProviderTarget(provider="b", model="m"),
        ],
        weights={"a/m": 0.7, "b/m": 0.3},
    )
    router = _router(role)
    router.circuit.record_failure(("a", "m"))
    router.circuit.record_failure(("a", "m"))  # open a
    for _ in range(20):
        targets = router._targets(role)
        assert targets[0].provider == "b", "open candidate must be excluded"


def test_least_cost_picks_cheapest() -> None:
    role = RoleConfig(
        primary=ProviderTarget(provider="a", model="m"),
        strategy="least_cost",
        candidates=[
            ProviderTarget(provider="a", model="m"),
            ProviderTarget(provider="b", model="m"),
        ],
    )
    router = _router(
        role,
        pricing={
            "a/m": {"prompt_per_1k": 0.01, "completion_per_1k": 0.02},
            "b/m": {"prompt_per_1k": 0.001, "completion_per_1k": 0.002},
        },
    )
    targets = router._targets(role)
    assert targets[0].provider == "b", "cheapest candidate must be first"


def test_least_cost_skips_open_cheapest() -> None:
    role = RoleConfig(
        primary=ProviderTarget(provider="a", model="m"),
        strategy="least_cost",
        candidates=[
            ProviderTarget(provider="a", model="m"),
            ProviderTarget(provider="b", model="m"),
        ],
    )
    router = _router(
        role,
        pricing={
            "a/m": {"prompt_per_1k": 0.001, "completion_per_1k": 0.002},
            "b/m": {"prompt_per_1k": 0.01, "completion_per_1k": 0.02},
        },
    )
    router.circuit.record_failure(("a", "m"))
    router.circuit.record_failure(("a", "m"))  # open the cheapest
    targets = router._targets(role)
    assert targets[0].provider == "b"
