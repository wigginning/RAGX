"""Reranker contract suite against the deterministic overlap reranker (RX-SPI-02)."""

from __future__ import annotations

from ragx.spi.contracts.reranker import RerankerContract
from tests.contract.dummy_plugins import DummyReranker


class TestDummyReranker(RerankerContract):
    async def make_reranker(self):
        return DummyReranker()
