"""Embedder contract suite against the built-in hash embedder (RX-SPI-02)."""

from __future__ import annotations

from ragx.spi.contracts.embedder import EmbedderContract
from tests.contract.dummy_plugins import DummyEmbedder


class TestHashEmbedder(EmbedderContract):
    async def make_embedder(self):
        return DummyEmbedder()
