"""VectorStore contract suite against the built-in SQLite store (RX-SPI-02)."""

from __future__ import annotations

from ragx.spi.contracts.vector_store import VectorStoreContract
from tests.contract.dummy_plugins import DummyVectorStore


class TestSQLiteVectorStore(VectorStoreContract):
    DIMENSION = 8

    async def make_store(self):
        return DummyVectorStore()
