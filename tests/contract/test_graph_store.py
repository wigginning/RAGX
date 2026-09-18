"""GraphStore contract suite against the built-in NetworkX store (RX-SPI-02)."""

from __future__ import annotations

from ragx.spi.contracts.graph_store import GraphStoreContract
from tests.contract.dummy_plugins import DummyGraphStore


class TestNetworkXGraphStore(GraphStoreContract):
    async def make_graph_store(self):
        return DummyGraphStore()
