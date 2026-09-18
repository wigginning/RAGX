"""QdrantVectorStore contract suite (RX-PLG-02 DoD: ``pytest tests/contract -k qdrant``).

Requires a running Qdrant (see ``deploy/compose/test.yml``, service
``qdrant``). When Qdrant is not reachable the suite is skipped.
"""

from __future__ import annotations

import urllib.request

import pytest

from ragx.plugins.vector_qdrant import QdrantVectorStore
from ragx.spi.contracts.vector_store import VectorStoreContract


def _qdrant_reachable() -> bool:
    try:
        with urllib.request.urlopen("http://localhost:6333", timeout=2.0) as resp:
            return resp.status == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _qdrant_reachable(), reason="Qdrant not reachable")


class TestQdrantVectorStore(VectorStoreContract):
    DIMENSION = 8

    async def make_store(self):
        # Wipe the shared collection so each test starts from an empty store.
        from qdrant_client import AsyncQdrantClient

        client = AsyncQdrantClient(url="http://localhost:6333")
        try:
            collections = await client.get_collections()
            if any(c.name == "ragx_test_kb_contract" for c in collections.collections):
                await client.delete_collection(collection_name="ragx_test_kb_contract")
        finally:
            await client.close()

        return QdrantVectorStore({
            "url": "http://localhost:6333",
            "collection": "ragx_test",
            "dim": 8,
            "kb_id": "kb_contract",
        })
