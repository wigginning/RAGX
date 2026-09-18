"""ESVectorStore contract suite (RX-PLG-02 DoD: ``pytest tests/contract -k es``).

Requires a running Elasticsearch (see ``deploy/compose/test.yml``, service
``es``). When ES is not reachable the suite is skipped, not failed, so the
default test run stays green without external services.
"""

from __future__ import annotations

import urllib.request

import pytest

from ragx.plugins.vector_es import ESVectorStore
from ragx.spi.contracts.vector_store import VectorStoreContract


def _es_reachable() -> bool:
    try:
        with urllib.request.urlopen("http://localhost:9200", timeout=2.0) as resp:
            return resp.status == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _es_reachable(), reason="ES not reachable")


class TestESVectorStore(VectorStoreContract):
    DIMENSION = 8

    async def make_store(self):
        # Wipe the shared index so each test starts from an empty store
        # (mirrors the Neo4j suite's per-test subgraph wipe).
        from elasticsearch import AsyncElasticsearch

        client = AsyncElasticsearch(hosts=["http://localhost:9200"])
        try:
            if await client.indices.exists(index="ragx_test_kb_contract"):
                await client.indices.delete(index="ragx_test_kb_contract")
        finally:
            await client.close()

        return ESVectorStore({
            "hosts": ["http://localhost:9200"],
            "index_prefix": "ragx_test",
            "dim": 8,
            "kb_id": "kb_contract",
        })
