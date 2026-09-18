"""Neo4jGraphStore contract suite (RX-PLG-03 DoD: ``pytest tests/contract -k neo4j``).

Requires a running Neo4j (see ``deploy/compose/test.yml``, service ``neo4j``).
When Neo4j is not reachable the suite is skipped.

Each test method creates a fresh store instance, but the underlying Neo4j
database is shared — so ``make_graph_store`` wipes the ``kb_contract`` subgraph
first to give every test isolated data (mirrors the in-memory NetworkX store,
which is naturally per-test).
"""

from __future__ import annotations

import os
import urllib.request

import pytest

from ragx.plugins.graph_neo4j import Neo4jGraphStore
from ragx.spi.contracts.graph_store import GraphStoreContract

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:17687")
NEO4J_AUTH = os.environ.get("NEO4J_AUTH", "neo4j/12345678")
# The store under test resolves credentials from this env var; ensure it is
# set so the suite passes without the caller exporting NEO4J_AUTH itself.
os.environ.setdefault("NEO4J_AUTH", NEO4J_AUTH)


def _neo4j_reachable() -> bool:
    try:
        with urllib.request.urlopen("http://localhost:7474", timeout=2.0) as resp:
            return resp.status == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _neo4j_reachable(), reason="Neo4j not reachable")


def _split_auth(auth_str: str) -> tuple[str, str]:
    sep = "/" if "/" in auth_str else (":" if ":" in auth_str else None)
    if sep is None:
        return ("neo4j", auth_str)
    user, password = auth_str.split(sep, 1)
    return (user, password)


class TestNeo4jGraphStore(GraphStoreContract):
    async def make_graph_store(self):
        # Wipe the shared kb subgraph so each test starts from an empty store.
        from neo4j import AsyncGraphDatabase

        user, password = _split_auth(NEO4J_AUTH)
        driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(user, password))
        try:
            async with driver.session(database="neo4j") as session:
                await (await session.run(
                    "MATCH (e:Entity {kb_id: $kb}) DETACH DELETE e",
                    kb="kb_contract",
                )).consume()
        finally:
            await driver.close()

        return Neo4jGraphStore({
            "uri": NEO4J_URI,
            "auth_env": "NEO4J_AUTH",
            "database": "neo4j",
            "kb_id": "kb_contract",
            "dim": 8,
        })
