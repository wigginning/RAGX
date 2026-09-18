"""Quota middleware tests (RX-API-03)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from ragx.api.middleware.quota import (
    InMemoryQuotaStore,
    MetadataQuotaStore,
    QuotaUsage,
)
from ragx.ingestion.store import MetadataStore


async def test_in_memory_store_accumulates_within_period() -> None:
    store = InMemoryQuotaStore()
    # Use a unique tenant to avoid bleed-over from other tests.
    await store.add_tokens("tenant-a", 10)
    snap1 = await store.usage("tenant-a")
    await store.add_tokens("tenant-a", 5)
    snap2 = await store.usage("tenant-a")
    assert snap1.tokens == 10
    assert snap2.tokens == 15


async def test_in_memory_store_resets_on_period_change() -> None:
    store = InMemoryQuotaStore()
    last_month = (datetime.now(UTC).replace(day=1) - timedelta(days=1)).isoformat()
    store._usage["default"] = QuotaUsage(tokens=999, period_start=last_month)  # type: ignore[attr-defined]
    usage = await store.add_tokens("default", 5)
    assert usage.tokens == 5


async def test_in_memory_upload_separate_counter() -> None:
    store = InMemoryQuotaStore()
    await store.add_upload("default", 1024)
    await store.add_tokens("default", 100)
    u = await store.usage("default")
    assert u.upload_bytes == 1024
    assert u.tokens == 100


async def test_metadata_store_persists_across_instances() -> None:
    db = MetadataStore(":memory:")
    await db.connect()
    store = MetadataQuotaStore(db)
    u = await store.add_tokens("acme", 42)
    assert u.tokens == 42
    # New instance, same DB — counters survive.
    store2 = MetadataQuotaStore(db)
    u2 = await store2.usage("acme")
    assert u2.tokens == 42
    await db.close()


async def test_metadata_quota_bump_returns_updated_usage() -> None:
    db = MetadataStore(":memory:")
    await db.connect()
    store = MetadataQuotaStore(db)
    await store.add_tokens("acme", 10)
    u = await store.add_upload("acme", 100)
    assert u.tokens == 10
    assert u.upload_bytes == 100
    await db.close()


def test_metadata_quota_rejects_unknown_field_in_metadata_store() -> None:
    """``MetadataStore.bump_quota`` itself rejects unknown columns."""

    async def go() -> None:
        db = MetadataStore(":memory:")
        await db.connect()
        # Manually call with bad field through bump_quota (the underlying API).
        with pytest.raises(ValueError):
            await db.bump_quota("default", "bogus_field", 1, "2026-01-01")
        await db.close()

    asyncio.run(go())
