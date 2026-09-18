"""Three-level VLM cost gate (03-ingestion.md §3.4).

Level 1: ``atom.content_hash`` description cache (cross-document reuse).
Level 2: batched + bounded-concurrency VLM calls.
Level 3: small-model first, confidence-based escalation to the large model,
capped by ``max_upgrade_ratio``.

``vlm_enabled=false`` short-circuits to no descriptions (03-ingestion.md §3.4.4).
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel

from ragx.core.models import Atom, AtomDescription, AtomType
from ragx.spi.interfaces import DescribeOptions, Processor


class BatchConfig(BaseModel):
    batch_window: float = 2.0
    max_batch_size: int = 16
    max_concurrent_batches: int = 4


class DegradationConfig(BaseModel):
    small_model: str = "vlm-small"
    large_model: str = "vlm-large"
    confidence_threshold: float = 0.75
    max_upgrade_ratio: float = 0.3


class _InMemoryDescCache:
    """Minimal content-hash description cache (in-process; a Redis backend can
    replace it later without changing the gate logic)."""

    def __init__(self) -> None:
        self._store: dict[str, AtomDescription] = {}

    async def get(self, content_hash: str) -> AtomDescription | None:
        return self._store.get(content_hash)

    async def set(self, content_hash: str, desc: AtomDescription) -> None:
        self._store[content_hash] = desc

    async def exists(self, content_hash: str) -> bool:
        return content_hash in self._store


async def describe_with_cache(
    atom: Atom,
    processor: Processor,
    cache: _InMemoryDescCache,
    opts: DescribeOptions,
) -> AtomDescription:
    """Level 1: reuse a cached description keyed by ``content_hash``."""
    cached = await cache.get(atom.content_hash)
    if cached is not None:
        cached.cached = True
        return cached
    desc = (await processor.describe([atom], options=opts))[0]
    await cache.set(atom.content_hash, desc)
    return desc


async def batch_describe(
    atoms: list[Atom],
    processor: Processor,
    cache: _InMemoryDescCache,
    opts: DescribeOptions,
    cfg: BatchConfig | None = None,
) -> list[AtomDescription]:
    """Level 2: filter cache hits, batch + bound concurrency, merge in order."""
    cfg = cfg or BatchConfig()
    to_process = [a for a in atoms if not await cache.exists(a.content_hash)]
    results: dict[str, AtomDescription] = {}
    by_id = {a.atom_id: a for a in atoms}

    sem = asyncio.Semaphore(cfg.max_concurrent_batches)
    batches = [
        to_process[i : i + cfg.max_batch_size]
        for i in range(0, len(to_process), cfg.max_batch_size)
    ]

    async def run_batch(batch: list[Atom]) -> None:
        async with sem:
            descs = await processor.describe(batch, options=opts)
        for d in descs:
            results[d.atom_id] = d
            await cache.set(by_id[d.atom_id].content_hash, d)

    await asyncio.gather(*(run_batch(b) for b in batches))

    return [
        results.get(a.atom_id) or (await cache.get(a.content_hash)) or _fallback(a)
        for a in atoms
    ]


def _fallback(atom: Atom) -> AtomDescription:
    return AtomDescription(
        atom_id=atom.atom_id,
        description=atom.text or "",
        confidence=1.0 if atom.text else 0.0,
        model="passthrough",
    )


async def describe_with_degradation(
    atoms: list[Atom],
    small_proc: Processor,
    large_proc: Processor,
    cache: _InMemoryDescCache,
    opts: DescribeOptions,
    batch_cfg: BatchConfig | None = None,
    degr_cfg: DegradationConfig | None = None,
) -> list[AtomDescription]:
    """Level 3: small model first, escalate low-confidence atoms to the large
    model, capped by ``max_upgrade_ratio`` (03-ingestion.md §3.4.3)."""
    degr_cfg = degr_cfg or DegradationConfig()
    descs = await batch_describe(atoms, small_proc, cache, opts, batch_cfg)

    low_conf = [d for d in descs if d.confidence < degr_cfg.confidence_threshold]
    upgrade_n = min(len(low_conf), int(len(atoms) * degr_cfg.max_upgrade_ratio))
    to_upgrade = low_conf[:upgrade_n]
    if not to_upgrade:
        return descs

    by_id = {a.atom_id: a for a in atoms}
    upgraded = await batch_describe(
        [by_id[d.atom_id] for d in to_upgrade], large_proc, cache, opts, batch_cfg
    )
    upgraded_map = {d.atom_id: d for d in upgraded}
    return [
        upgraded_map.get(d.atom_id, d) if d.atom_id in upgraded_map else d
        for d in descs
    ]


async def process_atoms(
    atoms: list[Atom],
    *,
    vlm_enabled: bool,
    small_proc: Processor | None,
    large_proc: Processor | None,
    cache: _InMemoryDescCache,
    opts: DescribeOptions,
    batch_cfg: BatchConfig | None = None,
    degr_cfg: DegradationConfig | None = None,
) -> list[AtomDescription]:
    """Entry point: skip entirely when VLM is off (03-ingestion.md §3.4.4)."""
    if not vlm_enabled:
        return []
    non_text = [a for a in atoms if a.type != AtomType.TEXT]
    if not non_text:
        return []
    if small_proc is None:
        return []
    return await describe_with_degradation(
        non_text, small_proc, large_proc or small_proc, cache, opts,
        batch_cfg, degr_cfg,
    )
