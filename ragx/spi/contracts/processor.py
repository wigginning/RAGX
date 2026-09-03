"""Processor contract (01-spi.md §1.4) - atom descriptions."""

from __future__ import annotations

import pytest

from ragx.core.models import Atom, AtomType
from ragx.spi.contracts.base import ContractBase
from ragx.spi.interfaces import DescribeOptions, Processor


class ProcessorContract(ContractBase):
    async def make(self) -> Processor:
        return await self.make_processor()

    async def make_processor(self) -> Processor:  # pragma: no cover
        raise NotImplementedError

    @staticmethod
    def _atoms(doc_id: str = "doc_t") -> list[Atom]:
        return [
            Atom(
                atom_id=f"{doc_id}#{i:04d}",
                doc_id=doc_id,
                type=AtomType.IMAGE,
                payload_ref=f"img-{i}.png",
                content_hash=f"h{i}",
                context=f"章节 {i}",
            )
            for i in range(3)
        ]

    @pytest.mark.contract
    async def test_supported_atom_types_declared(self) -> None:
        proc = await self._get()
        assert proc.supported_atom_types, "processor must declare supported atom types"
        assert all(isinstance(t, AtomType) for t in proc.supported_atom_types)
        assert isinstance(proc.name, str) and proc.name

    @pytest.mark.contract
    async def test_describe_covers_every_atom(self) -> None:
        """Batch contract: exactly one description per input atom, same order (01-spi §1.2)."""
        proc = await self._get()
        atoms = self._atoms()
        out = await proc.describe(atoms, options=DescribeOptions())
        assert len(out) == len(atoms)
        assert [d.atom_id for d in out] == [a.atom_id for a in atoms]

    @pytest.mark.contract
    async def test_description_carries_confidence_and_model(self) -> None:
        """``confidence`` and ``model`` are mandatory - they drive the cost gates."""
        proc = await self._get()
        out = await proc.describe(self._atoms(), options=DescribeOptions())
        for desc in out:
            assert desc.description.strip(), "description must be non-empty"
            assert 0.0 <= desc.confidence <= 1.0, desc.confidence
            assert desc.model, "model is required for cost attribution"

    @pytest.mark.contract
    async def test_describe_is_idempotent(self) -> None:
        proc = await self._get()
        first = await proc.describe(self._atoms(), options=DescribeOptions())
        second = await proc.describe(self._atoms(), options=DescribeOptions())
        assert [d.description for d in first] == [d.description for d in second]

    @pytest.mark.contract
    async def test_describe_empty_batch(self) -> None:
        proc = await self._get()
        assert await proc.describe([], options=DescribeOptions()) == []
