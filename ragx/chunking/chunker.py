"""Chunker (04-chunking.md §4.8).

Joint chunking: visual hard bounds (page / heading / non-text atom) split first,
then text atoms are semantically soft-merged toward ``target_tokens``, then a
sentence-level overlap window is applied, then each group is assembled into a
:class:`Chunk`.
"""

from __future__ import annotations

import hashlib
from typing import Any

from ragx.core.exceptions import ChunkTooLargeError
from ragx.core.hashing import cosine_similarity
from ragx.core.ids import new_id
from ragx.core.models import Atom, AtomDescription, AtomType, Chunk
from ragx.core.settings import ChunkingOptions
from ragx.core.tokens import count_tokens


def is_hard_boundary(prev: Atom, curr: Atom) -> bool:
    """04-chunking.md §4.3.2."""
    if prev.page is not None and curr.page is not None and prev.page != curr.page:
        return True
    if curr.context and curr.context != prev.context:
        return True
    if curr.type != AtomType.TEXT:
        return True
    return False


def split_by_visual_bounds(atoms: list[Atom]) -> list[list[Atom]]:
    groups: list[list[Atom]] = []
    current: list[Atom] = []
    prev: Atom | None = None
    for atom in atoms:
        if prev is not None and is_hard_boundary(prev, atom):
            if current:
                groups.append(current)
            current = []
        current.append(atom)
        prev = atom
    if current:
        groups.append(current)
    return groups


def merge_text_atoms(
    text_atoms: list[Atom],
    opts: ChunkingOptions,
    vectors: list[list[float]],
) -> list[list[Atom]]:
    """Semantic soft-merge (04-chunking.md §4.3.3). ``vectors`` are the
    pre-computed embeddings of ``text_atoms`` in the same order."""
    if not text_atoms:
        return []
    groups: list[list[Atom]] = []
    current: list[Atom] = [text_atoms[0]]
    current_tokens = count_tokens(text_atoms[0].text or "")

    for i in range(1, len(text_atoms)):
        atom = text_atoms[i]
        atom_tokens = count_tokens(atom.text or "")
        sim = cosine_similarity(vectors[i - 1], vectors[i])
        should_split = (
            current_tokens + atom_tokens > opts.max_tokens
            or (current_tokens >= opts.target_tokens and sim < opts.semantic_threshold)
            or sim < opts.semantic_threshold * 0.7
        )
        if should_split:
            groups.append(current)
            current = []
            current_tokens = 0
        current.append(atom)
        current_tokens += atom_tokens

    if current:
        groups.append(current)

    if opts.merge_short_chunks:
        groups = merge_short(groups, opts.min_tokens)
    return groups


def merge_short(groups: list[list[Atom]], min_tokens: int) -> list[list[Atom]]:
    """Merge chunks shorter than ``min_tokens`` into the previous one."""
    if not groups:
        return groups
    out: list[list[Atom]] = [groups[0]]
    for group in groups[1:]:
        tokens = sum(count_tokens(a.text or "") for a in group)
        if tokens < min_tokens and out:
            out[-1] = out[-1] + group
        else:
            out.append(group)
    return out


def split_sentences(text: str) -> list[str]:
    """Split into sentences on CJK/ASCII sentence terminators."""
    import re

    parts = re.split(r"(?<=[。！？!?；;])", text.strip())
    return [p for p in parts if p.strip()]


def extract_tail_sentences(atoms: list[Atom], target_tokens: int) -> list[Atom]:
    """04-chunking.md §4.3.4 - sentence-level overlap tail."""
    collected: list[Atom] = []
    total = 0
    for atom in reversed(atoms):
        text = atom.text or ""
        for sent in reversed(split_sentences(text)):
            t = count_tokens(sent)
            if total + t > target_tokens:
                return collected
            collected.insert(
                0,
                Atom(
                    atom_id=f"{atom.atom_id}#ov",
                    doc_id=atom.doc_id,
                    type=AtomType.TEXT,
                    text=sent,
                    page=atom.page,
                    bbox=atom.bbox,
                    content_hash=hashlib.sha256(sent.encode("utf-8")).hexdigest(),
                    context=atom.context,
                ),
            )
            total += t
    return collected


def apply_overlap(groups: list[list[Atom]], opts: ChunkingOptions) -> list[list[Atom]]:
    overlap_tokens = int(opts.target_tokens * opts.overlap_ratio)
    result: list[list[Atom]] = []
    for i, group in enumerate(groups):
        if i == 0:
            result.append(group)
            continue
        tail = extract_tail_sentences(groups[i - 1], overlap_tokens)
        result.append(tail + group)
    return result


def extract_table_header(table_md: str) -> str:
    """04-chunking.md §4.4.3 - first row + separator row."""
    lines = table_md.strip().split("\n")
    if len(lines) >= 2 and "|" in lines[0] and set(lines[1].strip()) <= {"|", "-", ":", " "}:
        return "\n".join(lines[:2])
    return ""


def assemble_chunk(
    atoms: list[Atom],
    kb_id: str,
    doc_id: str,
    descriptions: dict[str, AtomDescription],
    opts: ChunkingOptions,
) -> Chunk:
    """04-chunking.md §4.3.5."""
    parts: list[str] = []
    if opts.context_prefix:
        ctx = next((a.context for a in atoms if a.context), None)
        if ctx:
            parts.append(f"[{ctx}]")

    for atom in atoms:
        if atom.type == AtomType.TEXT:
            parts.append(atom.text or "")
        elif atom.type == AtomType.TABLE:
            desc = descriptions.get(atom.atom_id)
            table_md = atom.text or ""
            if opts.keep_table_header:
                header = extract_table_header(table_md)
                if header:
                    parts.append(header)
            if desc:
                parts.append(desc.description)
            parts.append(table_md)
        elif atom.type == AtomType.FORMULA:
            desc = descriptions.get(atom.atom_id)
            if desc:
                parts.append(f"[公式描述] {desc.description}")
            if atom.text:
                parts.append(f"$${atom.text}$$")
        elif atom.type == AtomType.IMAGE:
            desc = descriptions.get(atom.atom_id)
            if desc:
                parts.append(f"[图片] {desc.description}")
            else:
                parts.append("[图片]")

    text = "\n".join(parts)
    token_count = count_tokens(text)
    if token_count > opts.max_tokens * 1.2:
        raise ChunkTooLargeError(
            code=3003,
            message="chunk exceeds hard token cap and cannot be split",
            details={"token_count": token_count, "max_tokens": opts.max_tokens},
        )
    return Chunk(
        chunk_id=new_id("chk_"),
        doc_id=doc_id,
        kb_id=kb_id,
        atom_ids=[a.atom_id for a in atoms],
        text=text,
        token_count=token_count,
        page=atoms[0].page,
        bbox=atoms[0].bbox,
        metadata={},
        edited=False,
        version=1,
    )


class Chunker:
    """Joint chunking orchestrator (04-chunking.md §4.8)."""

    def __init__(self, opts: ChunkingOptions, embedder: Any | None = None) -> None:
        self.opts = opts
        self.embedder = embedder

    async def chunk(
        self,
        atoms: list[Atom],
        kb_id: str,
        doc_id: str,
        descriptions: dict[str, AtomDescription] | None = None,
    ) -> list[Chunk]:
        descriptions = descriptions or {}
        visual_groups = split_by_visual_bounds(atoms)

        chunks: list[Chunk] = []
        for group in visual_groups:
            non_text = [a for a in group if a.type != AtomType.TEXT]
            text_atoms = [a for a in group if a.type == AtomType.TEXT]

            for atom in non_text:
                chunks.append(assemble_chunk([atom], kb_id, doc_id, descriptions, self.opts))

            if text_atoms:
                vectors = await self._embed(text_atoms)
                merged = merge_text_atoms(text_atoms, self.opts, vectors)
                merged = apply_overlap(merged, self.opts)
                for mg in merged:
                    chunks.append(assemble_chunk(mg, kb_id, doc_id, descriptions, self.opts))
        return chunks

    async def _embed(self, atoms: list[Atom]) -> list[list[float]]:
        if self.embedder is None:
            # no embedder -> identical vectors, so semantic split never fires
            return [[1.0, 0.0]] * len(atoms)
        return await self.embedder.embed([a.text or "" for a in atoms])
