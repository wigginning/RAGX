"""Chunking layer (04-chunking.md)."""

from ragx.chunking.chunker import (
    Chunker,
    apply_overlap,
    assemble_chunk,
    extract_table_header,
    extract_tail_sentences,
    is_hard_boundary,
    merge_short,
    merge_text_atoms,
    split_by_visual_bounds,
    split_sentences,
)

__all__ = [
    "Chunker",
    "apply_overlap",
    "assemble_chunk",
    "extract_table_header",
    "extract_tail_sentences",
    "is_hard_boundary",
    "merge_short",
    "merge_text_atoms",
    "split_by_visual_bounds",
    "split_sentences",
]
