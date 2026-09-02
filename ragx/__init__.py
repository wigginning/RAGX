"""RAGX - layered, plugin-based Retrieval-Augmented Generation platform.

Layering (00-overview.md §0.1):

    api/          L4 access layer
    agentic/ retrieval/ ingestion/ chunking/ kg/   L3 domain services
    llm/ observability/                              L2 cross-cutting
    spi/ plugins/                                    L1 contracts + plugins
    core/                                            L0 foundation
"""

from __future__ import annotations

__version__ = "0.1.0"

#: SPI compatibility range declared by this distribution (01-spi.md §1.5).
ragx_spi_version = ">=1,<2"
