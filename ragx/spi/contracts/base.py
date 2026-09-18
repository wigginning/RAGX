"""Shared helpers for the contract suites."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pytest

#: Small shared fixture corpus used by parser / embedder / reranker suites.
SAMPLE_MARKDOWN = """# 第一章 标题

这是一个关于检索增强生成的段落。RAGX 是一个分层插件化的平台。

## 1.1 子标题

第二段文本，包含关键实体 LightRAG 与 LightRAG 双层检索。

| 名称 | 值 |
| --- | --- |
| a | 1 |
| b | 2 |
"""

SAMPLE_TEXTS: tuple[str, ...] = (
    "检索增强生成通过向量检索提升问答质量。",
    "今天天气很好，适合出去散步。",
    "Python 是一种流行的编程语言，支持面向对象。",
    "LightRAG 的双层检索分为 Low-Level 与 High-Level。",
    "表格中的数据表明 2023 年的成本下降了 30%。",
    "公式 E=mc^2 表达了质量与能量的等价关系。",
    "知识图谱由实体与关系组成，用于结构化推理。",
    "Agentic 管线可以分解复杂问题并并行执行子任务。",
)


class ContractBase(ABC):
    """Instance caching + shared assertions for contract suites.

    Note: this class deliberately defines **no** ``__init__``. pytest refuses to
    collect a test class whose ``__init__`` is not ``object.__init__`` (it checks
    the resolved attribute, so an inherited one also disqualifies the subclass).
    The plugin instance is therefore cached on a class-level slot instead.
    """

    _plugin: Any = None

    @abstractmethod
    async def make(self) -> Any:
        """Create (and cache) one plugin instance per test session."""

    async def _get(self) -> Any:
        if self._plugin is None:
            self._plugin = await self.make()
        return self._plugin

    @staticmethod
    def _skip_if_absent(plugin: Any, capability: str, reason: str) -> None:
        value = getattr(plugin, capability, True)
        if not value:
            pytest.skip(f"{reason} ({capability}=False)")
