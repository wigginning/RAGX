"""VLMDescribeProcessor (11-plugins-builtin.md §11.2.7).

Produces :class:`AtomDescription` for image / table / formula atoms by calling
a multimodal model **through the Resilient Model Router** - the plugin never
instantiates a provider client (00-overview.md §0.2).

Wiring (composition stays in the app factory, so this L1 plugin keeps its
"only spi + core + third-party" dependency rule):

    processor = VLMDescribeProcessor({
        "chat": router.chat_describe,      # async (msgs, role=...) -> str
        "prompts": prompt_registry,        # optional: get_prompt(name) -> str
        "batch_size": 8,
        "confidence_threshold": 0.6,
        "small_model": "vlm-small",
        "large_model": "vlm-large",
    })

The three-level cost gate (description cache -> batching -> confidence
escalation) lives in ``ragx.ingestion.costs`` (03-ingestion.md §3.4); this
plugin only performs one describe call per batch.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel

from ragx.core.exceptions import ConfigError
from ragx.core.models import Atom, AtomDescription, AtomType
from ragx.spi.interfaces import DescribeOptions

_NAME = "vlm"

#: (system, user) default templates; overridden by the PromptRegistry when wired.
_DEFAULT_PROMPTS: dict[str, tuple[str, str]] = {
    AtomType.IMAGE: (
        "你是文档图像理解专家。描述给定图像，输出一段结构化文本，供检索系统索引。\n"
        "1. 第一句说明图像类型（架构图/流程图/图表/照片/截图/示意图）。\n"
        "2. 客观描述图中关键元素、文字、数据与它们的关系；图表需读出坐标轴、图例、"
        "关键数值与趋势。\n3. 结合所在章节上下文说明图像用途，但禁止臆测图中不存在的信息。\n"
        "4. 长度 100–300 字。末尾给一行 \"confidence: 0.xx\"。",
        "所在章节：{context}\n[图像输入：{payload_ref}]",
    ),
    AtomType.TABLE: (
        "你是数据分析专家。描述给定表格（含扫描件表格的 OCR 结果）：\n"
        "1. 第一句概括表格主题与行列结构（表头含义）。\n"
        "2. 指出关键数据、极值、对比关系与异常值；禁止逐行复述。\n"
        "3. 说明该表格在文档中支撑什么结论（结合章节上下文）。\n"
        "4. 长度 80–200 字。末尾给一行 \"confidence: 0.xx\"。",
        "所在章节：{context}\n表格内容（Markdown）：\n{text}",
    ),
    AtomType.FORMULA: (
        "你是数学/工程文档专家。解释给定公式（LaTeX）：\n"
        "1. 用自然语言说明公式表达的含义与每个符号的物理/数学意义。\n"
        "2. 说明公式用途（定义/推导/约束/计算方法）及与上下文的关系。\n"
        "3. 长度 60–150 字。末尾给一行 \"confidence: 0.xx\"。",
        "所在章节：{context}\n公式：{text}",
    ),
}

_PROMPT_NAME: dict[AtomType, str] = {
    AtomType.IMAGE: "describe_image",
    AtomType.TABLE: "describe_table",
    AtomType.FORMULA: "describe_formula",
}

_CONFIDENCE = re.compile(r"confidence\s*:\s*(0?\.\d+|1\.0+)", re.I)

ChatCallable = Callable[[list[tuple[str, str]]], Awaitable[str]]
PromptLookup = Callable[[str], str]


class _ParsedDescription(BaseModel):
    """Scratch model for the trailing ``confidence: 0.xx`` line."""

    confidence: float = 0.5


def parse_confidence(text: str) -> tuple[str, float]:
    """Split the last ``confidence: 0.xx`` line from the description body."""
    match = _CONFIDENCE.search(text)
    if not match:
        return text.strip(), 0.5
    value = max(0.0, min(1.0, float(match.group(1))))
    body = (text[: match.start()] + text[match.end() :]).strip()
    return body, value


class VLMDescribeProcessor:
    """SPI ``Processor`` delegating to the router's ``describe`` role."""

    name: str = _NAME
    supported_atom_types: list[AtomType] = [AtomType.IMAGE, AtomType.TABLE, AtomType.FORMULA]

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self._chat = cfg.get("chat")
        self._prompts = cfg.get("prompts")
        self.batch_size: int = int(cfg.get("batch_size", 8))
        self.confidence_threshold: float = float(cfg.get("confidence_threshold", 0.6))
        self.small_model: str = str(cfg.get("small_model", "vlm-small"))
        self.large_model: str = str(cfg.get("large_model", "vlm-large"))
        self.model_name: str = str(cfg.get("model", self.small_model))

    def _system_prompt(self, atom_type: AtomType) -> str:
        if self._prompts is not None:
            try:
                return str(self._prompts(_PROMPT_NAME[atom_type]))
            except Exception:
                return _DEFAULT_PROMPTS[atom_type][0]
        return _DEFAULT_PROMPTS[atom_type][0]

    def _user_prompt(self, atom: Atom, atom_type: AtomType) -> str:
        template = _DEFAULT_PROMPTS[atom_type][1]
        try:
            return template.format(
                context=atom.context or "", text=atom.text or "", payload_ref=atom.payload_ref or ""
            )
        except (KeyError, IndexError, ValueError):
            return f"上下文：{atom.context or ''}\n内容：{atom.text or atom.payload_ref or ''}"

    async def describe(
        self, atoms: list[Atom], *, options: DescribeOptions
    ) -> list[AtomDescription]:
        if not atoms:
            return []
        if self._chat is None:
            raise ConfigError(
                "VLM processor needs an injected router chat callable",
                details={"hint": "pass config['chat'] wired to the Resilient Router"},
            )
        out: list[AtomDescription] = []
        for start in range(0, len(atoms), self.batch_size):
            batch = atoms[start : start + self.batch_size]
            results = await asyncio.gather(
                *(self._describe_one(atom, options) for atom in batch),
                return_exceptions=True,
            )
            for _atom, result in zip(batch, results, strict=True):
                if isinstance(result, BaseException):
                    raise result if isinstance(result, Exception) else RuntimeError(str(result))
                out.append(result)
        return out

    async def _describe_one(self, atom: Atom, options: DescribeOptions) -> AtomDescription:

        if atom.type not in self.supported_atom_types:
            return AtomDescription(
                atom_id=atom.atom_id,
                description=atom.text or "",
                confidence=1.0 if atom.text else 0.0,
                model="passthrough",
            )
        model = self._model_for(options)
        messages = [
            ("system", self._system_prompt(atom.type)),
            ("user", self._user_prompt(atom, atom.type)),
        ]
        raw = await self._chat(messages)
        body, confidence = parse_confidence(raw or "")
        return AtomDescription(
            atom_id=atom.atom_id,
            description=body or (atom.text or ""),
            confidence=confidence,
            model=model,
            cached=False,
        )

    def _model_for(self, options: DescribeOptions) -> str:
        hint = options.metadata.get("model") or options.metadata.get("tier")
        if hint == "large":
            return self.large_model
        if hint == "small":
            return self.small_model
        return self.model_name

    async def startup(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None
