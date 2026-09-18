"""PromptRegistry tests (12-prompts.md §12.1, RX-LLM-01 DoD)."""

from __future__ import annotations

import pytest

from ragx.core.exceptions import ConfigError
from ragx.core.roles import LLMRole
from ragx.llm.prompts import PromptRegistry


@pytest.fixture(scope="module")
def registry() -> PromptRegistry:
    return PromptRegistry()


def test_loads_all_15_prompts(registry: PromptRegistry) -> None:
    """14 base prompts + the ``extract_entities.medical`` override."""
    expected = {
        "extract_entities", "augment_extraction", "describe_image",
        "describe_table", "describe_formula", "rewrite_query", "seed_query",
        "planner", "task_answer", "synthesis", "verify", "generate_standard",
        "topic_summary", "entity_merge", "extract_entities.medical",
    }
    assert set(registry.names()) == expected


def test_get_latest_version(registry: PromptRegistry) -> None:
    p = registry.get("generate_standard")
    assert p.name == "generate_standard"
    assert p.version == 1
    assert p.role == LLMRole.GENERATE


def test_render_substitutes_variables(registry: PromptRegistry) -> None:
    system, user = registry.render(
        "generate_standard", query="什么是 RAGX？", assembled_context="上下文内容"
    )
    assert "什么是 RAGX？" in user
    assert "上下文内容" in user
    assert "知识库问答助手" in system


def test_render_missing_variable_raises_9003(registry: PromptRegistry) -> None:
    with pytest.raises(ConfigError) as ei:
        registry.render("generate_standard", query="缺一个变量")
    assert ei.value.code == 9003


def test_unknown_prompt_raises(registry: PromptRegistry) -> None:
    with pytest.raises(ConfigError):
        registry.get("does_not_exist")


def test_unknown_version_raises(registry: PromptRegistry) -> None:
    with pytest.raises(ConfigError):
        registry.get("generate_standard", version=99)


def test_structured_prompts_declare_schema(registry: PromptRegistry) -> None:
    for name in ("extract_entities", "planner", "verify", "entity_merge"):
        p = registry.get(name)
        assert p.output == "structured"
        assert p.schema, f"{name} must declare a schema reference"


def test_describe_prompts_use_describe_role(registry: PromptRegistry) -> None:
    for name in ("describe_image", "describe_table", "describe_formula"):
        assert registry.get(name).role == LLMRole.DESCRIBE
