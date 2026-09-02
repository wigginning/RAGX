"""Tests for KBConfig.prompt_overrides (RX-LLM-03)."""

from __future__ import annotations

from ragx.core.settings import KBConfig
from ragx.llm.prompts import PromptRegistry


def test_render_for_kb_uses_default_when_no_override() -> None:
    reg = PromptRegistry()
    overrides: dict[str, str] = {}
    sys_default, _ = reg.render_for_kb("extract_entities", overrides, chunk_text="x")
    sys_explicit, _ = reg.render_for_kb("extract_entities", None, chunk_text="x")
    # Same content because both resolve to the default extract_entities.v1.
    assert "JSON" in sys_default or "json" in sys_default
    assert sys_default == sys_explicit


def test_render_for_kb_applies_override() -> None:
    reg = PromptRegistry()
    overrides = {"extract_entities": "extract_entities.medical"}
    sys_med, _ = reg.render_for_kb("extract_entities", overrides, chunk_text="x")
    sys_default, _ = reg.render_for_kb("extract_entities", None, chunk_text="x")
    # The medical override mentions "medical-domain"; the default does not.
    assert "medical" in sys_med.lower()
    assert sys_med != sys_default


def test_unknown_override_falls_back_to_default() -> None:
    reg = PromptRegistry()
    overrides = {"unknown_prompt": "another_unknown"}
    sys_fallback, _ = reg.render_for_kb("extract_entities", overrides, chunk_text="x")
    sys_default, _ = reg.render_for_kb("extract_entities", None, chunk_text="x")
    assert sys_fallback == sys_default


def test_kbconfig_prompt_overrides_field_default_empty() -> None:
    cfg = KBConfig()
    assert cfg.prompt_overrides == {}


def test_kbconfig_prompt_overrides_persists() -> None:
    cfg = KBConfig(prompt_overrides={"generate_standard": "generate_standard.safety"})
    assert cfg.prompt_overrides["generate_standard"] == "generate_standard.safety"
