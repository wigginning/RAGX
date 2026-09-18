"""PromptRegistry (12-prompts.md §12.1).

Loads the versioned prompt YAML files from ``ragx/prompts/*.v1.yaml`` at startup,
validates that every ``{{var}}`` placeholder has a matching caller argument, and
renders ``(system, user)`` pairs. Per-kb overrides come from
``KBConfig.prompt_overrides`` (02-core.md §2.4 three-level config).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from ragx.core.exceptions import ConfigError
from ragx.core.roles import LLMRole

_PLACEHOLDER = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")

#: Directory holding the versioned prompt files (12-prompts.md §12.1).
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


class Prompt:
    """One loaded, validated prompt template."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.name: str = str(data["name"])
        self.version: int = int(data["version"])
        self.lang: str = str(data.get("lang", "auto"))
        self.role: LLMRole = LLMRole(str(data["role"]))
        self.output: str = str(data.get("output", "text"))
        self.schema: str | None = data.get("schema")
        self.system: str = str(data.get("system", ""))
        self.user: str = str(data.get("user", ""))
        self.metadata: dict[str, Any] = data.get("metadata", {})
        self._vars: set[str] = set(_PLACEHOLDER.findall(self.system)) | set(
            _PLACEHOLDER.findall(self.user)
        )

    @property
    def key(self) -> str:
        return f"{self.name}.v{self.version}"

    def render(self, **vars: Any) -> tuple[str, str]:
        """Substitute ``{{var}}`` in system/user. Missing vars -> ConfigError(9003)."""
        missing = self._vars - set(vars)
        if missing:
            raise ConfigError(
                "prompt placeholder not provided",
                details={"prompt": self.key, "missing": sorted(missing)},
            )
        system = self._sub(self.system, vars)
        user = self._sub(self.user, vars)
        return system, user

    @staticmethod
    def _sub(template: str, vars: dict[str, Any]) -> str:
        def repl(match: re.Match[str]) -> str:
            return str(vars[match.group(1)])

        return _PLACEHOLDER.sub(repl, template)


class PromptRegistry:
    """Loads and serves versioned prompts."""

    def __init__(self, prompts_dir: Path | str | None = None) -> None:
        self._dir = Path(prompts_dir) if prompts_dir else PROMPTS_DIR
        #: name -> {version: Prompt}
        self._by_name: dict[str, dict[int, Prompt]] = {}
        self.load_all()

    def load_all(self) -> None:
        """Load every ``*.v*.yaml`` file in the prompts dir (12-prompts.md §12.1)."""
        self._by_name.clear()
        if not self._dir.is_dir():
            raise ConfigError(
                "prompts directory not found",
                details={"path": str(self._dir)},
            )
        for path in sorted(self._dir.glob("*.v*.yaml")):
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            prompt = Prompt(data)
            self._by_name.setdefault(prompt.name, {})[prompt.version] = prompt

    def get(self, name: str, version: int | str = "latest") -> Prompt:
        """Resolve a prompt by name and version (default: highest version)."""
        versions = self._by_name.get(name)
        if not versions:
            raise ConfigError(
                "unknown prompt",
                details={"name": name, "known": sorted(self._by_name)},
            )
        if version == "latest":
            chosen = max(versions)
        else:
            chosen = int(version)
            if chosen not in versions:
                raise ConfigError(
                    "prompt version not found",
                    details={"name": name, "version": chosen,
                             "available": sorted(versions)},
                )
        return versions[chosen]

    def render(self, name: str, *, version: int | str = "latest", **vars: Any) -> tuple[str, str]:
        """Convenience: resolve + render in one call."""
        return self.get(name, version).render(**vars)

    def render_for_kb(
        self,
        name: str,
        kb_overrides: dict[str, str] | None,
        *,
        version: int | str = "latest",
        **vars: Any,
    ) -> tuple[str, str]:
        """Resolve a prompt via the kb's overrides and render it.

        ``kb_overrides`` is typically ``KBConfig.prompt_overrides``: a map from
        a logical prompt name (``"extract_entities"``) to a concrete file
        (``"extract_entities.medical.v1"``). When no override matches, the
        original name is used (12-prompts.md §12.0).
        """
        resolved = self.resolve_override(name, kb_overrides or {})
        return self.render(resolved, version=version, **vars)

    def names(self) -> list[str]:
        return sorted(self._by_name)

    def resolve_override(self, name: str, overrides: dict[str, str]) -> str:
        """Apply a per-kb override (``KBConfig.prompt_overrides``): map a logical
        prompt name to an actual file name, e.g. ``extract_entities`` ->
        ``extract_entities.medical.v1``. Returns the resolved file name."""
        return overrides.get(name, name)
