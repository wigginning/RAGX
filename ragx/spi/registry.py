"""Plugin registry: registration, discovery, resolution (01-spi.md §1.3).

Resolution priority: knowledge-base plugin name -> profile default -> error.
Instances are cached per ``(interface, name, kb_id)`` and their optional
``startup()`` / ``shutdown()`` hooks are driven by the API service lifecycle.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from importlib.metadata import entry_points
from typing import Any

from ragx.core.exceptions import ConfigError
from ragx.core.settings import KBConfig, Settings
from ragx.spi.interfaces import (
    Embedder,
    GraphStore,
    LLMProvider,
    Parser,
    Processor,
    Reranker,
    VectorStore,
)

logger = logging.getLogger(__name__)

Factory = Callable[[dict[str, Any]], Any]

_IFACE_TYPES: dict[str, type] = {
    "parser": Parser,
    "processor": Processor,
    "embedder": Embedder,
    "reranker": Reranker,
    "vector_store": VectorStore,
    "graph_store": GraphStore,
    "llm_provider": LLMProvider,
}

_ENTRY_POINTS: dict[str, str] = {
    "parser": "ragx.parsers",
    "processor": "ragx.processors",
    "embedder": "ragx.embedders",
    "reranker": "ragx.rerankers",
    "vector_store": "ragx.vector_stores",
    "graph_store": "ragx.graph_stores",
    "llm_provider": "ragx.llm_providers",
}

#: Profile -> default plugin name (11-plugins-builtin.md §11.3.2).
#:
#: Deviation from the doc table, intentional and documented: the lite embedder
#: default is ``hash`` (zero-dependency deterministic embedder) instead of
#: ``st`` (sentence-transformers/torch). The critical path must run in a clean
#: Python environment; ``st`` remains available as an explicit choice behind
#: the ``ragx[embed-st]`` extra.
_PROFILE_DEFAULTS: dict[str, dict[str, str | None]] = {
    "lite": {
        "parser": "text",
        "processor": None,
        "embedder": "hash",
        "reranker": None,
        "vector_store": "sqlite",
        "graph_store": "nx",
        "llm_provider": "openai_compat",
    },
    "full": {
        "parser": "deepdoc",
        "processor": "vlm",
        "embedder": "st",
        "reranker": "bge",
        "vector_store": "es",
        "graph_store": "neo4j",
        "llm_provider": "openai_compat",
    },
}


def default_plugin(profile: str, interface: str) -> str | None:
    """Return the profile default plugin name for ``interface``."""
    return _PROFILE_DEFAULTS.get(profile, _PROFILE_DEFAULTS["lite"]).get(interface)


class PluginRegistry:
    """Holds factories and hands out cached plugin instances."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings: Settings = settings or Settings()
        self._factories: dict[tuple[str, str], Factory] = {}
        self._instances: dict[tuple[str, str, str], Any] = {}
        self._discovered = False
        self._lock = threading.Lock()

    # -- registration ------------------------------------------------------
    def register(
        self, interface: str | type, name: str, factory: Factory | Callable[[], Any] | type
    ) -> None:
        """Register a factory for ``interface[name]``.

        ``factory`` may be a callable taking a config dict, a zero-arg callable,
        or a class (called with the config dict).
        """
        key = self._iface_key(interface)
        if not callable(factory):
            raise ConfigError(
                "plugin factory must be a callable or a class",
                details={"interface": key, "name": name, "factory": repr(factory)},
            )
        wrapped: Factory
        if isinstance(factory, type):
            wrapped = lambda cfg, cls=factory: cls(cfg)  # noqa: E731
        elif _takes_config(factory):
            wrapped = factory
        else:
            wrapped = lambda cfg, f=factory: f()  # noqa: E731
        self._factories[(key, name)] = wrapped

    def unregister(self, interface: str | type, name: str) -> bool:
        key = self._iface_key(interface)
        return self._factories.pop((key, name), None) is not None

    def names(self, interface: str | type) -> list[str]:
        key = self._iface_key(interface)
        return sorted(n for (i, n) in self._factories if i == key)

    def has(self, interface: str | type, name: str) -> bool:
        return (self._iface_key(interface), name) in self._factories

    # -- discovery ---------------------------------------------------------
    def discover(self) -> list[str]:
        """Load builtin plugins + installed entry points (idempotent)."""
        if self._discovered:
            return self._all_names()
        if self.settings.registry.discover_entry_points:
            self._discover_entry_points()
        self._register_builtins()
        self._discovered = True
        logger.info("plugins discovered: %s", ", ".join(self._all_names()) or "(none)")
        return self._all_names()

    def _discover_entry_points(self) -> None:
        for interface, group in _ENTRY_POINTS.items():
            try:
                eps = entry_points(group=group)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("entry point discovery failed for %s: %s", group, exc)
                continue
            for ep in eps:
                try:
                    obj = ep.load()
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("entry point %s/%s failed to load: %s", group, ep.name, exc)
                    continue
                if isinstance(obj, type):
                    self.register(interface, ep.name, obj)
                elif callable(obj):
                    self.register(interface, ep.name, obj)

    def _register_builtins(self) -> None:
        """Direct registration of bundled plugins (no packaging required)."""
        try:
            from ragx.plugins import register_builtins

            register_builtins(self)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("builtin plugin registration failed: %s", exc)

    def _all_names(self) -> list[str]:
        return sorted(f"{i}:{n}" for i, n in self._factories)

    # -- resolution --------------------------------------------------------
    def resolve(
        self,
        interface: str | type,
        name: str,
        config: dict[str, Any] | None = None,
        *,
        kb_id: str = "_global",
    ) -> Any:
        """Resolve and cache a plugin instance.

        Raises ``ConfigError(9003)`` when the interface or plugin name is unknown.
        """
        key = self._iface_key(interface)
        cfg_key = (key, name, kb_id)
        if not self._discovered:
            self.discover()
        with self._lock:
            if cfg_key in self._instances:
                return self._instances[cfg_key]
        factory = self._factories.get((key, name))
        if factory is None:
            raise ConfigError(
                "plugin not found",
                details={
                    "interface": key,
                    "name": name,
                    "known": self.names(key),
                },
            )
        # kb-scoped plugins receive their kb_id in the config so the instance
        # can enforce isolation (01-spi.md §1.2). The explicit kb_id wins over
        # any config value.
        merged = dict(config or {})
        if kb_id != "_global":
            merged["kb_id"] = kb_id
        instance = factory(merged)
        with self._lock:
            self._instances.setdefault(cfg_key, instance)
        return self._instances[cfg_key]

    def resolve_from_kb(
        self, interface: str, kb_cfg: KBConfig, *, kb_id: str = "_global"
    ) -> Any:
        """Resolve the plugin named in ``KBConfig`` for ``interface``.

        Interface -> KBConfig field mapping matches 02-core.md §2.4.
        """
        field = _IFACE_TO_KBCFG_FIELD.get(interface)
        if field is None:
            raise ConfigError(
                "interface has no KBConfig mapping", details={"interface": interface}
            )
        name = getattr(kb_cfg, field)
        if not name:
            raise ConfigError(
                "KBConfig does not enable this interface",
                details={"interface": interface, "field": field},
            )
        config = self.settings.plugins.for_plugin(field, name)
        return self.resolve(interface, name, config, kb_id=kb_id)

    def resolve_default(
        self, interface: str, *, kb_id: str = "_global"
    ) -> Any | None:
        """Resolve the profile default plugin; ``None`` when the profile disables it."""
        name = default_plugin(self.settings.profile, interface)
        if not name:
            return None
        config = self.settings.plugins.for_plugin(
            _IFACE_TO_KBCFG_FIELD.get(interface, interface), name
        )
        return self.resolve(interface, name, config, kb_id=kb_id)

    # -- capabilities (01-spi.md §1.3, 11-plugins-builtin.md §11.0) --------
    def validate_capabilities(self, kb_cfg: KBConfig, *, kb_id: str = "_global") -> None:
        """Startup capability gate. Hard failures raise ``ConfigError(9003)``.

        * ``flags.kg_enabled`` without ``graph_store`` -> 9003 (11.0 explicit).
        * ``vector_store.supports_filter == false`` -> 9003 (filter pushdown is
          a hard requirement of the retrieval layer, 06-retrieval.md §6.6.1).
        * ``supports_bm25 == false`` -> **soft**: the retrieval layer skips the
          keyword route at runtime (11.0), so no startup failure.
        """
        from ragx.core.exceptions import ConfigError as _CE

        if kb_cfg.flags.kg_enabled and not kb_cfg.graph_store:
            raise _CE(
                "kg_enabled requires a graph_store",
                details={"kb_id": kb_id, "flags.kg_enabled": True},
            )
        store = self.resolve("vector_store", kb_cfg.vector_store,
                             self.settings.plugins.for_plugin("vector_store", kb_cfg.vector_store),
                             kb_id=kb_id)
        caps = getattr(store, "capabilities", None)
        if caps is None:
            raise _CE(
                "vector store does not declare capabilities",
                details={"plugin": kb_cfg.vector_store},
            )
        if not caps.supports_filter:
            raise _CE(
                "vector store must support metadata filters (supports_filter=true)",
                details={"plugin": kb_cfg.vector_store},
            )
        if kb_cfg.graph_store:
            gs = self.resolve(
                "graph_store", kb_cfg.graph_store,
                self.settings.plugins.for_plugin("graph_store", kb_cfg.graph_store),
                kb_id=kb_id,
            )
            if getattr(gs, "capabilities", None) is None:
                raise _CE(
                    "graph store does not declare capabilities",
                    details={"plugin": kb_cfg.graph_store},
                )
        if kb_cfg.processor:
            proc = self.resolve(
                "processor", kb_cfg.processor,
                self.settings.plugins.for_plugin("processor", kb_cfg.processor),
                kb_id=kb_id,
            )
            if getattr(proc, "supported_atom_types", None) is None:
                raise _CE(
                    "processor does not declare supported_atom_types",
                    details={"plugin": kb_cfg.processor},
                )

    # -- lifecycle ---------------------------------------------------------
    async def startup_all(self) -> None:
        """Call ``startup()`` on every cached instance (optional hook)."""
        for instance in list(self._instances.values()):
            hook = getattr(instance, "startup", None)
            if callable(hook):
                try:
                    result = hook()
                    if hasattr(result, "__await__"):
                        await result
                except Exception as exc:  # pragma: no cover - defensive
                    logger.error("plugin startup failed: %s", exc)
                    raise

    async def shutdown_all(self) -> None:
        for instance in list(self._instances.values()):
            hook = getattr(instance, "shutdown", None)
            if callable(hook):
                try:
                    result = hook()
                    if hasattr(result, "__await__"):
                        await result
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("plugin shutdown failed: %s", exc)

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _iface_key(interface: str | type) -> str:
        if isinstance(interface, str):
            if interface not in _IFACE_TYPES:
                raise ConfigError(
                    "unknown plugin interface",
                    details={"interface": interface, "known": sorted(_IFACE_TYPES)},
                )
            return interface
        for name, cls in _IFACE_TYPES.items():
            if interface is cls:
                return name
        raise ConfigError(
            "unknown plugin interface",
            details={"interface": getattr(interface, "__name__", repr(interface)),
                     "known": sorted(_IFACE_TYPES)},
        )


_IFACE_TO_KBCFG_FIELD: dict[str, str] = {
    "parser": "parser",
    "processor": "processor",
    "embedder": "embedder",
    "reranker": "reranker",
    "vector_store": "vector_store",
    "graph_store": "graph_store",
}


def _takes_config(factory: Callable[..., Any]) -> bool:
    """Heuristic: does ``factory`` accept one positional argument?"""
    import inspect

    try:
        sig = inspect.signature(factory)
    except (TypeError, ValueError):
        return True
    positional = [
        p for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    return len(positional) >= 1
