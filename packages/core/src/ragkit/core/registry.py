"""Plugin registry.

Plugins come from two places: explicit `registry.register(...)` calls, and
package entry points in the `ragkit.<kind>` groups (for example
`ragkit.connector`). Entry points are only imported when first requested, so
installing many plugins does not slow startup.
"""

from collections.abc import Callable
from importlib.metadata import EntryPoint, entry_points
from typing import Any, Literal, get_args, overload

from ragkit.core.errors import PluginNotFoundError

PluginKind = Literal[
    "connector",
    "parser",
    "chunker",
    "embedder",
    "vector_store",
    "retriever",
    "reranker",
    "tool",
]

PLUGIN_KINDS: tuple[str, ...] = get_args(PluginKind)


class Registry:
    def __init__(self, *, use_entry_points: bool = True) -> None:
        self._plugins: dict[str, dict[str, Any]] = {kind: {} for kind in PLUGIN_KINDS}
        self._entry_points: dict[str, dict[str, EntryPoint]] | None = (
            None if use_entry_points else {kind: {} for kind in PLUGIN_KINDS}
        )

    @overload
    def register(self, kind: PluginKind, name: str) -> Callable[[Any], Any]: ...

    @overload
    def register(self, kind: PluginKind, name: str, plugin: Any) -> Any: ...

    def register(self, kind: PluginKind, name: str, plugin: Any = None) -> Any:
        """Register `plugin` under `kind`/`name`; without `plugin`, act as a decorator."""
        self._check_kind(kind)

        def add(obj: Any) -> Any:
            self._plugins[kind][name] = obj
            return obj

        return add if plugin is None else add(plugin)

    def get(self, kind: PluginKind, name: str) -> Any:
        self._check_kind(kind)
        if name in self._plugins[kind]:
            return self._plugins[kind][name]
        entry_point = self._discover()[kind].get(name)
        if entry_point is None:
            available = ", ".join(self.names(kind)) or "none"
            raise PluginNotFoundError(f"No {kind} named {name!r}. Available: {available}.")
        plugin = self._plugins[kind][name] = entry_point.load()
        return plugin

    def names(self, kind: PluginKind) -> list[str]:
        self._check_kind(kind)
        return sorted(self._plugins[kind].keys() | self._discover()[kind].keys())

    def _discover(self) -> dict[str, dict[str, EntryPoint]]:
        if self._entry_points is None:
            self._entry_points = {
                kind: {ep.name: ep for ep in entry_points(group=f"ragkit.{kind}")}
                for kind in PLUGIN_KINDS
            }
        return self._entry_points

    @staticmethod
    def _check_kind(kind: str) -> None:
        if kind not in PLUGIN_KINDS:
            raise ValueError(f"Unknown plugin kind {kind!r}. Expected one of {PLUGIN_KINDS}.")


registry = Registry()
