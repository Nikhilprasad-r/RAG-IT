import pytest

from ragkit.core.errors import PluginNotFoundError
from ragkit.core.registry import Registry


def test_register_as_decorator_and_directly() -> None:
    registry = Registry(use_entry_points=False)

    @registry.register("chunker", "fixed")
    class FixedChunker:
        pass

    registry.register("embedder", "fake", object())

    assert registry.get("chunker", "fixed") is FixedChunker
    assert registry.names("chunker") == ["fixed"]
    assert registry.names("embedder") == ["fake"]


def test_missing_plugin_lists_available_names() -> None:
    registry = Registry(use_entry_points=False)
    registry.register("reranker", "cohere", object())

    with pytest.raises(PluginNotFoundError, match="Available: cohere"):
        registry.get("reranker", "voyage")


def test_unknown_kind_is_rejected() -> None:
    registry = Registry(use_entry_points=False)

    with pytest.raises(ValueError, match="Unknown plugin kind"):
        registry.register("widget", "x", object())  # type: ignore[arg-type]
