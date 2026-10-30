from ragkit.core.config import ProjectConfig
from ragkit.core.errors import ConfigError
from ragkit.retrieval.stores.base import CollectionStats, Store, StoredDocument
from ragkit.retrieval.stores.sqlite import SQLiteStore


def create_store(project: ProjectConfig) -> Store:
    """Build the store configured in `ragkit.yaml`. Open it with `async with`."""
    if project.store.backend == "sqlite":
        return SQLiteStore(project.root / project.store.path)
    if not project.database_url:
        raise ConfigError("store.backend is 'postgres' but database_url is not set.")
    from ragkit.retrieval.stores.postgres import PostgresStore

    return PostgresStore(project.database_url)


__all__ = ["CollectionStats", "SQLiteStore", "Store", "StoredDocument", "create_store"]
