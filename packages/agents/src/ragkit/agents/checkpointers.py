from collections.abc import AsyncIterator, Iterable
from contextlib import AsyncExitStack, asynccontextmanager

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from ragkit.core.config import ProjectConfig
from ragkit.core.errors import ConfigError


@asynccontextmanager
async def open_checkpointers(
    project: ProjectConfig,
    kinds: Iterable[str],
    *,
    allowed_types: Iterable[tuple[str, str]] = (),
) -> AsyncIterator[dict[str, BaseCheckpointSaver]]:
    """Open one shared checkpointer per kind for the lifetime of the context.

    `allowed_types` are (module, class) pairs of project classes stored in agent
    state, such as output schemas; LangGraph only restores registered classes.
    """
    serde = JsonPlusSerializer(allowed_msgpack_modules=sorted(set(allowed_types)))
    async with AsyncExitStack() as stack:
        savers: dict[str, BaseCheckpointSaver] = {}
        for kind in set(kinds):
            if kind == "memory":
                savers[kind] = InMemorySaver(serde=serde)
            elif kind == "postgres":
                savers[kind] = await _open_postgres(project, stack, serde)
            else:
                raise ConfigError(f"Unknown checkpointer {kind!r}.")
        yield savers


async def _open_postgres(
    project: ProjectConfig, stack: AsyncExitStack, serde: JsonPlusSerializer
) -> BaseCheckpointSaver:
    if not project.database_url:
        raise ConfigError("memory.checkpointer is 'postgres' but database_url is not set.")
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    except ImportError as exc:
        raise ConfigError(
            "The postgres checkpointer needs `ragkit-agents[postgres]` installed."
        ) from exc
    import psycopg

    from ragkit.retrieval.stores.postgres import schema_lock

    saver = await stack.enter_async_context(
        AsyncPostgresSaver.from_conn_string(project.database_url, serde=serde)
    )
    # Several pods may start together; run LangGraph's migrations one at a time.
    async with (
        await psycopg.AsyncConnection.connect(project.database_url, autocommit=True) as conn,
        schema_lock(conn),
    ):
        await saver.setup()
    return saver
