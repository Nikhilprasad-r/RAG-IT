from collections.abc import AsyncIterator, Iterable
from contextlib import AsyncExitStack, asynccontextmanager

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from ragkit.core.config import ProjectConfig
from ragkit.core.errors import ConfigError


@asynccontextmanager
async def open_checkpointers(
    project: ProjectConfig, kinds: Iterable[str]
) -> AsyncIterator[dict[str, BaseCheckpointSaver]]:
    """Open one shared checkpointer per kind for the lifetime of the context."""
    async with AsyncExitStack() as stack:
        savers: dict[str, BaseCheckpointSaver] = {}
        for kind in set(kinds):
            if kind == "memory":
                savers[kind] = InMemorySaver()
            elif kind == "postgres":
                savers[kind] = await _open_postgres(project, stack)
            else:
                raise ConfigError(f"Unknown checkpointer {kind!r}.")
        yield savers


async def _open_postgres(project: ProjectConfig, stack: AsyncExitStack) -> BaseCheckpointSaver:
    if not project.database_url:
        raise ConfigError("memory.checkpointer is 'postgres' but database_url is not set.")
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    except ImportError as exc:
        raise ConfigError(
            "The postgres checkpointer needs `ragkit-agents[postgres]` installed."
        ) from exc
    saver = await stack.enter_async_context(
        AsyncPostgresSaver.from_conn_string(project.database_url)
    )
    await saver.setup()
    return saver
