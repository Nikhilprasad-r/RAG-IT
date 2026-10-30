import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

from pydantic import Field

from ragkit.core.errors import ConfigError
from ragkit.core.registry import registry
from ragkit.core.types import RawDocument
from ragkit.ingest.connectors.base import ConnectorOptions, glob_match, guess_mime_type


class FilesOptions(ConnectorOptions):
    path: str = Field(description="Directory to read, relative to the project root.")
    include: list[str] = Field(default_factory=lambda: ["**/*"])
    exclude: list[str] = Field(default_factory=list)
    include_hidden: bool = False
    max_file_mb: float = Field(default=50, gt=0)


class FilesConnector:
    """Reads files from a local directory (or a mounted network share)."""

    def __init__(self, options: FilesOptions, *, root: Path) -> None:
        self.options = options
        self.directory = (root / options.path).resolve()
        self.name = options.name or f"files:{options.path}"

    @classmethod
    def from_options(cls, options: dict[str, Any], *, root: Path) -> Self:
        return cls(FilesOptions.model_validate(options), root=root)

    async def fetch(self) -> AsyncIterator[RawDocument]:
        if not self.directory.is_dir():
            raise ConfigError(f"Source directory does not exist: {self.directory}")
        max_bytes = self.options.max_file_mb * 1024 * 1024
        for path in sorted(self.directory.rglob("*")):
            relative = path.relative_to(self.directory).as_posix()
            if not path.is_file() or not self._wanted(relative):
                continue
            stat = path.stat()
            if stat.st_size > max_bytes:
                continue
            yield RawDocument(
                source=self.name,
                uri=relative,
                data=await asyncio.to_thread(path.read_bytes),
                mime_type=guess_mime_type(path.name),
                metadata={
                    "path": relative,
                    "modified": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
                },
            )

    def _wanted(self, relative: str) -> bool:
        if not self.options.include_hidden and any(
            part.startswith(".") for part in relative.split("/")
        ):
            return False
        return glob_match(relative, self.options.include) and not glob_match(
            relative, self.options.exclude
        )


registry.register("connector", "files", FilesConnector)
