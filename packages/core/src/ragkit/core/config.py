"""Project configuration, loaded from `ragkit.yaml`.

`${VAR}` and `${VAR:-default}` in the file are replaced from the environment,
after the project's `.env` has been loaded, so secrets stay out of the config.
"""

import os
import re
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from ragkit.core.errors import ConfigError, ProjectNotFoundError

CONFIG_FILENAME = "ragkit.yaml"
DEFAULT_MODEL = "google_genai:gemini-3.5-flash-lite"
DEFAULT_EMBEDDING_MODEL = "google_genai:gemini-embedding-2"
COLLECTION_NAME_PATTERN = r"^[a-z][a-z0-9_-]*$"

_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ServerConfig(_Strict):
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=list)


class StoreConfig(_Strict):
    backend: Literal["sqlite", "postgres"] = "sqlite"
    path: str = Field(
        default=".ragkit/index.db", description="SQLite file, relative to the project root."
    )


class EmbeddingsConfig(_Strict):
    model: str = Field(default=DEFAULT_EMBEDDING_MODEL, description="provider:model")
    dimensions: int | None = Field(
        default=768, ge=1, description="Output size for models that support shortening."
    )
    batch_size: int = Field(default=64, ge=1)


class RerankConfig(_Strict):
    provider: str = Field(description="Reranker plugin: cohere, voyage or cross-encoder.")
    model: str | None = None
    top_n: int = Field(default=8, ge=1)


class ChunkingConfig(_Strict):
    max_tokens: int = Field(default=400, ge=50)
    overlap_tokens: int = Field(default=40, ge=0)

    @model_validator(mode="after")
    def _overlap_fits(self) -> Self:
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")
        return self


class SourceConfig(BaseModel):
    """One data source. Fields besides `type` are validated by the connector."""

    model_config = ConfigDict(extra="allow")

    type: str = Field(description="Connector name, or 'module:Class' for a custom one.")

    @property
    def options(self) -> dict[str, Any]:
        return dict(self.model_extra or {})


class CollectionConfig(_Strict):
    description: str = ""
    sources: list[SourceConfig] = Field(default_factory=list)
    parser: Literal["basic", "docling"] = "basic"
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    acl: list[str] = Field(
        default_factory=list, description="Principals allowed to read; empty means everyone."
    )


class McpServerConfig(_Strict):
    """An MCP server whose tools agents can use (`tools.mcp` in agent.yaml)."""

    transport: Literal["stdio", "http", "sse", "websocket"] = "stdio"
    command: str | None = Field(default=None, description="stdio: program to start.")
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = Field(default=None, description="stdio: working directory (default: root).")
    url: str | None = Field(default=None, description="http, sse, websocket: server URL.")
    headers: dict[str, str] = Field(default_factory=dict)
    timeout: float = Field(default=30, gt=0)

    @model_validator(mode="after")
    def _endpoint(self) -> Self:
        if self.transport == "stdio" and not self.command:
            raise ValueError("stdio MCP servers need a command")
        if self.transport != "stdio" and not self.url:
            raise ValueError(f"{self.transport} MCP servers need a url")
        return self


class LangfuseConfig(_Strict):
    host: str = "https://cloud.langfuse.com"
    public_key_env: str = "LANGFUSE_PUBLIC_KEY"
    secret_key_env: str = "LANGFUSE_SECRET_KEY"


class ObservabilityConfig(_Strict):
    """Where traces go. Spans follow the OpenTelemetry GenAI conventions."""

    exporter: Literal["none", "console", "otlp", "langfuse", "phoenix"] = "none"
    endpoint: str | None = Field(
        default=None, description="OTLP/HTTP traces URL; defaults depend on the exporter."
    )
    headers: dict[str, str] = Field(default_factory=dict)
    service_name: str | None = Field(default=None, description="Defaults to the project name.")
    capture_content: bool = Field(
        default=False,
        description="Record prompts, replies, tool arguments and results on spans.",
    )
    sample_ratio: float = Field(default=1.0, ge=0, le=1)
    langfuse: LangfuseConfig = Field(default_factory=LangfuseConfig)


class ModelPrice(_Strict):
    input_per_million: float = Field(ge=0, description="USD per million input tokens.")
    output_per_million: float = Field(ge=0, description="USD per million output tokens.")

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_per_million + output_tokens * self.output_per_million
        ) / 1_000_000


class EvalsConfig(_Strict):
    judge_model: str | None = Field(
        default=None, description="Model that grades answers; defaults to default_model."
    )
    concurrency: int = Field(default=4, ge=1)
    on_approval: Literal["approve", "reject"] = Field(
        default="reject",
        description="How evals answer approval requests; reject avoids real side effects.",
    )


class ProjectConfig(_Strict):
    name: str
    mode: Literal["local", "cluster"] = "local"
    default_model: str = DEFAULT_MODEL
    agents_dir: str = "agents"
    database_url: str | None = None
    server: ServerConfig = Field(default_factory=ServerConfig)
    store: StoreConfig = Field(default_factory=StoreConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    rerank: RerankConfig | None = None
    collections: dict[str, CollectionConfig] = Field(default_factory=dict)
    mcp_servers: dict[str, McpServerConfig] = Field(default_factory=dict)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    pricing: dict[str, ModelPrice] = Field(
        default_factory=dict, description="Prices by provider:model, used for cost tracking."
    )
    evals: EvalsConfig = Field(default_factory=EvalsConfig)

    root: Path = Field(default=Path("."), exclude=True)

    @field_validator("collections")
    @classmethod
    def _collection_names(cls, value: dict[str, CollectionConfig]) -> dict[str, CollectionConfig]:
        for name in value:
            if not re.fullmatch(COLLECTION_NAME_PATTERN, name):
                raise ValueError(
                    f"Collection name {name!r} must use lowercase letters, digits, - and _."
                )
        return value

    @property
    def agents_path(self) -> Path:
        return self.root / self.agents_dir

    def collection(self, name: str) -> CollectionConfig:
        try:
            return self.collections[name]
        except KeyError:
            available = ", ".join(sorted(self.collections)) or "none"
            raise ConfigError(f"No collection named {name!r}. Available: {available}.") from None

    @classmethod
    def load(cls, start: Path | str | None = None) -> Self:
        """Load the config of the project containing `start` (default: the cwd)."""
        root = find_project_root(start)
        load_dotenv(root / ".env", override=False)
        path = root / CONFIG_FILENAME
        try:
            data = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"{path} is not valid YAML: {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigError(f"{path} must contain a mapping at the top level.")
        try:
            return cls.model_validate({**_expand_env(data), "root": root})
        except ValidationError as exc:
            raise ConfigError(f"Invalid {path}:\n{exc}") from exc


def find_project_root(start: Path | str | None = None) -> Path:
    current = Path(start or Path.cwd()).resolve()
    for directory in (current, *current.parents):
        if (directory / CONFIG_FILENAME).is_file():
            return directory
    raise ProjectNotFoundError(
        f"No {CONFIG_FILENAME} found in {current} or its parents. Run `ragkit init` first."
    )


def _expand_env(value: Any) -> Any:
    """Replace `${VAR}` references in every string inside parsed YAML."""
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        resolved = os.environ.get(name, default)
        if resolved is None:
            raise ConfigError(f"Environment variable {name} is referenced but not set.")
        return resolved

    return _ENV_REF.sub(replace, value)
