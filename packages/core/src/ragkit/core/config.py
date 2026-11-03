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
    # Docstrings under fields become their descriptions (used by the reference docs).
    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)


class ServerConfig(_Strict):
    host: str = "127.0.0.1"
    """Address `ragkit serve` binds to."""
    port: int = 8000
    """Port `ragkit serve` listens on."""
    cors_origins: list[str] = Field(default_factory=list)
    """Browser origins allowed to call the API."""


class StoreConfig(_Strict):
    backend: Literal["sqlite", "postgres"] = "sqlite"
    """`sqlite` keeps everything in one local file; `postgres` uses `database_url` with pgvector."""
    path: str = Field(
        default=".ragkit/index.db", description="SQLite file, relative to the project root."
    )


class EmbeddingsConfig(_Strict):
    model: str = Field(default=DEFAULT_EMBEDDING_MODEL, description="provider:model")
    dimensions: int | None = Field(
        default=768, ge=1, description="Output size for models that support shortening."
    )
    batch_size: int = Field(default=64, ge=1)
    """Texts per embedding request during ingestion."""


class RerankConfig(_Strict):
    provider: str = Field(description="Reranker plugin: cohere, voyage or cross-encoder.")
    model: str | None = None
    """Model name; each provider has a default."""
    top_n: int = Field(default=8, ge=1)
    """Results kept after reranking when a search does not ask for a number."""


class ChunkingConfig(_Strict):
    max_tokens: int = Field(default=400, ge=50)
    """Largest chunk, estimated at four characters per token."""
    overlap_tokens: int = Field(default=40, ge=0)
    """Words from the end of a chunk repeated at the start of the next one."""

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
    """Shown to agents deciding what to search, and in listings."""
    sources: list[SourceConfig] = Field(default_factory=list)
    """Where the documents come from."""
    parser: Literal["basic", "docling"] = "basic"
    """`docling` uses layout-aware conversion for PDF and office files (needs `ragkit-
    ingest[docling]`).
    """
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    """How documents are split into chunks."""
    acl: list[str] = Field(
        default_factory=list, description="Principals allowed to read; empty means everyone."
    )
    tenant: str = Field(default="default", description="Tenant that owns the documents.")
    shared: bool = Field(
        default=False, description="Searchable by every tenant (still subject to acl)."
    )
    sync_every: str | None = Field(
        default=None,
        pattern=r"^\d+[smhd]$",
        description="Re-ingest on this interval when a scheduler runs, e.g. 15m or 6h.",
    )

    @property
    def sync_seconds(self) -> float | None:
        if not self.sync_every:
            return None
        units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
        return int(self.sync_every[:-1]) * units[self.sync_every[-1]]


class McpServerConfig(_Strict):
    """An MCP server whose tools agents can use (`tools.mcp` in agent.yaml)."""

    transport: Literal["stdio", "http", "sse", "websocket"] = "stdio"
    """How to reach the server."""
    command: str | None = Field(default=None, description="stdio: program to start.")
    args: list[str] = Field(default_factory=list)
    """stdio: command arguments."""
    env: dict[str, str] = Field(default_factory=dict)
    """stdio: extra environment variables (PATH and HOME are kept)."""
    cwd: str | None = Field(default=None, description="stdio: working directory (default: root).")
    url: str | None = Field(default=None, description="http, sse, websocket: server URL.")
    headers: dict[str, str] = Field(default_factory=dict)
    """http and sse: request headers, such as Authorization."""
    timeout: float = Field(default=30, gt=0)
    """http and sse: seconds to wait for a response."""

    @model_validator(mode="after")
    def _endpoint(self) -> Self:
        if self.transport == "stdio" and not self.command:
            raise ValueError("stdio MCP servers need a command")
        if self.transport != "stdio" and not self.url:
            raise ValueError(f"{self.transport} MCP servers need a url")
        return self


class LangfuseConfig(_Strict):
    host: str = "https://cloud.langfuse.com"
    """Langfuse base URL (the EU cloud by default)."""
    public_key_env: str = "LANGFUSE_PUBLIC_KEY"
    """Variable holding the Langfuse public key."""
    secret_key_env: str = "LANGFUSE_SECRET_KEY"
    """Variable holding the Langfuse secret key."""


class ObservabilityConfig(_Strict):
    """Where traces go. Spans follow the OpenTelemetry GenAI conventions."""

    exporter: Literal["none", "console", "otlp", "langfuse", "phoenix"] = "none"
    """Where spans go; `none` turns tracing off."""
    endpoint: str | None = Field(
        default=None, description="OTLP/HTTP traces URL; defaults depend on the exporter."
    )
    headers: dict[str, str] = Field(default_factory=dict)
    """Extra headers for OTLP requests."""
    service_name: str | None = Field(default=None, description="Defaults to the project name.")
    capture_content: bool = Field(
        default=False,
        description="Record prompts, replies, tool arguments and results on spans.",
    )
    sample_ratio: float = Field(default=1.0, ge=0, le=1)
    """Share of traces kept (parent-based)."""
    langfuse: LangfuseConfig = Field(default_factory=LangfuseConfig)
    """Langfuse host and key variables, for `exporter: langfuse`."""


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
    """Eval cases run at the same time."""
    on_approval: Literal["approve", "reject"] = Field(
        default="reject",
        description="How evals answer approval requests; reject avoids real side effects.",
    )


Scope = Literal["chat", "search", "ingest", "admin"]
ALL_SCOPES: tuple[str, ...] = ("chat", "search", "ingest", "admin")


class ApiKeyConfig(_Strict):
    key_env: str = Field(description="Environment variable that holds the key.")
    name: str | None = Field(default=None, description="Label used as the user id in traces.")
    tenant: str = "default"
    """Tenant the caller acts as."""
    principals: list[str] = Field(default_factory=list)
    """Groups matched against document ACLs."""
    scopes: list[Scope] = Field(default_factory=lambda: ["chat", "search"])
    """Operations the key may perform."""


class JwtConfig(_Strict):
    jwks_url: str | None = Field(default=None, description="For RS256/ES256 tokens.")
    secret_env: str | None = Field(default=None, description="For HS256 tokens.")
    algorithms: list[str] = Field(default_factory=lambda: ["RS256"])
    """Accepted signing algorithms."""
    issuer: str | None = None
    """Required `iss` claim, if set."""
    audience: str | None = None
    """Required `aud` claim, if set."""
    tenant_claim: str = "tenant"
    """Claim holding the tenant."""
    principals_claim: str = "groups"
    """Claim holding the caller's groups (a list or a string)."""
    scopes_claim: str = "scope"
    """Claim holding scopes (space-separated or a list)."""
    default_scopes: list[Scope] = Field(default_factory=lambda: ["chat", "search"])
    """Scopes granted when the token has none."""

    @model_validator(mode="after")
    def _key_source(self) -> Self:
        if bool(self.jwks_url) == bool(self.secret_env):
            raise ValueError("set exactly one of jwks_url or secret_env")
        return self


class AuthConfig(_Strict):
    mode: Literal["none", "api_key", "jwt"] = "none"
    """`none` (local use), `api_key` or `jwt`."""
    api_keys: list[ApiKeyConfig] = Field(default_factory=list)
    """Accepted keys, for `mode: api_key`."""
    jwt: JwtConfig | None = None
    """Token validation, for `mode: jwt`."""
    rate_limit_per_minute: int | None = Field(
        default=None, ge=1, description="Per caller and server process."
    )

    @model_validator(mode="after")
    def _mode_settings(self) -> Self:
        if self.mode == "api_key" and not self.api_keys:
            raise ValueError("auth mode api_key needs api_keys")
        if self.mode == "jwt" and self.jwt is None:
            raise ValueError("auth mode jwt needs a jwt section")
        return self


class QueueConfig(_Strict):
    lease_seconds: int = Field(default=300, ge=10, description="A job is retried after this.")
    max_attempts: int = Field(default=3, ge=1)
    """Tries before a job is marked failed."""
    poll_seconds: float = Field(default=2.0, gt=0)
    """How often an idle worker checks for jobs."""


class CacheConfig(_Strict):
    query_embeddings: int = Field(
        default=10_000, ge=0, description="Query embeddings kept in memory; 0 disables."
    )
    search_ttl_seconds: float = Field(
        default=0, ge=0, description="Reuse identical search results for this long; 0 disables."
    )
    search_entries: int = Field(default=10_000, ge=1)
    """Most search results kept in memory."""


class ProjectConfig(_Strict):
    name: str
    """Project name; also the default tracing service name."""
    mode: Literal["local", "cluster"] = "local"
    """Informational only; backends are chosen by `store.backend` and each agent's
    `memory.checkpointer`.
    """
    default_model: str = DEFAULT_MODEL
    """Chat model for agents that do not set `model`, as provider:model."""
    agents_dir: str = "agents"
    """Folder that holds agent folders, relative to the project root."""
    database_url: str | None = None
    """Postgres URL for the postgres store, job queue and checkpointer."""
    server: ServerConfig = Field(default_factory=ServerConfig)
    """Defaults for `ragkit serve`."""
    store: StoreConfig = Field(default_factory=StoreConfig)
    """Where documents, chunks and embeddings are kept."""
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    """Embedding model for ingestion and search."""
    rerank: RerankConfig | None = None
    """Default reranker for every search, unless a search or agent turns it off or overrides it."""
    collections: dict[str, CollectionConfig] = Field(default_factory=dict)
    """Document collections by name (lowercase letters, digits, - and _)."""
    mcp_servers: dict[str, McpServerConfig] = Field(default_factory=dict)
    """MCP servers whose tools agents can use, by name."""
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    """Tracing export."""
    pricing: dict[str, ModelPrice] = Field(
        default_factory=dict, description="Prices by provider:model, used for cost tracking."
    )
    evals: EvalsConfig = Field(default_factory=EvalsConfig)
    """Settings for `ragkit eval`."""
    auth: AuthConfig = Field(default_factory=AuthConfig)
    """Authentication for the HTTP API."""
    queue: QueueConfig = Field(default_factory=QueueConfig)
    """Ingestion job queue."""
    cache: CacheConfig = Field(default_factory=CacheConfig)
    """In-process caches."""

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
