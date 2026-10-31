"""Schema for an agent's `agent.yaml`.

The schema already covers the full design (for example long-term memory) so
manifests stay stable as features land. Sections the runtime
does not act on yet are listed by `AgentManifest.unwired_features()`.
"""

from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from ragkit.agents.guardrails import parse_guardrails
from ragkit.core.config import RerankConfig
from ragkit.core.errors import ConfigError

MANIFEST_FILENAME = "agent.yaml"
NAME_PATTERN = r"^[a-z][a-z0-9-]*$"

Pattern = Literal["react", "rag-chat", "plan-execute", "supervisor", "swarm", "pipeline"]


class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)


class RetrievalSpec(_Spec):
    collections: list[str] = Field(min_length=1)
    """Collections to search, as named in ragkit.yaml."""
    mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    """`dense` (embeddings), `sparse` (BM25 keywords) or `hybrid` (both, fused)."""
    top_k: int = Field(default=6, ge=1, description="Passages given to the model per search.")
    candidates: int = Field(default=50, ge=1, description="Results fetched before reranking.")
    rerank: RerankConfig | None = Field(
        default=None, description="Overrides the project's reranker for this agent."
    )
    use_rerank: bool = True
    """Set to false to skip reranking for this agent."""


class McpServerSpec(_Spec):
    server: str
    """Name of a server under `mcp_servers` in ragkit.yaml."""
    allow: list[str] = Field(default_factory=list, description="Tool allowlist; empty allows all.")


class ToolsSpec(_Spec):
    local: list[str] = Field(
        default_factory=list, description="Names of functions or tools defined in tools.py."
    )
    mcp: list[McpServerSpec] = Field(default_factory=list)
    """Tools from MCP servers."""


class MemorySpec(_Spec):
    checkpointer: Literal["memory", "postgres"] = "memory"
    """`memory` (per process, lost on restart) or `postgres` (shared by every replica)."""
    long_term: bool = False
    """Not acted on yet."""


Decision = Literal["approve", "edit", "reject"]


class HitlSpec(_Spec):
    require_approval: list[str] = Field(
        default_factory=list, description="Tools that pause for a human decision before running."
    )
    decisions: list[Decision] = Field(
        default_factory=lambda: ["approve", "edit", "reject"], min_length=1
    )
    """What a reviewer may do with a pending call."""


class BudgetSpec(_Spec):
    max_steps: int = Field(default=12, ge=1, description="Model calls allowed per request.")
    max_cost_usd: float | None = Field(default=None, gt=0)
    """End the run politely once it has cost this much; needs a price under `pricing`."""


class EvalSpec(_Spec):
    dataset: str | None = None
    """JSONL eval cases, relative to the agent folder."""
    gate: dict[str, float] = Field(default_factory=dict)
    """Minimum per metric, or a maximum with a `max_` prefix; `ragkit eval` exits 1 when one fails.
    """


class AgentManifest(_Spec):
    name: str = Field(pattern=NAME_PATTERN)
    """Unique name: lowercase letters, digits and hyphens."""
    description: str = ""
    """Shown in listings and to supervisors choosing whom to delegate to."""
    pattern: Pattern = "react"
    """`react` (tool loop), `rag-chat` (retrieve once, then answer) or `supervisor` (delegates to
    `subagents`); other values run as `react`.
    """
    model: str | None = Field(default=None, description="Defaults to the project's model.")
    fallback_models: list[str] = Field(default_factory=list)
    """Models tried in order when the main model fails."""
    system_prompt: str = Field(
        default="prompts/system.md", description="Path relative to the agent's folder."
    )
    retrieval: RetrievalSpec | None = None
    """Search the project's collections."""
    subagents: list[str] = Field(
        default_factory=list, description="Other agents in the project this agent can delegate to."
    )
    output_schema: str | None = Field(
        default=None,
        description="'file.py:Model' in the agent's folder; the agent returns that structure.",
        pattern=r"^[\w./-]+\.py:\w+$",
    )
    tools: ToolsSpec = Field(default_factory=ToolsSpec)
    """Functions and MCP tools the agent can call."""
    memory: MemorySpec = Field(default_factory=MemorySpec)
    """Conversation memory."""
    guardrails: list[Any] = Field(
        default_factory=list,
        description="pii, prompt_injection or deny guardrails; see ragkit.agents.guardrails.",
    )
    hitl: HitlSpec = Field(default_factory=HitlSpec)
    """Human approval of tool calls."""
    budget: BudgetSpec = Field(default_factory=BudgetSpec)
    """Limits for each request."""
    evals: EvalSpec = Field(default_factory=EvalSpec)
    """Dataset and gates for `ragkit eval agent`."""

    @field_validator("guardrails", mode="before")
    @classmethod
    def _parse_guardrails(cls, value: Any) -> list[Any]:
        return parse_guardrails(value or [])

    @model_validator(mode="after")
    def _pattern_requirements(self) -> Self:
        if self.pattern == "rag-chat" and self.retrieval is None:
            raise ValueError("pattern 'rag-chat' needs a retrieval section")
        if self.pattern == "supervisor" and not self.subagents:
            raise ValueError("pattern 'supervisor' needs subagents")
        if self.name in self.subagents:
            raise ValueError("an agent cannot list itself as a subagent")
        return self

    @classmethod
    def from_file(cls, path: Path) -> Self:
        try:
            data = yaml.safe_load(path.read_text()) or {}
            return cls.model_validate(data)
        except FileNotFoundError as exc:
            raise ConfigError(f"Agent manifest not found: {path}") from exc
        except (yaml.YAMLError, ValidationError) as exc:
            raise ConfigError(f"Invalid agent manifest {path}:\n{exc}") from exc

    def unwired_features(self) -> list[str]:
        """Manifest sections that are declared but not yet acted on by the runtime."""
        unwired = []
        if self.pattern not in ("react", "rag-chat", "supervisor"):
            unwired.append(f"pattern '{self.pattern}' (runs as 'react')")
        if self.memory.long_term:
            unwired.append("memory.long_term")
        return unwired
