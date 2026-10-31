# Extending ragkit

Each stage has a small interface in `ragkit.core.interfaces`. Implementations are
structural (`Protocol`s): a class only needs the right attributes and methods.

| Kind | Interface | Registered under |
|---|---|---|
| Connector | `name`, async `fetch()` yielding `RawDocument` | `ragkit.connector` |
| Reranker | `name`, async `rerank(query, candidates, top_n)` | `ragkit.reranker` |
| Tool | a function or LangChain tool | `tools.py` or MCP |

Connectors are covered in [Custom connectors](../guides/custom-connectors.md).

## The registry

```python
from ragkit.core import registry

registry.register("reranker", "my-reranker", MyRerankerFactory)
factory = registry.get("reranker", "my-reranker")
registry.names("connector")  # built-in and installed names
```

Plugins can also come from installed packages through entry points in the
`ragkit.<kind>` groups. They are imported only when first used:

```toml title="pyproject.toml"
[project.entry-points."ragkit.reranker"]
my-reranker = "acme_ragkit.rerank:make_reranker"

[project.entry-points."ragkit.connector"]
crm = "acme_ragkit.crm:CrmArticles"
```

## A custom reranker

A reranker factory receives the `RerankConfig` from `ragkit.yaml`:

```python
from collections.abc import Sequence

from ragkit.core.config import RerankConfig
from ragkit.core.types import Query, ScoredChunk


class KeywordBoost:
    name = "keyword-boost"

    def __init__(self, config: RerankConfig) -> None:
        self.model = config.model

    async def rerank(
        self, query: Query, candidates: Sequence[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]:
        words = set(query.text.lower().split())
        scored = [
            c.model_copy(update={"score": c.score + sum(w in c.chunk.text.lower() for w in words)})
            for c in candidates
        ]
        return sorted(scored, key=lambda c: -c.score)[:top_n]


def make_reranker(config: RerankConfig) -> KeywordBoost:
    return KeywordBoost(config)
```

```yaml
# ragkit.yaml
rerank:
  provider: keyword-boost
```

## Embedding models

Embedding models are LangChain integrations named `provider:model`. To use another
provider, install its LangChain package; no plugin is needed.

## Agents in your own application

```python
from ragkit.agents import AgentRuntime
from ragkit.core.config import ProjectConfig

project = ProjectConfig.load("path/to/project")
async with AgentRuntime(project, only=["rag-chat"]) as runtime:
    result = await runtime.get("rag-chat").ainvoke("How do I reset my password?")
    print(result.reply, result.usage.cost_usd)
```

`create_app(project)` from `ragkit.server` returns the FastAPI app, to mount in a
larger application. See the [Python API](../reference/python-api.md).
