# ragkit

A framework for building scalable RAG and agent applications. You point it at
your data sources, it ingests them incrementally into a hybrid (vector +
keyword) index, and you copy in agent templates that search that index. Agents
and search are served over HTTP.

> Status: **M2 (agents)**. M0 delivered the workspace, runtime, server and CLI;
> M1 ingestion, hybrid search and reranking. M2 adds MCP tools, human approval,
> structured output, multi-agent supervisors, the `sql-analyst`, `extractor` and
> `supervisor` templates, templates from GitHub or local folders, and Confluence
> and Notion connectors.

## Quickstart

```bash
uv sync
cd examples/quickstart
cp .env.example .env                       # add your GOOGLE_API_KEY
uv run ragkit ingest                       # index data/ into the "docs" collection
uv run ragkit search docs "where is my data stored?"
uv run ragkit chat rag-chat "How long is a password reset link valid?"
uv run ragkit serve                        # http://127.0.0.1:8000/docs
```

Start a new project:

```bash
uv run ragkit init my-app && cd my-app     # creates ragkit.yaml, agents/ and data/
uv run ragkit templates
uv run ragkit add agent rag-chat
```

## Data: collections and sources

`ragkit.yaml` defines collections. Each collection has one or more sources:

```yaml
store:
  backend: sqlite            # sqlite (one file, no server) | postgres (pgvector)
embeddings:
  model: google_genai:gemini-embedding-2   # any LangChain provider:model
  dimensions: 768
rerank:                      # optional
  provider: cohere           # cohere | voyage | cross-encoder
collections:
  docs:
    description: Product documentation
    acl: []                  # principals allowed to read; empty = everyone
    parser: basic            # basic | docling
    sources:
      - type: files
        path: data
        include: ["**/*.md", "**/*.pdf"]
      - type: web
        urls: [https://docs.example.com/]
        max_pages: 200
      - type: github
        repo: owner/name
      - type: confluence                   # CONFLUENCE_EMAIL + CONFLUENCE_API_TOKEN
        base_url: https://acme.atlassian.net/wiki
        spaces: [ENG]
      - type: notion                       # NOTION_TOKEN; pages shared with the integration
        database_ids: [abc123]
      - type: my_connectors:CrmSource      # your own class, imported from the project
        region: eu
    chunking: { max_tokens: 400, overlap_tokens: 40 }
```

| Stage | What it does |
|---|---|
| Connectors | `files` (local folders), `web` (breadth-first crawl, robots.txt, same-site scope), `github` (repo files through the API), `confluence` (Cloud v2 API, spaces), `notion` (databases, page trees or everything shared, converted to Markdown), or any `module:Class` with a `name` and an async `fetch()`. Hosted connectors retry on rate limits |
| Parsers | Text, Markdown, code, HTML, PDF (with page numbers), DOCX, CSV and JSON built in. `parser: docling` adds layout-aware PDF (tables, page numbers), PPTX, XLSX and image OCR (`ragkit-ingest[docling]`). Docling loads its models in about 10 s per process, then converts about 1.5 s per PDF page on a laptop; office formats are near-instant |
| Chunking | Splits along headings, packs paragraphs up to `max_tokens`, and records the heading path and page. A title and section header is indexed with each chunk |
| Incremental sync | Unchanged items (same content hash) are skipped. Items removed from a source are deleted, but only when the source listed completely, so an outage never empties a collection. Failures are reported per item |
| Stores | **SQLite**: one file, FTS5 BM25 and NumPy cosine; good up to about 100k chunks. **Postgres**: pgvector HNSW per collection plus built-in full-text search; for scale and for several servers sharing one index (`deploy/docker-compose.yml`) |
| Search | Dense and BM25 run in parallel and are fused with Reciprocal Rank Fusion. An optional reranker follows. Tenant, ACL and metadata filters are applied in the store |

Changing the embedding model is detected: `ragkit ingest --full` rebuilds the collection.

## Agents

Each agent is a folder under `agents/`, copied from a template:

| Template | Pattern | Use |
|---|---|---|
| `rag-chat` | single-pass RAG | Retrieves once per message and answers with numbered citations. Cheapest and fastest |
| `knowledge-agent` | agentic retrieval | Gets a `search_knowledge` tool and searches as often as it needs. For multi-step questions |
| `support-agent` | tools + approval | Looks up orders; refunds pause until a person approves, edits or rejects them |
| `sql-analyst` | tools | Explores a schema and answers with read-only SQL (`SQL_ANALYST_DATABASE_URL`) |
| `extractor` | structured output | Returns a validated Pydantic object (an invoice by default) |
| `supervisor` | multi-agent | Delegates to other agents (`ask_<name>` tools) and combines their answers |

`ragkit add agent` also installs templates from a folder or GitHub
(`gh:owner/repo/path@ref` or a github.com tree URL). Python from those
templates runs with your permissions, so the CLI asks you to review it.

```yaml
# agent.yaml (excerpt)
tools:
  local: [lookup_order, issue_refund]       # functions in tools.py
  mcp:
    - server: tickets                       # defined under mcp_servers in ragkit.yaml
      allow: [create_ticket]                # empty = every tool the server offers
hitl:
  require_approval: [issue_refund, create_ticket]
  decisions: [approve, edit, reject]
subagents: [knowledge-agent]                # becomes an ask_knowledge_agent tool
output_schema: schema.py:Invoice            # Pydantic model in the agent folder
```

```yaml
# ragkit.yaml
mcp_servers:
  tickets:
    transport: http                         # stdio | http | sse | websocket
    url: https://mcp.example.com/mcp
    headers: {Authorization: "Bearer ${TICKETS_TOKEN}"}
  files:
    command: npx                            # stdio is the default transport
    args: ["-y", "@modelcontextprotocol/server-filesystem", "./data"]
```

MCP sessions open once per process and stay open. `ragkit chat` loads only the
agent you talk to and its sub-agents. Agents that need approval cannot be
sub-agents yet. `guardrails`, `memory.long_term` and `budget.max_cost_usd` are in
the schema but not acted on yet; the runtime logs a warning when they are set.

## HTTP API

| Endpoint | |
|---|---|
| `GET /agents`, `POST /agents/{name}/chat`, `POST /agents/{name}/stream` (SSE) | Agents. Replies carry `status` (`completed` or `waiting_for_approval`), `pending` actions and `structured` output |
| `POST /agents/{name}/resume`, `POST /agents/{name}/resume/stream` | Continue with `{"thread_id", "decisions": [{"type": "approve" \| "edit" \| "reject", "args", "message"}]}`, one per pending action. A thread that is waiting rejects new messages with 409 |
| `GET /collections`, `POST /collections/{name}/search` | Search with `query`, `top_k`, `mode`, `filters`, `rerank` |

## Layout

| Package | Import | Contents |
|---|---|---|
| `packages/core` | `ragkit.core` | Types, plugin protocols, registry, `ragkit.yaml` config |
| `packages/retrieval` | `ragkit.retrieval` | Embedders, SQLite and Postgres stores, hybrid search, rerankers, `KnowledgeBase` |
| `packages/ingest` | `ragkit.ingest` | Connectors, parsers, chunker, incremental pipeline |
| `packages/agents` | `ragkit.agents` | Manifest schema, LangGraph runtime, MCP, approvals, sub-agents, retrieval wiring, templates |
| `packages/server` | `ragkit.server` | FastAPI gateway |
| `packages/cli` | `ragkit.cli` | `init`, `ingest`, `search`, `collections`, `templates`, `add agent`, `agents`, `chat`, `serve` |

## Local models

Docling and the `cross-encoder` reranker (default `BAAI/bge-reranker-v2-m3`,
multilingual) run on your machine and need PyTorch, about 1 GB installed. Model
weights download on first use and are cached by Hugging Face; the bge reranker
is about 2.3 GB. Install them with:

```bash
uv sync --group local-models
```

A plain `uv sync` removes them again; their tests are skipped when they are missing.

## Development

```bash
uv run pytest
docker compose -f deploy/docker-compose.yml up -d
RAGKIT_TEST_DATABASE_URL=postgresql://ragkit:ragkit@localhost:5433/ragkit uv run pytest
uv run ruff check packages tests && uv run ruff format packages tests
```

## Roadmap

| Milestone | Scope |
|---|---|
| M0 Foundation (done) | Workspace, core interfaces, config, CLI, runtime, server, `support-agent` |
| M1 Ingestion (done) | Connectors (files, web, GitHub, custom), parsers, chunking, incremental sync, SQLite and pgvector stores, hybrid search, rerankers, RAG templates |
| M2 Agents (done) | MCP tools, human approval, structured output, supervisors, SQL analyst / extractor / supervisor templates, templates from GitHub and folders, Confluence and Notion connectors, verified Postgres checkpointer |
| M3 Harness | OpenTelemetry GenAI tracing, Langfuse, `ragkit eval` with CI gate, guardrails |
| M4 Scale | Queue-based ingestion workers, scheduled syncs, Helm + KEDA, multi-tenancy and auth, caching |
