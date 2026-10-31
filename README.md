# ragkit

A framework for building scalable RAG and agent applications. You point it at
your data sources, it ingests them incrementally into a hybrid (vector +
keyword) index, and you copy in agent templates that search that index. Agents
and search are served over HTTP.

> Status: **M4 (scale)**. M0 delivered the workspace, runtime, server and CLI;
> M1 ingestion, hybrid search and reranking; M2 MCP tools, human approval,
> structured output and supervisors; M3 tracing, cost budgets, evals and
> guardrails. M4 adds API keys and JWT auth with tenant isolation, a durable job
> queue with ingestion workers and scheduled syncs, query and result caching,
> and deployment: a container image, a Compose stack and a Helm chart with KEDA.

## Quickstart

```bash
uv sync
cd examples/quickstart
cp .env.example .env                       # add your GOOGLE_API_KEY
uv run ragkit ingest                       # index data/ into the "docs" collection
uv run ragkit search docs "where is my data stored?"
uv run ragkit chat rag-chat "How long is a password reset link valid?"
uv run ragkit eval agent rag-chat          # golden set, LLM judge, gates from agent.yaml
uv run ragkit eval search docs evals/search.jsonl --mode sparse --mode hybrid
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
sub-agents yet. `memory.long_term` is in the schema but not acted on yet; the
runtime logs a warning when it is set.

### Guardrails

```yaml
guardrails:
  - pii                                    # redact email, card, IP, phone, IBAN in input and output
  - type: pii
    kinds: [email, employee_id]
    patterns: {employee_id: "EMP-\\d{5}"}
    strategy: mask                         # redact | mask | hash
    apply_to: [input, output, tool_results]
  - type: prompt_injection                 # screens tool results and retrieved passages
    action: annotate                       # annotate (mark as untrusted data) | drop
  - type: deny                             # refuse without calling the model
    patterns: ['\bmedical advice\b']
    message: I can't give medical advice.
budget:
  max_cost_usd: 0.05                       # needs a price for the model under pricing
```

Every guardrail action is a `ragkit.guardrail` span event and is listed in the
API response's `usage.guardrails`.

## Observability

Spans follow the OpenTelemetry GenAI conventions: `invoke_agent <agent>` contains
`chat <model>` (tokens, cost, finish reason) and `execute_tool <tool>`. Retrieval
(`retrieve <collection>`, `rerank`, `embeddings`) and sub-agents nest under the
step that triggered them, and `ingest <collection>` covers ingestion runs.
Conversation ids are set as `gen_ai.conversation.id` and `langfuse.session.id`.

```yaml
observability:
  exporter: langfuse        # none | console | otlp | langfuse | phoenix
  capture_content: true     # prompts, replies, tool arguments and results
pricing:
  "google_genai:gemini-3.5-flash-lite": {input_per_million: 0.10, output_per_million: 0.40}
```

Langfuse reads `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`; `phoenix`
defaults to `http://localhost:6006/v1/traces`; `otlp` to
`http://localhost:4318/v1/traces`. If the process already has an OpenTelemetry
tracer provider, ragkit adds its exporter to it. HTTP requests are traced too.

## Evals

`agents/<name>/evals/golden.jsonl`, one case per line:

```json
{"input": "How long is a reset link valid?", "reference": "30 minutes.",
 "expected_tools": ["search_knowledge"], "expected_sources": ["password.md"]}
```

| Metric | Measured when the case has |
|---|---|
| `tool_recall` | `expected_tools` (an empty list expects no tool calls) |
| `source_recall` | `expected_sources`, matched against retrieved passage URIs |
| `field_accuracy` | a JSON object `reference` (structured output) |
| `correctness` | a text `reference`, graded by an LLM judge |
| `groundedness` | retrieval during the run, graded by an LLM judge |
| `error_rate`, `latency_p50_s`, `latency_p95_s`, `cost_usd` | always (`cost_usd` when priced) |

Gates in `agent.yaml` (`evals.gate: {correctness: 0.8, max_latency_p95_s: 5}`) make
`ragkit eval agent` exit 1 when a metric misses its threshold, so it can run in CI.
Approval requests are rejected during evals by default (`evals.on_approval`).
`ragkit eval search <collection> <dataset>` reports `hit_rate`, `recall` and `mrr`
per search mode. Reports are written to `.ragkit/evals/`.

```yaml
# .github/workflows/evals.yml (excerpt)
- run: uv run ragkit ingest
- run: uv run ragkit eval agent rag-chat
- run: uv run ragkit eval search docs evals/search.jsonl --gate hit_rate=0.9
```

## Running at scale

`examples/production` is the reference layout: Postgres holds the index, the job
queue and agent memory, so API replicas and workers are stateless.

| Piece | How |
|---|---|
| Auth | `auth.mode: api_key` (keys from env vars, each with a tenant, principals and scopes) or `jwt` (HS256 secret or a JWKS URL; tenant, groups and scopes from claims). Scopes: `chat`, `search`, `ingest`, `admin`. Optional per-caller `rate_limit_per_minute` |
| Tenants | A collection's documents belong to its `tenant`; callers only search their own tenant's collections plus `shared: true` ones, filtered by ACL principals. Agent searches run as the caller. Conversation threads are private to each tenant and user |
| Ingestion jobs | `POST /collections/{name}/ingest` or `ragkit ingest --queue` queues a job; `ragkit worker` runs jobs with renewed leases (a crashed worker's job is retried, up to `queue.max_attempts`), one job per collection at a time, and finishes running jobs on SIGTERM. Requests for an already-queued collection are merged |
| Scheduled syncs | `sync_every: 6h` on a collection; `ragkit scheduler` (or workers without `--no-schedule`) queues it when due |
| Caching | Query embeddings (LRU, on by default); identical search results for `cache.search_ttl_seconds` (off by default). Keys include tenant and principals |
| Probes | `GET /health` (liveness), `GET /ready` (agents loaded, store reachable) |

```bash
docker build -f deploy/Dockerfile --build-arg PROJECT=examples/production -t ragkit-app .
cp examples/production/.env.example deploy/compose/.env   # fill in keys
docker compose -f deploy/compose/docker-compose.yml up -d  # Postgres, 2 APIs, 2 workers, scheduler, gateway
helm install app deploy/helm/ragkit --set image.repository=ghcr.io/you/ragkit-app
```

The Helm chart deploys the API (HPA, PDB, probes, optional Ingress), workers
scaled by KEDA on queued and running jobs (down to zero), and one scheduler. Pods
run as non-root with a read-only root filesystem. Secrets come from
`existingSecret`: `DATABASE_URL`, model keys, the API key variables, and
`KEDA_POSTGRES_CONNECTION` for KEDA. Schema setup is serialised with a Postgres
advisory lock, so many pods can start at once.

## HTTP API

| Endpoint | |
|---|---|
| `GET /agents`, `POST /agents/{name}/chat`, `POST /agents/{name}/stream` (SSE) | Agents. Replies carry `status` (`completed` or `waiting_for_approval`), `pending` actions, `structured` output and `usage` (tokens, cost, guardrails) |
| `POST /agents/{name}/resume`, `POST /agents/{name}/resume/stream` | Continue with `{"thread_id", "decisions": [{"type": "approve" \| "edit" \| "reject", "args", "message"}]}`, one per pending action. A thread that is waiting rejects new messages with 409 |
| `GET /collections`, `POST /collections/{name}/search` | Search with `query`, `top_k`, `mode`, `filters`, `rerank` |
| `POST /collections/{name}/ingest`, `GET /jobs`, `GET /jobs/{id}` | Queue ingestion and follow jobs (`ingest` scope) |
| `GET /health`, `GET /ready` | Liveness and readiness |

Send `Authorization: Bearer <key or JWT>` (or `X-API-Key: <key>`) when `auth.mode` is not `none`.

## Layout

| Package | Import | Contents |
|---|---|---|
| `packages/core` | `ragkit.core` | Types, plugin protocols, registry, `ragkit.yaml` config |
| `packages/retrieval` | `ragkit.retrieval` | Embedders, SQLite and Postgres stores, hybrid search, rerankers, `KnowledgeBase` |
| `packages/ingest` | `ragkit.ingest` | Connectors, parsers, chunker, incremental pipeline, job queue, workers |
| `packages/agents` | `ragkit.agents` | Manifest schema, LangGraph runtime, MCP, approvals, sub-agents, guardrails, telemetry, templates |
| `packages/observe` | `ragkit.observe` | Tracing exporters (Langfuse, Phoenix, OTLP, console) |
| `packages/evals` | `ragkit.evals` | Agent and search evals, LLM judge, gates |
| `packages/server` | `ragkit.server` | FastAPI gateway, auth, rate limits |
| `packages/cli` | `ragkit.cli` | `init`, `ingest`, `search`, `collections`, `templates`, `add agent`, `agents`, `chat`, `eval`, `serve`, `worker`, `scheduler`, `jobs` |
| `deploy/` | | Dockerfile, Compose stack (`compose/`), Helm chart with KEDA (`helm/ragkit`), Postgres for development |

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
| M3 Harness (done) | OpenTelemetry GenAI tracing (Langfuse, Phoenix, OTLP), token and cost accounting, cost budgets, `ragkit eval` for agents and search with CI gates, PII / prompt-injection / deny guardrails |
| M4 Scale (done) | API keys and JWT, scopes, rate limits, tenant isolation, durable job queue, workers, scheduler, caching, Dockerfile, Compose stack, Helm chart with KEDA |
