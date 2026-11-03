# ragkit

ragkit is a framework for building retrieval-augmented generation (RAG) and agent
applications that can grow from a laptop to a cluster without rewrites.

You describe a project in YAML: where the documents come from, how to index them,
and which agents answer questions. Agents are copied into your project from
templates, so their prompts, tools and settings are ordinary files you own and edit.

```bash
ragkit init my-app && cd my-app
ragkit ingest                         # index the documents in data/
ragkit add agent rag-chat             # copy in a template
ragkit chat rag-chat "How do I reset my password?"
ragkit serve                          # HTTP API with streaming
```

## What it does

| Area | What you get |
|---|---|
| **Ingestion** | Connectors for folders, websites, GitHub, Confluence, Notion and your own code; parsers for text, Markdown, HTML, PDF, Word, CSV and JSON (plus Docling for complex layouts); structure-aware chunking; incremental syncs that skip unchanged items |
| **Search** | Hybrid search (embeddings plus BM25) fused with Reciprocal Rank Fusion, optional reranking (Cohere, Voyage or a local cross-encoder), metadata filters, ACLs and tenants; SQLite locally, Postgres with pgvector at scale |
| **Agents** | Built on LangGraph: single-pass RAG, agentic retrieval, tool use, MCP servers, human approval of tool calls, structured output and multi-agent supervisors |
| **Harness** | OpenTelemetry tracing (Langfuse, Phoenix, any OTLP backend), token and cost accounting with budgets, evals with CI gates, and PII, prompt-injection and deny guardrails |
| **Scale** | API keys and JWT, scopes, rate limits, tenant isolation, a durable job queue with workers and scheduled syncs, caching, a container image, Docker Compose and a Helm chart with KEDA autoscaling |

## Where to start

- New here? Follow the [Quickstart](getting-started/quickstart.md), then build
  [your first project](getting-started/first-project.md).
- Want the big picture? Read [Architecture](concepts/architecture.md).
- Looking for a setting? See the [ragkit.yaml](reference/project-config.md) and
  [agent.yaml](reference/agent-manifest.md) references.
- Going to production? Start with [Production setup](deploy/production.md).
