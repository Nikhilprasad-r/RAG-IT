# Production setup

In production, Postgres holds the index, the ingestion job queue and agent
memory. API servers and workers then keep no state of their own, so you can run
as many as you need and replace them freely.

`examples/production` is a complete reference project.

## The project

```yaml
# ragkit.yaml
name: production
database_url: ${DATABASE_URL}

store:
  backend: postgres

embeddings:
  model: google_genai:gemini-embedding-2
  dimensions: 768

collections:
  docs:
    description: Product docs
    tenant: acme
    sync_every: 6h
    sources:
      - type: files
        path: data

auth:
  mode: api_key
  rate_limit_per_minute: 120
  api_keys:
    - {key_env: RAGKIT_KEY_APP, name: acme-app, tenant: acme, scopes: [chat, search]}
    - {key_env: RAGKIT_KEY_OPS, name: acme-ops, tenant: acme, scopes: [chat, search, ingest]}

cache:
  search_ttl_seconds: 30

server:
  host: 0.0.0.0
  port: 8000
```

Agents that keep conversations must use the shared checkpointer:

```yaml
# agent.yaml
name: rag-chat
pattern: rag-chat
retrieval: {collections: [docs]}
memory:
  checkpointer: postgres
```

## Checklist

| Item | Why |
|---|---|
| `store.backend: postgres` and `database_url` | One index shared by every process |
| `memory.checkpointer: postgres` in conversational agents | Threads and pending approvals survive restarts and work on any replica |
| `auth.mode: api_key` or `jwt` | The API is open by default |
| Collections have a `tenant` | Isolation between customers or teams |
| Secrets in the environment, not in `ragkit.yaml` | `${VAR}` and `*_env` settings |
| `pricing` and `budget.max_cost_usd` | Cost visibility and runaway protection |
| `observability.exporter` | Traces for debugging and cost tracking |
| Eval gates in CI | Catch quality regressions before deploying |
| Postgres backups | It holds everything; chunks can be rebuilt by re-ingesting, conversations cannot |

## Processes

| Process | Command | How many |
|---|---|---|
| API | `ragkit serve --host 0.0.0.0` | Two or more, behind a load balancer |
| Workers | `ragkit worker --no-schedule` | Scaled with the queue; zero when idle |
| Scheduler | `ragkit scheduler` | One (only needed for `sync_every`) |

`ragkit worker` without `--no-schedule` also schedules, which is enough when at
least one worker always runs.

## Build the image

One image runs every process. It contains the installed packages and your project
folder:

```bash
docker build -f deploy/Dockerfile --build-arg PROJECT=examples/production -t ragkit-app .
```

The image runs as a non-root user, listens on port 8000 and includes a health
check. Its default command is `ragkit serve`; pass `ragkit worker` or
`ragkit scheduler` to run the others.

## Health checks

| Endpoint | Meaning |
|---|---|
| `GET /health` | The process is up (liveness) |
| `GET /ready` | Agents are loaded and the store answers (readiness) |

## Start-up

Each process creates any missing tables when it starts. Schema creation is
serialised with a Postgres advisory lock, so many pods can start at the same time.
The Postgres user needs permission to create tables and, once, the `vector`
extension; afterwards a user without those rights works.

Next: run it locally with [Docker Compose](docker-compose.md) or deploy to
[Kubernetes](kubernetes.md).
