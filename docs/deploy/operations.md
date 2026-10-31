# Operations

## Ingestion jobs

Ingestion runs as **jobs** in a queue stored next to the index (Postgres in
production, a SQLite file locally).

```bash
ragkit ingest --queue             # queue every collection
ragkit ingest docs --queue --full # queue a full re-ingest of one
ragkit jobs                       # recent jobs
ragkit jobs --collection docs
ragkit worker                     # run jobs until stopped
ragkit worker --until-idle        # run jobs until the queue is empty, then exit
```

Over HTTP (`ingest` scope): `POST /collections/{name}/ingest` with
`{"full": true}` optional, then `GET /jobs/{id}`.

| Guarantee | How |
|---|---|
| No duplicate work | Queuing a collection that already has a queued job returns that job (a `full` request upgrades it) |
| One run per collection | A collection's next job waits while one is running |
| Crash recovery | A running job holds a lease its worker renews; if the worker dies, the lease expires and another worker retries the job |
| Bounded retries | A job is tried `queue.max_attempts` times (default 3), then marked `failed` |
| Clean shutdown | On SIGTERM or SIGINT a worker stops claiming and finishes its running jobs |

A job's status is `queued`, `running`, `succeeded` or `failed`. A succeeded job
can still report items that failed to parse or embed; check its report.

## Scheduled syncs

```yaml
# ragkit.yaml
collections:
  docs:
    sync_every: 6h          # s, m, h or d
    sources: [{type: files, path: data}]
```

`ragkit scheduler` checks every minute (`--every` to change it) and queues each
collection whose last request is older than its interval. Workers schedule too
unless started with `--no-schedule`. Running several schedulers is safe: requests
for a queued collection are merged.

## Scaling

| Part | Scales by | Notes |
|---|---|---|
| API | Replicas, `ragkit serve --workers N` | Stateless with the postgres checkpointer |
| Workers | Replicas, `--concurrency N` | Embedding API rate limits are usually the ceiling |
| Postgres | Instance size, read replicas for heavy search | HNSW indexes live in memory |

Embedding calls are the main cost and latency of ingestion. Unchanged items are
never re-embedded, so after the first run a sync costs little.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `Collection 'docs' was indexed with X but the project now uses Y` | `embeddings.model` changed; run `ragkit ingest docs --full` |
| Searches return nothing for a caller | Their tenant or principals do not match the documents; check the collection's `tenant`, `shared` and `acl` |
| `409 Thread ... is waiting for approval` | A tool call is pending; resume the thread |
| Ingest reports `source failed` | The source was unreachable; items it already had are kept |
| Workers idle while jobs are queued | Another job for the same collection is running, or a lease from a crashed worker has not expired yet (`queue.lease_seconds`) |
| `Environment variable X is referenced but not set` | A `${X}` in `ragkit.yaml` has no value and no default |
| Startup fails naming an agent | The message names the missing tool, collection, MCP server or schema |

Run any command with `-v` for more logging, and turn on tracing to see each step
of a slow or wrong answer.
