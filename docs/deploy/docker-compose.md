# Docker Compose

`deploy/compose/docker-compose.yml` runs the production example as a small
cluster on one machine:

| Service | What it runs |
|---|---|
| `postgres` | Postgres 18 with pgvector |
| `api` (2 replicas) | `ragkit serve` |
| `worker` (2 replicas) | `ragkit worker --no-schedule --concurrency 2` |
| `scheduler` | `ragkit scheduler` |
| `gateway` | Caddy on port 8080, round-robin across API replicas, unbuffered for streaming |

## Run it

```bash
cp examples/production/.env.example deploy/compose/.env
# edit deploy/compose/.env: GOOGLE_API_KEY, RAGKIT_KEY_APP, RAGKIT_KEY_OPS
docker compose -f deploy/compose/docker-compose.yml up --build -d
curl localhost:8080/ready
```

The scheduler queues the first sync of `docs` on its own (it has never been
synced), and a worker ingests it within seconds:

```bash
source deploy/compose/.env
curl -s localhost:8080/jobs -H "X-API-Key: $RAGKIT_KEY_OPS"
```

Then search and chat through the gateway:

```bash
curl -s localhost:8080/collections/docs/search -H "X-API-Key: $RAGKIT_KEY_APP" \
  -H 'content-type: application/json' -d '{"query": "can I change my region?"}'

curl -s localhost:8080/agents/rag-chat/chat -H "X-API-Key: $RAGKIT_KEY_APP" \
  -H 'content-type: application/json' \
  -d '{"message": "How long is a reset link valid?", "thread_id": "demo"}'
```

Follow-up messages with the same `thread_id` may land on the other API replica;
the conversation continues because memory is in Postgres.

## Use your own project

Point the build at your project folder:

```yaml
x-app: &app
  build:
    context: ../..
    dockerfile: deploy/Dockerfile
    args: {PROJECT: path/to/your-project}
```

## Operate it

```bash
docker compose -f deploy/compose/docker-compose.yml up -d --scale worker=4
docker compose -f deploy/compose/docker-compose.yml logs -f worker
docker compose -f deploy/compose/docker-compose.yml down        # keeps the database volume
docker compose -f deploy/compose/docker-compose.yml down -v     # deletes it
```

Workers have a five-minute stop grace period, so running jobs finish during
`down` or a redeploy.
