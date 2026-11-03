# Contributing

## Set up

```bash
uv sync --group docs               # add --group local-models for Docling and cross-encoder tests
uv run pytest
uv run ruff check packages tests docs && uv run ruff format packages tests docs
```

Tests use fake chat models and an offline `hash` embedder, so they need no API
keys. Some tests are skipped unless something optional is available:

| Tests | Run when |
|---|---|
| Postgres store, queue and checkpointer | `RAGKIT_TEST_DATABASE_URL` points at Postgres with pgvector |
| Docling and the cross-encoder | `uv sync --group local-models` |

```bash
docker compose -f deploy/docker-compose.yml up -d
RAGKIT_TEST_DATABASE_URL=postgresql://ragkit:ragkit@localhost:5433/ragkit uv run pytest
```

## Repository layout

| Path | Contents |
|---|---|
| `packages/*` | One package per concern; see [Architecture](concepts/architecture.md) |
| `packages/agents/src/ragkit/agents/templates/` | Built-in agent templates |
| `tests/` | The test suite |
| `examples/` | `quickstart` and `production` projects |
| `deploy/` | Dockerfile, Compose stack, Helm chart |
| `docs/` | This site |

## Documentation

```bash
uv run mkdocs serve                 # live preview at http://127.0.0.1:8000
uv run python docs/gen_reference.py # regenerate the reference pages
uv run mkdocs build                 # strict build: broken links fail
```

The `ragkit.yaml`, `agent.yaml`, CLI, HTTP and template reference pages are
generated from the code. After changing a config field, command, endpoint or
template, regenerate them; `tests/test_docs.py` fails while they are stale. Field
descriptions come from the docstring under each field.

YAML examples in the docs that start with `# ragkit.yaml` or `# agent.yaml` are
validated against the real models by the same test, so keep that first line on
complete examples.
