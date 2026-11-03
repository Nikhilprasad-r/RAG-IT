# Installation

ragkit is a [uv](https://docs.astral.sh/uv/) workspace of Python packages. It needs
Python 3.12 or later.

!!! note
    The packages are not published to PyPI yet. Work from a clone of the repository,
    or use the [container image](../deploy/production.md#build-the-image) for deployments.

```bash
git clone https://github.com/Nikhilprasad-r/RAG-IT.git
cd RAG-IT
uv sync
uv run ragkit --help
```

`uv sync` creates `.venv/` with every package, the Gemini integration and the
Postgres drivers. Prefix commands with `uv run`, or activate the environment with
`source .venv/bin/activate`.

## Model providers

Models are named `provider:model`, for example `google_genai:gemini-3.5-flash-lite`
or `openai:gpt-5.5`, and loaded through LangChain. Gemini is installed by default.
For another provider, add its LangChain package and set its API key:

```bash
uv add langchain-openai        # or langchain-anthropic, langchain-ollama, ...
export OPENAI_API_KEY=...
```

API keys usually go in the project's `.env` file, which ragkit loads automatically.

## Optional extras

| Need | Install |
|---|---|
| Docling parsing and the local cross-encoder reranker (adds PyTorch, about 1 GB) | `uv sync --group local-models` |
| Building these docs | `uv sync --group docs` |

A plain `uv sync` removes groups you did not name, so pass every group you want
each time, for example `uv sync --group local-models --group docs`.

## Postgres (optional)

Local projects keep their index in a SQLite file and need no server. For the
Postgres store, a shared job queue or persistent agent memory, start Postgres with
pgvector:

```bash
docker compose -f deploy/docker-compose.yml up -d
# database_url: postgresql://ragkit:ragkit@localhost:5433/ragkit
```
