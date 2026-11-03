# Projects and configuration

A **project** is a folder containing `ragkit.yaml`. Commands find it by looking in
the current directory and then its parents, so you can run them from any subfolder.

```
my-app/
├─ ragkit.yaml          # project settings
├─ .env                 # secrets, loaded automatically (keep out of git)
├─ data/                # documents for a files source
├─ agents/              # one folder per agent
│  └─ rag-chat/agent.yaml
├─ evals/               # optional search eval datasets
└─ .ragkit/             # local index, job queue and eval reports (generated)
```

## ragkit.yaml

A minimal file needs only a name:

```yaml
# ragkit.yaml
name: my-app
```

Everything else has defaults. A typical local project:

```yaml
# ragkit.yaml
name: my-app
default_model: google_genai:gemini-3.5-flash-lite

store:
  backend: sqlite

embeddings:
  model: google_genai:gemini-embedding-2
  dimensions: 768

collections:
  docs:
    description: Product documentation
    sources:
      - type: files
        path: data
```

Every key is listed in the [ragkit.yaml reference](../reference/project-config.md).
Unknown keys are rejected with an error that names them, so typos do not pass
silently.

## Secrets and environment variables

Any string can reference an environment variable:

```yaml
# ragkit.yaml
name: my-app
database_url: ${DATABASE_URL}
observability:
  exporter: ${TRACE_EXPORTER:-none}
```

`${VAR}` fails with a clear error when the variable is unset;
`${VAR:-default}` falls back to the default. The project's `.env` is loaded before
the file is read, without overriding variables already set in the environment.

Settings that hold secrets, such as API keys for `auth`, connector tokens and
Langfuse keys, take the **name of an environment variable** (`key_env`,
`token_env`, `secret_env`) rather than the secret itself.

## Models

Chat and embedding models are named `provider:model` and loaded through LangChain:

| Setting | Example |
|---|---|
| `default_model` | `google_genai:gemini-3.5-flash-lite`, `openai:gpt-5.5`, `anthropic:claude-sonnet-5-5` |
| `embeddings.model` | `google_genai:gemini-embedding-2`, `openai:text-embedding-3-small`, `ollama:nomic-embed-text` |

`embeddings.model: hash:256` is an offline embedder for tests and demos; it
matches words, not meaning.

Changing the embedding model after ingesting is detected: searches and ingests
refuse to mix vectors, and `ragkit ingest --full` rebuilds the collection.

## Agents

Each folder under `agents/` with an `agent.yaml` is an agent. See
[Agents](agents.md) and the [agent.yaml reference](../reference/agent-manifest.md).
