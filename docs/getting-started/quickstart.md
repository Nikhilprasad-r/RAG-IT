# Quickstart

The repository includes `examples/quickstart`: four short documents about a
fictional product, a `rag-chat` agent, a `knowledge-agent` and a `support-agent`.
You need a Gemini API key.

```bash
cd examples/quickstart
cp .env.example .env            # set GOOGLE_API_KEY
```

## Index the documents

```bash
uv run ragkit ingest
```

ragkit reads `data/`, splits each document into chunks, embeds them and stores them
in `.ragkit/index.db`. Run it again and every file is reported as *unchanged*:
only new or edited files are processed.

## Search

```bash
uv run ragkit search docs "where is my data stored?"
```

Results show the source file, the heading path inside it and a score. Search is
hybrid: it combines meaning (embeddings) with exact words (BM25).

## Ask an agent

```bash
uv run ragkit chat rag-chat "How long is a password reset link valid?"
```

`rag-chat` retrieves the most relevant passages and answers with numbered
citations. Try `knowledge-agent` for questions that need several searches, and
`support-agent`, which pauses for your approval before issuing a refund:

```bash
uv run ragkit chat support-agent "Order 1042 arrived late, please refund 10 EUR"
```

## Serve it

```bash
uv run ragkit serve
```

Open <http://127.0.0.1:8000/docs> for interactive API docs, or call it directly:

```bash
curl -s localhost:8000/agents/rag-chat/chat \
  -H 'content-type: application/json' \
  -d '{"message": "Which plan includes SSO?"}'
```

## Measure quality

```bash
uv run ragkit eval agent rag-chat
uv run ragkit eval search docs evals/search.jsonl --mode sparse --mode hybrid
```

The first runs `rag-chat` against its golden dataset and grades the answers with
an LLM judge; the second measures how often search finds the right document.

Next: [build your own project](first-project.md).
