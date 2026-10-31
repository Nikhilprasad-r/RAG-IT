# Your first project

This walk-through builds a documentation assistant from scratch.

## 1. Create the project

```bash
uv run ragkit init handbook
cd handbook
cp .env.example .env            # add GOOGLE_API_KEY
```

`init` creates:

| Path | Purpose |
|---|---|
| `ragkit.yaml` | Project settings: store, embedding model, collections, server |
| `data/` | Documents for the starter `docs` collection |
| `agents/` | Your agents, one folder each |
| `.env.example` | API keys and secrets to copy into `.env` |

## 2. Add documents

Copy Markdown, PDF, Word, HTML or text files into `data/`. The starter
collection reads them:

```yaml
# ragkit.yaml
collections:
  docs:
    description: Company handbook
    sources:
      - type: files
        path: data
        include: ["**/*.md", "**/*.pdf", "**/*.docx", "**/*.html", "**/*.txt"]
```

To add a website instead, or as well:

```yaml
# ragkit.yaml
collections:
  docs:
    sources:
      - type: files
        path: data
      - type: web
        urls: [https://docs.example.com/]
        max_pages: 200
```

## 3. Ingest

```bash
uv run ragkit ingest
uv run ragkit collections        # documents, chunks and embedding model per collection
```

## 4. Add an agent

```bash
uv run ragkit templates          # list the built-in templates
uv run ragkit add agent rag-chat
```

This copies the template to `agents/rag-chat/`:

```
agents/rag-chat/
├─ agent.yaml          # model, retrieval, memory, budget, evals
├─ prompts/system.md   # instructions
├─ evals/golden.jsonl  # test questions
└─ README.md
```

The template searches a collection named `docs`, which matches the starter
project. Edit `prompts/system.md` to change tone or rules.

## 5. Chat and serve

```bash
uv run ragkit chat rag-chat "What is the leave policy?"
uv run ragkit serve
```

## 6. Write evals

Replace the placeholder lines in `agents/rag-chat/evals/golden.jsonl` with real
questions and answers:

```json
{"input": "How many days of leave do employees get?", "reference": "25 days.", "expected_sources": ["leave-policy.md"]}
```

Add a gate to `agent.yaml` so the eval fails when quality drops:

```yaml
# agent.yaml
evals:
  dataset: evals/golden.jsonl
  gate: {correctness: 0.8, source_recall: 0.8}
```

```bash
uv run ragkit eval agent rag-chat
```

## Where next

- More data sources: [Ingesting data](../guides/ingestion.md)
- Agents that use tools: [Agents and templates](../guides/agents-and-templates.md)
- Tracing and cost: [Observability](../guides/observability.md)
- Production: [Production setup](../deploy/production.md)
