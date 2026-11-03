# Observability and cost

ragkit emits OpenTelemetry traces that follow the GenAI semantic conventions, so
they work with Langfuse, Arize Phoenix, Jaeger, Datadog, Grafana Tempo or any
OTLP backend.

```yaml
# ragkit.yaml
observability:
  exporter: langfuse            # none | console | otlp | langfuse | phoenix
  capture_content: true         # record prompts, replies and tool payloads
```

| Exporter | Sends to | Settings |
|---|---|---|
| `none` (default) | Nowhere; tracing is off | |
| `console` | Standard output | |
| `otlp` | `endpoint` (default `http://localhost:4318/v1/traces`, or `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`) | `headers` |
| `langfuse` | `<langfuse.host>/api/public/otel/v1/traces` | `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` |
| `phoenix` | `http://localhost:6006/v1/traces` | `endpoint` to change it |

All exporters use OTLP over HTTP. If your process already configures an
OpenTelemetry tracer provider, ragkit adds its exporter to it.

## What a trace contains

```text
POST /agents/knowledge-agent/chat               (HTTP server span)
└─ invoke_agent knowledge-agent                 tokens, cost, conversation id
   ├─ chat gemini-3.5-flash-lite                model, tokens, cost, finish reason
   ├─ execute_tool search_knowledge             tool name, call id
   │  └─ retrieve docs                          collection, mode, results
   │     ├─ embeddings gemini-embedding-2
   │     └─ rerank cohere
   └─ chat gemini-3.5-flash-lite
```

| Span | Key attributes |
|---|---|
| `invoke_agent <agent>` | `gen_ai.agent.name`, `gen_ai.conversation.id`, `langfuse.session.id`, total tokens and cost |
| `chat <model>` | `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.usage.input_tokens` / `output_tokens`, `gen_ai.usage.cost` |
| `execute_tool <tool>` | `gen_ai.tool.name`, `gen_ai.tool.call.id` |
| `retrieve <collection>` | `ragkit.search.mode`, `ragkit.search.top_k`, `ragkit.search.results` |
| `embeddings <model>`, `rerank <provider>` | model, number of inputs |
| `ingest <collection>`, `ingest_document`, `ingest_job` | counts and outcome per document |

With `capture_content: true`, spans also carry `gen_ai.input.messages`,
`gen_ai.output.messages`, tool arguments and results, and `langfuse.observation.*`
input and output, so Langfuse shows them in its panes. Leave it off when prompts
contain data that must not leave your systems.

Guardrail actions appear as `ragkit.guardrail` events on the span where they
happened.

## Cost

Add prices to compute cost:

```yaml
# ragkit.yaml
pricing:
  "google_genai:gemini-3.5-flash-lite": {input_per_million: 0.10, output_per_million: 0.40}
```

Keys are the model as written in configuration (`provider:model`). Prices are
yours to keep current; ragkit ships none.

Cost then appears:

- on `chat` and `invoke_agent` spans as `gen_ai.usage.cost`,
- in every chat response's `usage` (`input_tokens`, `output_tokens`, `cost_usd`),
- in eval reports (`cost_usd`),

and `budget.max_cost_usd` can stop expensive runs. `cost_usd` is `null` when any
model in the run has no price.

## Langfuse

1. Create a project in Langfuse and copy its keys.
2. Set `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` in `.env`.
3. Set `observability.exporter: langfuse`, and `observability.langfuse.host` if you
   are not on the EU cloud (for example `https://us.cloud.langfuse.com` or your
   self-hosted URL).

Conversations are grouped as Langfuse sessions by thread id.

## Phoenix

```bash
docker run -p 6006:6006 arizephoenix/phoenix
```

Then set `observability.exporter: phoenix`.
