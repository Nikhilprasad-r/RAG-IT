# Evals

`ragkit eval` measures quality on fixed datasets, so changes to prompts, models,
chunking or reranking can be compared, and CI can fail when quality drops.

## Agent evals

Each agent has a dataset of cases, one JSON object per line
(`evals.dataset` in `agent.yaml`, usually `evals/golden.jsonl`):

```json
{"id": "reset-link", "input": "How long is a reset link valid?", "reference": "It expires after 30 minutes.", "expected_sources": ["account/password.md"]}
{"id": "chitchat", "input": "Hello!", "expected_tools": []}
{"id": "invoice", "input": "INVOICE #A-1043 ... 95.20 EUR", "reference": {"total": 95.2, "currency": "EUR"}}
```

```bash
ragkit eval agent rag-chat
ragkit eval agent rag-chat knowledge-agent --no-judge
ragkit eval agent rag-chat --dataset evals/regressions.jsonl --output report.json
```

Metrics are computed only for the cases that provide what they need:

| Metric | Case field | Meaning |
|---|---|---|
| `tool_recall` | `expected_tools` | Share of expected tools the agent called; an empty list means no tools should be called |
| `source_recall` | `expected_sources` | Share of expected sources (substrings of URIs) among retrieved passages |
| `field_accuracy` | `reference` as an object | Share of reference fields matched by the structured output |
| `correctness` | `reference` as text | LLM judge: does the reply state the reference's facts? |
| `groundedness` | (retrieval during the run) | LLM judge: are the reply's claims supported by the retrieved passages? |
| `error_rate` | | Share of cases that raised an error |
| `latency_p50_s`, `latency_p95_s` | | Time per case |
| `cost_usd` | | Total, when every model is priced |

The judge uses `evals.judge_model` (default: `default_model`) and returns a score
from 0 to 1 with a reason, saved per case in the report. `--no-judge` skips the
judged metrics.

Cases run in parallel (`evals.concurrency`, default 4), each on a fresh thread.
Approval requests are answered with `evals.on_approval` (default `reject`), so a
case that triggers a refund does not issue one, while `tool_recall` still sees the
attempt.

## Gates

```yaml
# agent.yaml
name: rag-chat
evals:
  dataset: evals/golden.jsonl
  gate:
    correctness: 0.8              # minimum
    source_recall: 0.75
    max_error_rate: 0             # max_ prefix: maximum
    max_latency_p95_s: 8
```

`ragkit eval agent` exits with status 1 when any gate fails or names a metric that
was not measured. `--no-gate` reports without failing.

## Search evals

Measure retrieval alone, without an LLM:

```json
{"query": "password reset link expiry", "expected_sources": ["account/password.md"]}
{"query": "too many requests 429", "expected_sources": ["api.md"]}
```

```bash
ragkit eval search docs evals/search.jsonl -k 5 --mode dense --mode sparse --mode hybrid --gate hit_rate=0.9
```

| Metric | Meaning |
|---|---|
| `hit_rate` | Share of queries with an expected source in the top k |
| `recall` | Share of expected sources found in the top k |
| `mrr` | Mean reciprocal rank of the first expected source |
| `latency_p50_s`, `latency_p95_s` | Time per query |

Each `--mode` produces its own report, which makes mode and reranker comparisons
quick.

## Reports

Reports are written as JSON to `.ragkit/evals/` (or `--output`), with every case's
reply, metrics, judge reasons, tools called, sources and latency.

## In CI

```yaml title=".github/workflows/evals.yml"
jobs:
  evals:
    runs-on: ubuntu-latest
    env:
      GOOGLE_API_KEY: ${{ secrets.GOOGLE_API_KEY }}
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
      - run: uv sync
      - working-directory: my-app
        run: |
          uv run ragkit ingest
          uv run ragkit eval search docs evals/search.jsonl --gate hit_rate=0.9
          uv run ragkit eval agent rag-chat
```
