# rag-chat

Single-pass RAG: for every user message the agent searches the configured
collections once, puts the best passages in its prompt, and answers with
numbered citations. It makes one model call per message, so it is the
cheapest and fastest way to answer questions about your documents.

Use `knowledge-agent` instead when questions need several searches (comparisons,
multi-step questions).

| File | What to change |
|---|---|
| `agent.yaml` | `retrieval.collections` must name collections in `ragkit.yaml`; tune `top_k` and reranking |
| `prompts/system.md` | Answer style and citation rules |
| `evals/golden.jsonl` | Real questions and expected answers for `ragkit eval` (coming in M3) |

```bash
ragkit ingest docs
ragkit chat rag-chat "How do I reset my password?"
```
