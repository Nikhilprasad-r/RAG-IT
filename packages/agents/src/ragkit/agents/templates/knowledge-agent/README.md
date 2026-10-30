# knowledge-agent

Agentic retrieval: the model gets a `search_knowledge` tool and decides what to
search for, how many times, and when it has enough to answer. Use it for
comparisons and multi-step questions. It costs more model calls than
`rag-chat`; `budget.max_steps` caps them.

You can add your own tools in a `tools.py` and list them under `tools.local`
next to retrieval.

```bash
ragkit ingest docs
ragkit chat knowledge-agent "Compare the Pro and Team plans' storage limits"
```
