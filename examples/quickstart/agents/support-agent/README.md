# support-agent

A tool-using support agent (ReAct pattern). It answers questions and calls
`lookup_order` when it needs an order's status.

| File | What to change |
|---|---|
| `agent.yaml` | Model, tools, memory, step budget |
| `prompts/system.md` | Tone and rules |
| `tools.py` | Replace `lookup_order` with calls to your real systems; add more tools and list them under `tools.local` |
| `evals/golden.jsonl` | Questions and expected behaviour, used by `ragkit eval` (coming in M3) |

Try it:

```bash
ragkit chat support-agent "Where is order 1042?"
```
