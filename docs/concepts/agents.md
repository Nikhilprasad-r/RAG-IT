# Agents

An agent is a folder in `agents/` with an `agent.yaml`, a system prompt and
optionally `tools.py` and an output schema. At start-up the runtime builds each
one into a LangGraph agent.

```
agents/support-agent/
├─ agent.yaml          # what the agent is and may do
├─ prompts/system.md   # its instructions
├─ tools.py            # Python functions it can call
└─ evals/golden.jsonl  # test cases
```

## Patterns

| `pattern` | How it works | Use for |
|---|---|---|
| `rag-chat` | Retrieves passages for each user message and answers in one model call | Questions about documents; cheapest and fastest |
| `react` | The model calls tools in a loop until it can answer. With `retrieval`, it gets a `search_knowledge` tool | Multi-step questions, actions, anything with tools |
| `supervisor` | A `react` agent whose tools are other agents (`ask_<name>`) | Routing work across specialists |

## What an agent can use

| In `agent.yaml` | Gives the agent |
|---|---|
| `retrieval` | Search over the project's collections |
| `tools.local` | Functions from its `tools.py` |
| `tools.mcp` | Tools from MCP servers declared in `ragkit.yaml` |
| `subagents` | Other agents as `ask_<name>` tools |
| `output_schema` | A Pydantic model it must return |
| `hitl` | Human approval before chosen tools run |
| `guardrails` | PII redaction, prompt-injection screening, deny rules |
| `budget` | A cap on model calls and on cost per request |

## Conversations and memory

Each conversation is a **thread** identified by a `thread_id`. Sending another
message with the same `thread_id` continues the conversation.

`memory.checkpointer` decides where threads are kept:

- `memory`: in the server process. Simple, but lost on restart and not shared
  between replicas.
- `postgres`: in `database_url`. Survives restarts and is shared by every API
  replica, which is required when running more than one process.

Over HTTP, threads are private to the caller: the same `thread_id` used by two
different users or tenants refers to two different conversations.

## A turn, step by step

1. Guardrails check the user message (deny rules, PII).
2. For `rag-chat`, passages are retrieved and added to the system prompt.
3. The model is called. If it requests tools, they run (approval-gated tools pause
   the turn here), and the model is called again with the results.
4. The loop stops when the model answers, when `budget.max_steps` model calls have
   been made, or when `budget.max_cost_usd` is reached.
5. Guardrails check the reply, and the result is returned with token and cost usage.
