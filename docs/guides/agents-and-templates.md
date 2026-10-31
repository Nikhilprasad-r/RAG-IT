# Agents and templates

Agents are created by copying a **template** into the project. The copy is plain
files, so you change prompts, tools and settings by editing them.

```bash
ragkit templates                                   # list built-in templates
ragkit add agent rag-chat                          # copy one
ragkit add agent support-agent --name helpdesk     # copy under another name
ragkit agents                                      # agents in this project
```

## Built-in templates

| Template | Pattern | Use |
|---|---|---|
| `rag-chat` | single-pass RAG | Answers from your documents with numbered citations |
| `knowledge-agent` | agentic retrieval | Searches as often as needed; for multi-step questions |
| `support-agent` | tools + approval | Looks up orders; refunds wait for human approval |
| `sql-analyst` | tools | Answers data questions with read-only SQL |
| `extractor` | structured output | Turns text into a validated object (an invoice by default) |
| `supervisor` | multi-agent | Delegates to other agents and combines their answers |

Details for each, including its `agent.yaml`, are on the
[Templates](../reference/templates.md) page.

## Templates from GitHub or a folder

```bash
ragkit add agent ./shared/agents/triage
ragkit add agent gh:acme/agent-templates/triage@v2
ragkit add agent https://github.com/acme/agent-templates/tree/main/triage
```

From GitHub, only the template's folder is extracted from the repository archive.
Set `GITHUB_TOKEN` for private repositories.

!!! warning "Review third-party code"
    A template can contain Python (`tools.py`, output schemas) that runs with your
    permissions. `ragkit add agent` lists those files for you to review whenever the
    template does not come from ragkit itself.

## Anatomy of an agent

```yaml
# agent.yaml
name: helpdesk
description: Answers customer questions and looks up orders.
pattern: react
model: google_genai:gemini-3.5-flash-lite      # optional; defaults to default_model
fallback_models: [openai:gpt-5.5]              # tried if the main model fails
system_prompt: prompts/system.md

retrieval:
  collections: [docs]
  top_k: 5

tools:
  local: [lookup_order]                        # functions in tools.py

memory:
  checkpointer: memory                         # postgres for several server processes

budget:
  max_steps: 8                                 # model calls per request
```

Every key is described in the [agent.yaml reference](../reference/agent-manifest.md).
Agents are checked when the runtime starts: a missing tool, an unknown collection
or MCP server, or a typo in a key stops start-up with a message naming it.

## Tools in Python

Any function in the agent's `tools.py` listed under `tools.local` becomes a tool.
The model sees its name, its parameters (from type hints) and its docstring, so
write the docstring for the model:

```python title="agents/helpdesk/tools.py"
def lookup_order(order_id: str) -> str:
    """Look up an order by id and return its status and delivery date."""
    order = orders_api.get(order_id)  # your code
    return f"Order {order_id}: {order.status}, delivery {order.eta:%d %b}."
```

Async functions and LangChain `BaseTool` objects work too.

## Writing your own template

A template is just an agent folder. To share one, put it in a repository (one
folder per template) and install it with `ragkit add agent gh:...`. Keep in it:

- `agent.yaml` with a meaningful `name` and `description`,
- `prompts/system.md`,
- `tools.py` and any schema files it needs,
- `evals/golden.jsonl` with example cases,
- a `README.md` explaining setup (environment variables, collections it expects).

## Talking to agents

```bash
ragkit chat helpdesk "Where is order 1042?"
ragkit chat helpdesk "And when will it arrive?" --thread <thread id from the first reply>
```

Over HTTP: `POST /agents/{name}/chat` for a full reply, `POST /agents/{name}/stream`
for server-sent events. Send the returned `thread_id` to continue a conversation.
See the [HTTP API reference](../reference/http-api.md).
