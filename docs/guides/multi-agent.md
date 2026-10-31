# Multi-agent supervisors

A supervisor delegates parts of a request to other agents and combines their
answers. Each agent listed under `subagents` becomes a tool named `ask_<name>`
(hyphens become underscores).

```yaml
# agent.yaml
name: supervisor
pattern: supervisor
system_prompt: prompts/system.md
subagents: [knowledge-agent, sql-analyst]
budget:
  max_steps: 8
```

```bash
ragkit add agent knowledge-agent
ragkit add agent sql-analyst
ragkit add agent supervisor
ragkit chat supervisor "Which plan includes SSO, and how many customers are on it?"
```

## How delegation works

- The supervisor sends each sub-agent a complete task in plain words. Sub-agents
  cannot see the supervisor's conversation, so the prompt tells the supervisor to
  include everything they need.
- Each sub-agent runs on its own thread, derived from the supervisor's
  (`<thread>/<agent>`), so it remembers earlier delegations in the same
  conversation.
- The sub-agent's reply (or its structured output, as JSON) comes back as the
  tool result.
- Only the supervisor's own reply is streamed to the caller.
- Tokens and cost of sub-agents are added to the supervisor's run, so budgets and
  usage cover the whole tree.
- In traces, each sub-agent's `invoke_agent` span sits under the
  `execute_tool ask_<name>` span that called it.

## Rules

- Sub-agents must be agents in the same project.
- Supervisors can delegate to other supervisors; cycles stop start-up with the
  cycle spelled out.
- Agents that need human approval cannot be sub-agents yet.
- `ragkit chat supervisor` loads the supervisor and its sub-agents only.

`subagents` also works on a `react` agent, which then mixes delegation with its
own tools and retrieval.
