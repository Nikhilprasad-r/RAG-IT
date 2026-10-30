# supervisor

A multi-agent coordinator. Each agent listed under `subagents` becomes an
`ask_<name>` tool; the supervisor decides whom to ask, sends each a
self-contained task, and merges the answers.

```bash
ragkit add agent knowledge-agent
ragkit add agent support-agent
ragkit add agent supervisor
ragkit chat supervisor "What does the Pro plan include, and where is my order 1042?"
```

How it behaves:

- Each sub-agent keeps its own memory per conversation (thread `<thread>/<agent>`), so
  follow-up delegations have context.
- Only the supervisor's own reply is streamed; sub-agent output reaches it as tool results.
- Supervisors can delegate to other supervisors; cycles are rejected at start-up.
- Sub-agents that need human approval are not supported yet.
