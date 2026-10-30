# supervisor

A multi-agent coordinator. Each agent listed under `subagents` becomes an
`ask_<name>` tool; the supervisor decides whom to ask, sends each a
self-contained task, and merges the answers.

```bash
ragkit add agent knowledge-agent
ragkit add agent sql-analyst
ragkit add agent supervisor
ragkit chat supervisor "Which plan has SSO, and how many Team-plan customers signed up last month?"
```

How it behaves:

- Each sub-agent keeps its own memory per conversation (thread `<thread>/<agent>`), so
  follow-up delegations have context.
- Only the supervisor's own reply is streamed; sub-agent output reaches it as tool results.
- Supervisors can delegate to other supervisors; cycles are rejected at start-up.
- Agents that need human approval (`hitl`) cannot be sub-agents yet; start-up fails with
  a clear error if one is listed.
