# Human approval

Tools listed under `hitl.require_approval` pause the agent before they run. A
person then approves the call, edits its arguments, or rejects it.

```yaml
# agent.yaml
name: support-agent
tools:
  local: [lookup_order, issue_refund]
hitl:
  require_approval: [issue_refund]
  decisions: [approve, edit, reject]      # what reviewers may do
```

The `support-agent` template is set up this way.

## In the terminal

`ragkit chat` asks for each decision:

```text
$ ragkit chat support-agent "Order 1042 arrived late, refund 40 EUR"
Approval needed: issue_refund
{"order_id": "1042", "amount": 40, "reason": "late delivery"}
[a]pprove / [e]dit / [r]eject: e
New arguments as JSON: {"order_id": "1042", "amount": 25, "reason": "late delivery"}
I've refunded 25.00 EUR for order 1042.
```

## Over HTTP

A turn that needs approval returns `status: "waiting_for_approval"` and the
pending calls:

```json
{
  "reply": "",
  "thread_id": "t-123",
  "status": "waiting_for_approval",
  "pending": [
    {"tool": "issue_refund", "args": {"order_id": "1042", "amount": 40, "reason": "late"},
     "description": "Tool execution requires approval ...",
     "allowed_decisions": ["approve", "edit", "reject"]}
  ]
}
```

Resume with one decision per pending call, in the same order:

```bash
curl -s localhost:8000/agents/support-agent/resume -H 'content-type: application/json' -d '{
  "thread_id": "t-123",
  "decisions": [{"type": "edit", "args": {"order_id": "1042", "amount": 25, "reason": "late"}}]
}'
```

| Decision | Effect |
|---|---|
| `{"type": "approve"}` | The call runs as proposed |
| `{"type": "edit", "args": {...}}` | The call runs with your arguments |
| `{"type": "reject", "message": "..."}` | The call does not run; the model is told why |

Streaming works the same way: `POST /agents/{name}/stream` ends with an
`approval_required` event instead of `end`, and `POST /agents/{name}/resume/stream`
continues it.

## Rules

- While a thread is waiting, new messages to it are refused with 409. Resume it
  first.
- Decisions must match the pending calls in number and be allowed by `decisions`;
  otherwise the request is refused with 409 and nothing runs.
- Over HTTP, only the caller who started the thread can resume it.
- Waiting threads survive restarts and work across replicas when the agent uses
  `memory.checkpointer: postgres`.
- During evals, approval requests are rejected by default
  (`evals.on_approval: reject`), so evals cause no side effects.
- Agents that need approval cannot be sub-agents of a supervisor yet.
