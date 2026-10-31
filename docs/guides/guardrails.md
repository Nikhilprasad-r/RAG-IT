# Guardrails

Guardrails are declared per agent under `guardrails`. Each action they take is
recorded on the trace (a `ragkit.guardrail` span event) and in the response's
`usage.guardrails`.

```yaml
# agent.yaml
name: helpdesk
guardrails:
  - pii
  - prompt_injection
  - type: deny
    patterns: ['\bmedical advice\b', '\blegal advice\b']
    message: I can't help with medical or legal questions.
```

## PII

Finds personal data in user messages, model replies and (optionally) tool
results, and replaces it before it reaches the model or the caller.

```yaml
# agent.yaml
name: helpdesk
guardrails:
  - type: pii
    kinds: [email, credit_card, phone, employee_id]
    patterns: {employee_id: "EMP-\\d{5}"}
    strategy: redact            # redact | mask | hash
    apply_to: [input, output, tool_results]
```

| Kind | Detects |
|---|---|
| `email`, `credit_card`, `ip`, `mac_address`, `url` | Built into LangChain's PII middleware |
| `phone` | 8 to 15 digits with phone-style separators; card numbers and dates are excluded |
| `iban` | International bank account numbers |
| your names under `patterns` | Your regular expressions |

| Strategy | `jo.doe@example.com` becomes |
|---|---|
| `redact` | `[REDACTED_EMAIL]` |
| `mask` | `jo.doe@****.com` |
| `hash` | `<email_hash:67f82335>` |

`pii` on its own applies the defaults: email, card, IP, phone and IBAN, redacted
in input and output.

## Prompt injection

Text an agent reads from tools, web pages and documents may try to instruct it
("ignore previous instructions", fake `</system>` tags, requests to reveal the
prompt or send data somewhere). The `prompt_injection` guardrail screens tool
results and retrieved passages for such text.

```yaml
# agent.yaml
name: helpdesk
guardrails:
  - type: prompt_injection
    action: annotate            # annotate | drop
    patterns: ['\bwire the money\b']
```

- `annotate` keeps the content but prefixes it with a note telling the model it
  is untrusted data, not instructions.
- `drop` removes it: a flagged tool result is withheld, a flagged passage is left
  out of the results.

Retrieved passages are screened one by one, so one bad document does not hide the
others. Detection is pattern-based: it catches common phrasings, not every
possible attack, so combine it with least-privilege tools and approval for risky
actions.

## Deny rules

Refuse requests matching case-insensitive regular expressions, without calling
the model:

```yaml
# agent.yaml
name: helpdesk
guardrails:
  - type: deny
    patterns: ['\bpassword\b.*\bfor\b.*\buser\b']
    message: I can't share account credentials.
```

Only fresh user messages are checked, not the agent's own tool loop.

## Cost and step budgets

`budget.max_steps` caps model calls per request. `budget.max_cost_usd` ends the
run politely once its cost reaches the limit; it needs a price for the model under
`pricing` in `ragkit.yaml` (see [Observability](observability.md#cost)).
