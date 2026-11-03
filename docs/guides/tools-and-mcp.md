# Tools and MCP

Agents call **local tools** (Python functions in their `tools.py`) and tools from
**MCP servers** (Model Context Protocol), such as ticketing systems, file systems
or internal services.

## Declaring MCP servers

Servers are declared once in `ragkit.yaml` and shared by every agent:

```yaml
# ragkit.yaml
mcp_servers:
  files:                                # starts a local process (stdio)
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem", "./data"]
  tickets:                              # a remote server over HTTP
    transport: http
    url: https://mcp.example.com/mcp
    headers: {Authorization: "Bearer ${TICKETS_TOKEN}"}
```

| Transport | Settings |
|---|---|
| `stdio` (default) | `command`, `args`, `env`, `cwd` (defaults to the project root) |
| `http` | `url`, `headers`, `timeout` (streamable HTTP) |
| `sse` | `url`, `headers`, `timeout` |
| `websocket` | `url` |

Each server is started or connected to once per process, and its session stays
open, so tool calls do not pay a start-up cost. `ragkit chat` only starts the
servers its agent (and that agent's sub-agents) use.

## Giving tools to an agent

```yaml
# agent.yaml
name: helpdesk
tools:
  local: [lookup_order]
  mcp:
    - server: tickets
      allow: [create_ticket, get_ticket]   # omit allow to take every tool
    - server: files
```

An `allow` entry the server does not offer stops start-up with the list of tools
it does offer. Two tools with the same name in one agent are an error.

## Approval for risky tools

Any tool, local or MCP, can require a human decision before it runs:

```yaml
# agent.yaml
name: helpdesk
tools:
  mcp: [{server: tickets}]
hitl:
  require_approval: [create_ticket]
```

See [Human approval](human-approval.md).

## Tracing

Each call appears as an `execute_tool <name>` span. With
`observability.capture_content: true` the span records the arguments and the
result. See [Observability](observability.md).
