# Auth and multi-tenancy

The HTTP API is open by default (`auth.mode: none`), which suits local use. For
anything shared, use API keys or JWTs.

## API keys

```yaml
# ragkit.yaml
auth:
  mode: api_key
  rate_limit_per_minute: 120
  api_keys:
    - {key_env: KEY_ACME_APP, name: acme-app, tenant: acme, scopes: [chat, search]}
    - {key_env: KEY_ACME_HR, name: acme-hr, tenant: acme, principals: [hr], scopes: [search]}
    - {key_env: KEY_OPS, name: ops, tenant: acme, scopes: [chat, search, ingest]}
```

Each key's value is read from the environment variable named by `key_env`;
start-up fails if one is missing. Callers send it as `Authorization: Bearer <key>`
or `X-API-Key: <key>`.

## JWT

For tokens from your identity provider:

```yaml
# ragkit.yaml
auth:
  mode: jwt
  jwt:
    jwks_url: https://login.example.com/.well-known/jwks.json
    algorithms: [RS256]
    issuer: https://login.example.com/
    audience: ragkit
    tenant_claim: org
    principals_claim: groups
    scopes_claim: scope
```

Use `secret_env` with `algorithms: [HS256]` for tokens signed with a shared secret.
Tokens must have `exp`; the issuer and audience are checked when set. Scopes may
be a space-separated string or a list; tokens without scopes get
`default_scopes` (`chat` and `search`).

## Scopes

| Scope | Allows |
|---|---|
| `chat` | `/agents/*` |
| `search` | `/collections` and `/collections/{name}/search` |
| `ingest` | `/collections/{name}/ingest` and `/jobs` |
| `admin` | Every scope, for every tenant's collections and jobs (searches run under the collection's tenant) |

Missing credentials return 401, a missing scope 403, and too many requests 429
with `Retry-After`. `/health` and `/ready` need no credentials. Rate limits are per
caller and per server process.

## Tenants

Each collection belongs to a tenant (`default` unless set). Documents ingested
into it carry that tenant, and searches only return documents of the caller's
tenant.

```yaml
# ragkit.yaml
collections:
  acme-docs:
    tenant: acme
    sources: [{type: files, path: tenants/acme}]
  globex-docs:
    tenant: globex
    sources: [{type: files, path: tenants/globex}]
  public:
    tenant: platform
    shared: true                # every tenant can search it
    sources: [{type: files, path: public}]
```

- `GET /collections` lists only the caller's collections and shared ones.
- Searching another tenant's collection returns 404.
- Agents search **as the caller**: the same agent returns acme passages to acme
  users and globex passages to globex users.
- Conversation threads are private to each tenant and user: the same `thread_id`
  from two callers refers to two conversations, and one caller cannot resume
  another's approval.

## Document access within a tenant

Restrict documents to groups with an ACL, on a collection or per document from a
connector:

```yaml
# ragkit.yaml
collections:
  hr-policies:
    tenant: acme
    acl: [hr, leadership]
    sources: [{type: files, path: hr}]
```

Only callers whose principals include `hr` or `leadership` see these documents.
Filtering happens inside the store, for searches and for agents alike.

## Identity in Python

The server sets the caller's identity for each request. In your own code, set it
with `acting_as`:

```python
from ragkit.core.identity import Identity, acting_as

with acting_as(Identity(tenant_id="acme", user_id="u1", principals=("hr",))):
    hits = await kb.search("hr-policies", "parental leave")
```
