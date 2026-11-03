# Custom connectors

A connector is any class with a `name` and an async `fetch()` that yields
`RawDocument`s. No base class is required.

```python title="connectors/crm.py"
import httpx

from ragkit.core.types import RawDocument


class CrmArticles:
    """Help-centre articles from our CRM."""

    def __init__(self, region: str, token_env: str = "CRM_TOKEN") -> None:
        self.region = region
        self.token_env = token_env
        # Stable identity: ingestion tracks documents per source name.
        self.name = f"crm:{region}"

    async def fetch(self):
        import os

        headers = {"Authorization": f"Bearer {os.environ[self.token_env]}"}
        async with httpx.AsyncClient(headers=headers) as client:
            response = await client.get(f"https://crm.example.com/{self.region}/articles")
            response.raise_for_status()
            for article in response.json():
                yield RawDocument(
                    source=self.name,
                    uri=article["url"],
                    data=article["html"].encode(),
                    mime_type="text/html",
                    metadata={"article_id": article["id"], "product": article["product"]},
                    acl=article.get("visible_to", []),
                )
```

Use it by import path. The project folder is importable, and every key except
`type` is passed to the constructor:

```yaml
# ragkit.yaml
collections:
  help:
    sources:
      - type: connectors.crm:CrmArticles
        region: eu
```

## Rules a connector must follow

- **Yield everything, every time.** Ingestion deletes stored documents that a
  source no longer yields, so `fetch()` must list the full set, not only changes.
  Unchanged items are cheap: their hash matches and they are skipped.
- **Keep `name` stable.** It identifies the source. Changing it makes every item
  look new and the old ones stale.
- **Use stable URIs.** The URI identifies an item within the source and is shown
  as the citation.
- **Report incomplete listings.** If some items could not be fetched for a
  temporary reason, set `self.complete = False` before `fetch()` returns. The
  missing items are then kept instead of deleted. Raising an exception has the
  same effect for the whole source.
- **Set access and tenancy when you know them.** `acl` restricts who can read the
  item; `tenant_id` overrides the collection's tenant.

## Validated options

For options with validation and defaults, give the class a `from_options`
class method. Built-in connectors use a Pydantic model for this:

```python
from pathlib import Path
from typing import Any, Self

from pydantic import Field

from ragkit.ingest.connectors import ConnectorOptions


class CrmOptions(ConnectorOptions):
    region: str = Field(pattern=r"^(eu|us)$")
    token_env: str = "CRM_TOKEN"


class CrmArticles:
    def __init__(self, options: CrmOptions) -> None:
        self.options = options
        self.name = options.name or f"crm:{options.region}"

    @classmethod
    def from_options(cls, options: dict[str, Any], *, root: Path) -> Self:
        return cls(CrmOptions.model_validate(options))

    async def fetch(self): ...
```

Invalid options then fail with a message naming the bad key.

## Sharing a connector as a package

To use a short `type` name across projects, publish the connector in a package
with an entry point in the `ragkit.connector` group:

```toml title="pyproject.toml"
[project.entry-points."ragkit.connector"]
crm = "acme_ragkit.crm:CrmArticles"
```

```yaml
# ragkit.yaml
collections:
  help:
    sources:
      - type: crm
        region: eu
```

## Retries and rate limits

`ragkit.ingest.connectors.base.request_with_retry` sends an `httpx` request and
retries on 429 and 5xx responses, honouring `Retry-After`.
`ragkit.ingest.connectors.bounded_map` runs downloads concurrently with a limit.
