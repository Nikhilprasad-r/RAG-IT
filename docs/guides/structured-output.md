# Structured output

An agent with `output_schema` returns an object validated against a Pydantic
model instead of free text. Use it for extraction, classification and any reply
that code will consume.

```python title="agents/extractor/schema.py"
from pydantic import BaseModel, Field


class Invoice(BaseModel):
    vendor: str = Field(description="Company that issued the invoice.")
    invoice_number: str | None = None
    invoice_date: str | None = Field(default=None, description="ISO 8601 date, e.g. 2026-10-30.")
    currency: str | None = Field(default=None, description="ISO 4217 code, e.g. EUR.")
    total: float = Field(description="Amount due, no currency sign.")
```

```yaml
# agent.yaml
name: extractor
system_prompt: prompts/system.md
output_schema: schema.py:Invoice
```

The model reads the field descriptions, so describe each field's meaning and
format, and make fields optional when documents may not contain them. The
`extractor` template is a complete example.

## Getting the result

=== "CLI"

    ```bash
    ragkit chat extractor "INVOICE #A-1043 from Acme GmbH ... Total due: 95.20 EUR"
    ```

    The validated object is printed as JSON.

=== "HTTP"

    `POST /agents/extractor/chat` returns the object in `structured`:

    ```json
    {"reply": "{...}", "status": "completed",
     "structured": {"vendor": "Acme GmbH", "invoice_number": "A-1043", "currency": "EUR", "total": 95.2}}
    ```

=== "Python"

    ```python
    result = await runtime.get("extractor").ainvoke(text)
    invoice = result.structured  # a dict
    ```

## Evaluating extraction

Give eval cases a JSON object as `reference`. The `field_accuracy` metric is the
share of reference fields the output matches (numbers within a tolerance, text
case-insensitively):

```json
{"input": "INVOICE #A-1043 from Acme GmbH ... 95.20 EUR", "reference": {"vendor": "Acme GmbH", "total": 95.2, "currency": "EUR"}}
```
