# extractor

Turns unstructured text into a validated object. The shape is a Pydantic model
in `schema.py` (an `Invoice` by default), named by `output_schema` in
`agent.yaml`. The result is validated against the model before it is returned.

```bash
ragkit chat extractor "INVOICE #A-1043 from Acme GmbH ... Total due: 95.20 EUR"
```

Over HTTP, `POST /agents/extractor/chat` returns the object in `structured`.

To extract something else, replace the models in `schema.py`, point
`output_schema` at your top-level model, and describe each field: the
descriptions are what the model reads.
