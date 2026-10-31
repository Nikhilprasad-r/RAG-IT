# Ingesting data

A **collection** is a named set of documents with one or more **sources**. Ingestion
fetches each source, parses and chunks the items, embeds the chunks and stores them.

```yaml
# ragkit.yaml
collections:
  docs:
    description: Product documentation and help centre
    sources:
      - type: files
        path: data
        include: ["**/*.md", "**/*.pdf"]
      - type: web
        urls: [https://help.example.com/]
        max_pages: 300
    chunking:
      max_tokens: 400
      overlap_tokens: 40
```

```bash
ragkit ingest                 # every collection
ragkit ingest docs            # one collection
ragkit ingest docs --full     # re-process everything, not just changes
ragkit ingest --queue         # hand the work to `ragkit worker` instead
```

## Connectors

All options are listed in the [ragkit.yaml reference](../reference/project-config.md#connector-options).

=== "files"

    Local folders or mounted shares. Hidden files are skipped unless
    `include_hidden: true`.

    ```yaml
    # ragkit.yaml
    collections:
      docs:
        sources:
          - type: files
            path: data
            include: ["**/*.md", "**/*.pdf", "**/*.docx"]
            exclude: ["drafts/**"]
    ```

=== "web"

    A breadth-first crawl from `urls`, staying under each start URL's host and
    path, following links up to `max_depth` hops and honouring robots.txt.

    ```yaml
    # ragkit.yaml
    collections:
      docs:
        sources:
          - type: web
            urls: [https://docs.example.com/guide/]
            max_depth: 3
            max_pages: 500
            exclude: ["*/changelog/*"]
    ```

=== "github"

    Files from a repository through the GitHub API. Public repositories need no
    token; set `GITHUB_TOKEN` for private ones and higher rate limits.

    ```yaml
    # ragkit.yaml
    collections:
      docs:
        sources:
          - type: github
            repo: acme/handbook
            branch: main
            include: ["docs/**/*.md"]
    ```

=== "confluence"

    Pages from Confluence Cloud spaces (v2 API). Set `CONFLUENCE_EMAIL` and
    `CONFLUENCE_API_TOKEN`.

    ```yaml
    # ragkit.yaml
    collections:
      wiki:
        sources:
          - type: confluence
            base_url: https://acme.atlassian.net/wiki
            spaces: [ENG, SUPPORT]
    ```

=== "notion"

    Pages shared with a Notion integration, converted to Markdown. Set
    `NOTION_TOKEN`. Give `database_ids` or `page_ids`, or neither to ingest every
    shared page.

    ```yaml
    # ragkit.yaml
    collections:
      notes:
        sources:
          - type: notion
            database_ids: [1f2e3d4c5b6a]
    ```

Need another system? Write a [custom connector](custom-connectors.md).

## Formats

| Format | Parsed as |
|---|---|
| Markdown, text, reStructuredText, code | Text (code is tagged as code) |
| HTML | Main content as Markdown; navigation, headers, footers and scripts dropped |
| PDF | Text per page, with page numbers on chunks |
| Word (`.docx`) | Headings, paragraphs and tables |
| CSV | One block per row, as `column: value` lines |
| JSON | Pretty-printed |
| Anything else | Read as text if it decodes as UTF-8, otherwise skipped and reported |

For complex layouts, set `parser: docling` on the collection. Docling handles
multi-column PDFs, tables, PowerPoint, Excel and images (with OCR) using local
models. It needs `uv sync --group local-models`, loads its models in about ten
seconds per process and converts about 1.5 seconds per PDF page on a laptop.

```yaml
# ragkit.yaml
collections:
  reports:
    parser: docling
    sources:
      - type: files
        path: reports
```

## Incremental syncs

Every run compares each item's content hash with the stored one:

| Item | What happens |
|---|---|
| New | Parsed, chunked, embedded, stored (*added*) |
| Changed | Re-processed and its old chunks replaced (*updated*) |
| Unchanged | Skipped without parsing or embedding (*unchanged*) |
| Gone from the source | Deleted (*deleted*), only if the source listed completely |
| Unsupported format or no text | Reported as *skipped* |
| Parse or embedding error | Reported as *failed*; the run continues |

If a source fails part-way (a network error, a website page that times out), its
missing items are **kept**, so an outage never empties a collection. Embedding
calls are retried with back-off before an item is marked failed.

`ragkit ingest` prints a table of these counts and exits with status 1 when any
item failed.

## Running ingestion in the background

In production, ingestion runs in workers rather than in a terminal:

```bash
ragkit ingest --queue          # or POST /collections/{name}/ingest
ragkit worker                  # runs queued jobs until stopped
ragkit jobs                    # recent jobs and their results
```

Set `sync_every` on a collection to re-sync it on a schedule. See
[Operations](../deploy/operations.md).

## Changing the embedding model

Vectors from different models cannot be compared. If `embeddings.model` changes,
`ragkit ingest` refuses to mix them and asks for `--full`, which drops the
collection and rebuilds it.
