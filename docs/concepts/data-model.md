# Data model

Data moves through four shapes on its way into the index.

| Type | Produced by | Holds |
|---|---|---|
| `RawDocument` | a connector | The source item's bytes, MIME type, URI, metadata, ACL and tenant |
| `ParsedContent` | a parser | Markdown-flavoured text, a title, page offsets for paged formats |
| `Document` | the pipeline | Parsed text plus identity: collection, source, URI, tenant, ACL, checksum |
| `Chunk` | the chunker | A slice of the text, its context header, heading path, page, embedding |

Searches return `ScoredChunk`s: a chunk with its score, the URI and title of its
document, and which retriever scored it.

## Identity of a document

A document's id is derived from its collection, tenant, source and URI. The
**source** is the connector's name, such as `files:data` or `github:owner/repo`.
That is how a re-run knows which stored documents came from which source, and
which ones disappeared.

The **checksum** is a hash of the raw bytes. When it matches the stored one, the
item is skipped without parsing or embedding.

## Chunks

Parsers turn every format into Markdown-like text, with headings marked by `#`.
The chunker:

1. splits the text at headings, tracking the heading path (`Guide > Install > Linux`);
2. packs paragraphs into chunks of up to `max_tokens` (estimated at four
   characters per token);
3. splits paragraphs that are too long at sentences, then at words;
4. repeats the last `overlap_tokens` of each chunk at the start of the next;
5. records the page for PDFs.

Each chunk gets a **context header** of the document title and heading path.
The header is embedded and keyword-indexed with the text, which helps short
chunks match questions about their section.

## Tenants and access control

Every document and chunk carries a **tenant** and an **ACL** (a list of
principals, such as group names).

- The tenant comes from the collection's `tenant` setting, or from the connector
  if it sets one.
- The ACL comes from the connector, or from the collection's `acl`.
- An empty ACL means everyone in the tenant can read the document.

Searches always filter by tenant and by the caller's principals inside the store,
so results a caller may not read never leave the database. See
[Auth and multi-tenancy](../guides/auth-and-tenancy.md).

## Metadata

Connectors attach metadata (for example `path`, `url`, `repo`, `space`) and parsers
add their own. Chunks inherit the document's metadata plus `heading_path` and
`page`. Searches can filter on metadata by equality:

```bash
curl -s localhost:8000/collections/docs/search -H 'content-type: application/json' \
  -d '{"query": "install", "filters": {"repo": "acme/handbook"}}'
```
