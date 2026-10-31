# Search and reranking

Every search, whether from the CLI, the HTTP API or an agent, goes through the
project's `KnowledgeBase`:

1. **Retrieve candidates** with dense search (embeddings), sparse search (BM25
   keywords) or both, run in parallel.
2. **Fuse** the two rankings with Reciprocal Rank Fusion (RRF), which needs no
   score calibration between them.
3. **Rerank** the candidates with a cross-encoder or a hosted reranker, if one is
   configured.
4. **Filter** by tenant, ACL principals and metadata inside the store, before
   anything leaves the database.

```bash
ragkit search docs "rate limit for the REST API" -k 5
ragkit search docs "Retry-After" --mode sparse        # exact words only
ragkit search docs "how do I get my data out" --no-rerank
```

## Modes

| Mode | Good at | Weak at |
|---|---|---|
| `dense` | Paraphrases and meaning ("get my data out" finds "export") | Exact codes, names, rare terms |
| `sparse` | Exact words, error codes, product names | Synonyms and phrasing differences |
| `hybrid` (default) | Both | Costs one embedding call per query (cached) |

Compare them on your own data with
`ragkit eval search docs evals/search.jsonl --mode dense --mode sparse --mode hybrid`.

## Reranking

A reranker reads the query and each candidate together and re-scores them. It is
slower than retrieval but usually more precise, so search fetches `candidates`
(50 by default) and reranks them down to the final number.

```yaml
# ragkit.yaml
rerank:
  provider: cohere          # cohere | voyage | cross-encoder
  top_n: 8
```

| Provider | Needs | Default model |
|---|---|---|
| `cohere` | `COHERE_API_KEY` | `rerank-v3.5` |
| `voyage` | `VOYAGE_API_KEY` | `rerank-2.5` |
| `cross-encoder` | `uv sync --group local-models` | `BAAI/bge-reranker-v2-m3` (multilingual; about 2.3 GB, downloaded on first use) |

The project-level reranker applies to every search. Agents can override it in
`retrieval.rerank` or turn it off with `retrieval.use_rerank: false`. Register
your own reranker as a [plugin](../extending/plugins.md).

## Stores

| | SQLite | Postgres |
|---|---|---|
| Setting | `store.backend: sqlite` | `store.backend: postgres` and `database_url` |
| Vectors | NumPy cosine over the collection | pgvector HNSW index per collection |
| Keywords | SQLite FTS5 (BM25) | Postgres full-text search |
| Good for | Up to roughly 100,000 chunks, one machine | Large corpora, many servers sharing one index |

On Postgres, each collection gets a partial HNSW index for its embedding size.
Embeddings over 2,000 dimensions use `halfvec`, which pgvector can index up to
4,000.

## Filters

Searches over HTTP accept equality filters on chunk metadata:

```json
{"query": "install on linux", "top_k": 5, "filters": {"repo": "acme/handbook", "version": 2}}
```

## Caching

Repeated queries reuse their embeddings from an in-memory LRU cache
(`cache.query_embeddings` entries, 10,000 by default). Identical searches can
also reuse results for `cache.search_ttl_seconds`; this is off by default because
results can be stale for that long after an ingest. Cache keys include the tenant
and principals, so cached results are never shown to someone who could not read
them. Each server process has its own caches.

## From Python

```python
from ragkit.core.config import ProjectConfig
from ragkit.retrieval import KnowledgeBase

project = ProjectConfig.load()
async with KnowledgeBase(project) as kb:
    hits = await kb.search("docs", "rate limit", top_k=5)
    for hit in hits:
        print(hit.score, hit.uri, hit.chunk.metadata.get("heading_path"), hit.chunk.text[:80])
```
