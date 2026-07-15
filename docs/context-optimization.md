# How context optimization works

contextx assembles the prompt an LLM sees for a request: select, order, compress,
and frame exactly the right context.

## Constraints

An LLM answers only from its context window, which is fixed in size. Assembling
context must respect four limits:

- **Size** — corpora exceed any context window.
- **Signal** — irrelevant text degrades answers.
- **Cost & latency** — priced and timed per token.
- **Trust** — raw retrieved text carries injection, PII, staleness, and
  un-attributable claims.

## Ingest / query split

```
INGEST (per document, amortized)
    document → chunk → embed → persistent ANN index + SQLite sidecar

QUERY (per request)
    request → embed the query → search the index → … → LLM
```

Query time embeds only the query and searches a pre-built index; cost is
independent of corpus size. Ingest persists and survives restarts.
See `contextx/store.py` and `contextx/pipeline.py` (`ingest()` / `run()`).

## The unit of flow: `ContextItem`

Every item in the pipeline is a `ContextItem` (`contextx/types.py`). Fields are
filled stage by stage:

| Field | Filled by | Meaning |
|---|---|---|
| `text`, `source`, `timestamp`, `metadata` | collect / ingest | content and provenance |
| `embedding`, `similarity` | retrieve | vector, cosine to query |
| `rerank_score`, `raw_rerank_score` | rerank | cross-encoder relevance (normalized / raw) |
| `rrf_score` | retrieve | hybrid fusion score |
| `score`, `importance` | rank | blended rank score, pinned weight |
| `tokens`, `included` | budget | token count, kept flag |
| `trusted` | source | false for externally-authored content |

## Stages

Numbers match the trace output. Memory (5) and Cache (12) are cross-cutting.

**1 — Collect** (`collect.py`). Gather per-request ephemeral context: user
message, conversation, tool/API outputs. Durable knowledge comes from the index
at retrieval.

**2 — Retrieve** (`retrieve.py`, `store.py`). Two channels:
- Semantic: ANN search over the HNSW index (`store.search`).
- Lexical: BM25 over a SQLite FTS5 index (`store.lexical_search`) — matches exact
  tokens (IDs, tickers, error codes).

Fused with Reciprocal Rank Fusion: `score(doc) = Σ 1/(k + rank_in_channel)`.
Ephemeral items are embedded inline. Tenant/ACL filtering is applied after the
ANN search, so the fetch expands (×4 per round) until `k` authorized hits are
found or the index is exhausted.

**2b — Rerank** (`rerank.py`). A cross-encoder scores each (query, document) pair
jointly over the recall set (`recall_k` → `rerank_k`). Produces the abstention
signal: a top raw score below `abstain_below` flags `low_confidence`.

**3 — Rank** (`rank.py`). Blend the signals (weights in `Config`):

```
score = w_rerank·rerank + w_similarity·cosine + w_bm25·keyword
      + w_recency·recency + w_importance·importance
      + w_conversation·conv + w_preference·pref
```

**4 — Filter** (`filter.py`). Drop near-duplicates (cosine ≥ `dup_threshold`) and
stale items (`expires_at`); resolve contradictions on `fact_key`/`fact_value`.

**5 — Memory** (`memory.py`). SQLite store (WAL, thread-safe, bounded), scoped
per `tenant:user`. Read at collect, written after the answer. Eviction by
importance × recency; pinned facts survive.

**6 — Compress** (`compress.py`). Shrink items over `max_item_tokens` —
extractive (keep query-relevant sentences) or abstractive (LLM rewrite).

**7 — Budget** (`budget.py`). Count tokens, reserve output, apply a safety
margin, include highest-ranked items until the budget is spent.

**8 — Build** (`build.py`). Assemble the prompt:
- Untrusted content fenced in `<untrusted_context>` with a data-not-instructions
  directive.
- Retrieved knowledge numbered `[1]…[n]`; the model cites; sources returned.
- System prompt marked cacheable for prompt-caching.

**9 — Validate** (`validate.py`). Token ceiling, balanced delimiters, duplicate
lines, injection scan over untrusted content. A hard failure blocks the call.

**10 — LLM** (`llm.py`). Call the model. Backends: Claude, OpenAI (and
compatible gateways), Ollama, mock. Retries with backoff + timeout; streaming via
`stream()`.

**11 — Observability** (`observability.py`). A `Trace` records per-stage timing,
counts, token usage, USD cost, request id, and a JSON log line.

**12 — Cache** (`cache.py`). Bounded LRU + TTL, plus a semantic response cache
(match by embedding cosine, threshold `semantic_cache_threshold`). Invalidated on
ingest/update/delete.

## Cross-cutting

- **Multi-tenancy** — tenant isolation + per-document ACLs in both recall
  channels; per-user memory; tenant-scoped delete.
- **Degradation** — each backend has a fallback (hash embeddings, numpy index,
  identity reranker, mock LLM, heuristic tokenizer); the trace footer shows which
  ran.

## Stage order

Cheap recall (2) → cross-encoder on the recall set (2b) → blends and filters
(3, 4) → compression and budgeting (6, 7) → assembly and one call (8–10). Each
stage shrinks the item set, so the costly stages run on the fewest items.
