# Architecture

How the code is organized and how data flows through it.

## High-level shape

contextx is a **library** (`contextx/`) with an optional **service** layer
(`contextx/service.py`, FastAPI). The library is a staged pipeline orchestrated
by `ContextEngine` (`contextx/pipeline.py`). The engine owns one instance of each
stage component and threads a single `ContextItem` list + `Trace` through them.

```
                          ┌──────────────────────────────────────────┐
   HTTP (optional)        │              ContextEngine               │
   service.py  ─────────▶ │  ingest() / run() / run_stream() / a*()  │
   (auth, SSE)            └───┬───────────────────────────────────┬──┘
                             │ ingest                             │ query
                    ┌────────▼─────────┐              ┌───────────▼───────────┐
                    │   VectorStore    │              │  Collect→Retrieve→...  │
                    │  (store.py)      │◀─────────────│  →Rerank→Rank→Filter→  │
                    │  backend + FTS5  │   search     │  Compress→Budget→Build │
                    │  + SQLite chunks │              │  →Validate→LLM         │
                    └────────┬─────────┘              └───────────┬───────────┘
                             │                                    │
                    ┌────────▼─────────┐              ┌───────────▼───────────┐
                    │  VectorBackend   │              │ Cache · Memory · Trace │
                    │ faiss/numpy/pg   │              │  (cross-cutting)       │
                    └──────────────────┘              └────────────────────────┘
```

## Module map

| Module | Responsibility |
|---|---|
| `config.py` | `Config` dataclass — every knob and threshold in one place |
| `types.py` | `ContextItem`, `Document`, `Request`, `Source`, trust sets |
| `pipeline.py` | `ContextEngine` — orchestrates all stages; `ingest/run/run_stream/arun/arun_stream` |
| `collect.py` | Stage 1 — gather ephemeral context |
| `store.py` | Persistent store: SQLite chunks + FTS5 lexical + pluggable vector index; ingest, search, delete/update, rebuild, model-version stamping, tenant/ACL |
| `backends.py` | `VectorBackend` interface + `FaissBackend`, `NumpyBackend`, `QdrantBackend`, `PgVectorBackend` |
| `chunking.py` | Structure-aware chunking (headings, atomic code blocks, overlap) |
| `retrieve.py` | Stage 2 — hybrid recall (semantic + lexical + RRF), ephemeral scoring, BM25 |
| `rerank.py` | Stage 2b — cross-encoder reranker (+ raw score for abstention) |
| `rank.py` | Stage 3 — weighted signal blend |
| `filter.py` | Stage 4 — dedup, stale removal, contradiction resolution |
| `memory.py` | Stage 5 — SQLite memory, scoped/bounded, lifecycle |
| `compress.py` | Stage 6 — extractive / abstractive compression |
| `budget.py` | Stage 7 — token counting + budget fitting |
| `build.py` | Stage 8 — prompt assembly, trust fencing, citations, cache markers |
| `validate.py` | Stage 9 — pre-flight checks + injection scan |
| `llm.py` | Stage 10 — pluggable LLM backends (Claude/OpenAI/Ollama/mock), retries, streaming, tool use |
| `observability.py` | Stage 11 — `Trace`, per-stage timing, cost estimation, structured logs |
| `cache.py` | Stage 12 — bounded LRU + TTL + semantic response cache |
| `security.py` | PII/financial redaction, append-only audit log |
| `auth.py` | API-key → tenant/principals (service auth) |
| `service.py` | FastAPI HTTP layer (ingest/query/stream/documents/health/stats) |
| `loaders.py` | Document loaders (txt/md/html/pdf) |
| `sync.py` | Incremental directory sync (content-hash diff → upsert/delete) |
| `tune.py` | Learned rank weights (nDCG search) + feedback store |
| `eval/` | Offline retrieval + faithfulness evaluation harness |
| `domains/` | Domain packs (currently `finance`) |

## Two entry phases

### Ingest (`engine.ingest(documents)`)
1. `store.add_documents` chunks each `Document` (`chunking.py`).
2. Embeds all new chunks in a batch (`embeddings.py`).
3. Persists: chunk text + metadata + tenant/ACL + the embedding blob into SQLite;
   the vector into the `VectorBackend`; the text into the FTS5 index.
4. Stamps the embedding-model name (so a later mismatch is caught, not silently
   wrong).
Also: `update(documents)` (delete-by-doc_id then re-add), `delete(doc_id)`
(tenant-scoped), both invalidate the response cache.

### Query (`engine.run(request)` / `run_stream` / `arun`)
`_prepare()` runs stages 1–9 and returns a `_Prepared` (prompt, validation
report, confidence, scope). Then:
- `run()` calls the LLM (through the semantic cache), writes memory, `_finalize()`
  records metrics/cost/audit, returns a `PipelineResult`.
- `run_stream()` returns a `StreamResult` whose `.stream` yields text chunks;
  memory is written when the stream is fully consumed. Sources and confidence are
  known up front.
- `arun()` / `arun_stream()` are the async wrappers — they offload the sync
  CPU/GPU-bound pipeline to a worker thread so an async host's event loop is never
  blocked (the correct async model for CPU-bound work).
- `arun_with_sources(request, fetchers)` awaits async source fetchers
  concurrently (`asyncio.gather`) — N remote fetches cost ~one round-trip — then
  runs the pipeline. Embedding is batched within a request (`embed_batch_size`);
  cross-request batching is a serving-layer concern, out of the library's scope.

## Request lifecycle

`run()` for one query:

```
Request
  → _prepare()                              # stages 1–9, in a worker thread under arun()
      1  collect ephemeral context + read memory (scope tenant:user)
      2  retrieve: vector ANN + FTS5 lexical → RRF, tenant/ACL filtered
      2b rerank recall set (cross-encoder) → confidence / abstention flag
      3  rank (weighted blend)   4 filter (dedup/stale/contradiction)
      6  compress oversized      7 budget to the window
      (redact PII if enabled)
      8  build prompt (fenced untrusted + numbered citations + cache marker)
      9  validate (ceiling, delimiters, injection scan)
      → _Prepared(prompt, report, confidence, scope)
  → 10 LLM (via semantic cache)             # blocked here if validation failed
  → 5  write memory (extract facts)
  → _finalize()                             # metrics, USD cost, audit, log
  → PipelineResult(answer, sources, low_confidence, trace)
```

`run_stream()` shares `_prepare()`, then streams stage 10 and writes memory when
the stream is drained. `run_with_tools()` replaces stage 10 with an agentic
tool loop.

## Storage & data model

State lives under `Config.index_dir` plus the memory DB. Nothing is global.

| Store | Location | Holds |
|---|---|---|
| Chunk store | `index_dir/chunks.db` (SQLite) | one row per chunk: `row` id, `chunk_id`, `doc_id`, `source`, `text`, `metadata`, `tenant_id`, `acl`, `embedding` blob, `timestamp` |
| Lexical index | `chunks_fts` (SQLite FTS5, same DB) | BM25 full-text over chunk text |
| Meta | `meta` table (same DB) | stamped embedding model + dim (mismatch guard) |
| Vector index | backend-specific (`index_dir/index.faiss`, `matrix.npy`, Qdrant, or Postgres) | one vector per chunk, keyed by `row` |
| Memory | `Config.memory_db_path` (SQLite) | per-`scope` facts: text, type, importance, ttl, `fact_key`/`value`, embedding |
| Sync manifest | `index_dir/sync_manifest.json` | file content-hashes for incremental `DirectorySync` |

The chunk store is the source of truth; the vector index is derived (a `rebuild()`
reconstructs it from the stored embeddings and compacts tombstones).

## Deployment

- **Embedded (library):** construct one `ContextEngine` and call it. State is on
  local disk; single process. Good for scripts, notebooks, one-box services.
- **Service:** `contextx.service:create_app` (FastAPI). One engine per worker;
  API-key auth maps callers to tenants; writes serialized with an async lock; the
  sync engine runs in a threadpool. Ships with a `Dockerfile`.
- **Scaling:** the CPU/GPU-bound work (embedding, reranking) scales with worker
  threads/processes; the vector index scales by swapping the `VectorBackend`
  (Qdrant or pgvector) off single-node FAISS. Chunk/memory SQLite is single-writer
  per process — move to Postgres/a shared store for multi-writer deployments.

## The vector-store seam

`VectorStore` owns everything *except* the raw ANN index: chunk text, metadata,
tenant/ACL, the FTS5 lexical index, tombstoning, and rebuild/compaction. The ANN
index — the part with the single-node RAM ceiling — lives behind the narrow
`VectorBackend` interface (`add / search / count / reset / save / load`). Swap
`Config.vector_backend` to change it:
- `faiss` — HNSW ANN, in-process (default).
- `numpy` — brute-force cosine fallback.
- `qdrant` — Qdrant (`:memory:`, local path, or server URL).
- `pgvector` — Postgres + pgvector (experimental; needs a DB; not CI-covered).

Contract: the store assigns each vector a dense integer `row` id (rebuild
renumbers 0..n-1), so FAISS/numpy can treat position == row; pgvector stores the
ids explicitly.

## Concurrency & persistence model

- **SQLite** (chunks + memory) uses WAL mode; write paths are guarded by an
  in-process `RLock`. Safe for concurrent *readers* and single-writer within a
  process.
- The **service** serializes index writes with an `asyncio.Lock` and offloads the
  sync engine to a threadpool — the correct model because the heavy work
  (embedding, reranking) is CPU/GPU-bound.
- The **response cache**, semantic-cache vectors, and memory are all bounded, so
  a long-running process does not leak.

## Failure & degradation

Every external dependency is optional with a fallback, chosen at construction and
reported in the trace footer (`embed_backend`, `index_backend`, `rerank_backend`,
`llm_backend`):

| Missing | Falls back to |
|---|---|
| `sentence-transformers` | deterministic hash embedder |
| `faiss` | numpy brute-force index |
| cross-encoder | identity reranker (keep bi-encoder order) |
| `tiktoken` | ~4-chars/token heuristic |
| LLM key/server | extractive mock |
| SQLite FTS5 | vector-only retrieval |
