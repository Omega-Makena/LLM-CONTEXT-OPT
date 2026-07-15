# Configuration reference

Every knob lives in the `Config` dataclass (`contextx/config.py`). Construct one
and thread it through the engine:

```python
from contextx import ContextEngine, Config
engine = ContextEngine(config=Config(rerank_k=20, redact_pii=True))
```

Defaults below are the shipped values.

## Models

| Field | Default | Meaning |
|---|---|---|
| `embed_model` | `all-MiniLM-L6-v2` | sentence-transformers embedding model. **Changing it invalidates an existing index** (dimension/space mismatch → `ModelMismatchError`). |
| `rerank_model` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | cross-encoder for stage 2b. |
| `llm_model` | `claude-sonnet-5` | model id used for the Anthropic backend and cost estimation. |

## Ingest / chunking

| Field | Default | Meaning |
|---|---|---|
| `chunk_target_tokens` | `320` | target size of each chunk. |
| `chunk_overlap_tokens` | `48` | overlap carried between adjacent chunks. |
| `embed_batch_size` | `64` | batch size passed to the embedder (throughput knob for large ingests). |

## Persistent index

| Field | Default | Meaning |
|---|---|---|
| `index_dir` | `.contextx_index` | directory for the SQLite chunk store + vector index. |
| `vector_backend` | `auto` | `auto` (faiss if installed else numpy), `faiss`, `numpy`, or `pgvector` (experimental). |
| `pg_dsn` | `postgresql://localhost/contextx` | Postgres DSN for the pgvector backend. |
| `pg_table` | `contextx_vectors` | table name for pgvector. |
| `hnsw_M` | `32` | HNSW graph degree — higher = better recall, more RAM. |
| `hnsw_ef_construction` | `200` | HNSW build-time quality. |
| `hnsw_ef_search` | `64` | HNSW query-time recall/latency tradeoff. |

## Retrieval

| Field | Default | Meaning |
|---|---|---|
| `recall_k` | `40` | candidates pulled from the index (recall stage). |
| `rerank_k` | `12` | kept after the cross-encoder rerank (precision stage). |
| `ephemeral_k` | `8` | top inline (conversation/tool/memory) items kept. |
| `min_similarity` | `0.15` | drop ephemeral items below this cosine. |
| `enable_hybrid` | `True` | fuse vector + BM25 lexical recall via RRF. |
| `rrf_k` | `60` | reciprocal-rank-fusion constant. |
| `abstain_below` | `-3.0` | if the top **raw** cross-encoder score is below this, flag `low_confidence`. |

## Ranking weights

Linear blend; the sum need not be 1 (nDCG ranking is scale-invariant). Tune with
`contextx.tune.tune_weights` — see [evaluation.md](evaluation.md).

| Field | Default | Signal |
|---|---|---|
| `w_rerank` | `0.55` | cross-encoder joint relevance (dominant). |
| `w_similarity` | `0.15` | bi-encoder cosine. |
| `w_bm25` | `0.10` | lexical/keyword match. |
| `w_recency` | `0.08` | exponential recency. |
| `w_importance` | `0.07` | pinned/critical/source-prior. |
| `w_conversation` | `0.03` | discussed this turn. |
| `w_preference` | `0.02` | matches user preferences. |
| `recency_half_life_s` | `604800` (1 week) | recency decay half-life. |

## Filtering

| Field | Default | Meaning |
|---|---|---|
| `dup_threshold` | `0.90` | embedding cosine above which two items are near-duplicates. |

## Compression

| Field | Default | Meaning |
|---|---|---|
| `max_item_tokens` | `220` | items larger than this are compressed. |
| `compress_target_ratio` | `0.5` | extractive target size (fraction kept). |

## Budget

| Field | Default | Meaning |
|---|---|---|
| `max_context_tokens` | `32000` | the model's context window. |
| `reserve_output_tokens` | `4000` | tokens reserved for the response. |
| `budget_safety_margin` | `0.10` | headroom for tokenizer drift / block delimiters. |

## Memory

| Field | Default | Meaning |
|---|---|---|
| `memory_db_path` | `.contextx_memory.db` | SQLite file for memory. |
| `memory_max_records` | `5000` | bound; lowest importance×recency evicted (pinned survive). |
| `memory_read_k` | `5` | memories retrieved per request. |

## Cache

| Field | Default | Meaning |
|---|---|---|
| `cache_max_entries` | `10000` | LRU bound (exact + semantic caches). |
| `cache_ttl_s` | `3600` | per-entry time-to-live. |
| `semantic_cache_threshold` | `0.97` | cosine above which a new request matches a cached response. |

## LLM

`llm_provider="auto"` resolves **Claude** (`ANTHROPIC_API_KEY`) → **OpenAI**
(`OPENAI_API_KEY`) → **Ollama** (local server reachable) → **mock**.

| Field | Default | Meaning |
|---|---|---|
| `llm_provider` | `auto` | `auto` / `anthropic` / `openai` / `ollama` / `mock`. |
| `ollama_host` | `http://localhost:11434` | Ollama server. |
| `ollama_model` | `llama3.1` | falls back to the first pulled model if absent. |
| `openai_base_url` | `https://api.openai.com/v1` | OpenAI-compatible base URL (Azure/OpenRouter/vLLM/LM Studio). |
| `openai_model` | `gpt-4o-mini` | OpenAI model id. |
| `llm_max_tokens` | `1024` | max output tokens. |
| `llm_timeout_s` | `60` | request timeout. |
| `llm_max_retries` | `4` | retry attempts on transient errors. |
| `llm_backoff_base_s` | `0.5` | exponential backoff base. |
| `enable_prompt_caching` | `True` | mark the system prompt cacheable (Anthropic). |

## Security & privacy

| Field | Default | Meaning |
|---|---|---|
| `injection_scan` | `True` | scan untrusted content for injection patterns (stage 9). |
| `redact_pii` | `False` | scrub PII from retrieved context before it reaches the model. |
| `audit_log_path` | `None` | JSONL provenance trail (off when None). |

## Observability

| Field | Default | Meaning |
|---|---|---|
| `log_requests` | `False` | emit a structured JSON log line per request. |

## Domain presets

`contextx.domains.finance.finance_config(**overrides)` returns a `Config` tuned
for finance (higher recency + BM25 weight, redaction + injection scan on, sooner
abstention). Override any field via kwargs.
