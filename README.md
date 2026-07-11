# contextx — a context-optimization engine

Turns a raw user request plus scattered sources into **one tight, validated,
in-budget prompt** for an LLM — then calls the model. Built around the
**ingest/query split** so retrieval cost is independent of corpus size.

```
INGEST (amortized):  documents → chunk → embed → persistent FAISS HNSW index
QUERY  (per request): Collect → Retrieve → Rerank → Rank → Filter → Compress
                      → Budget → Build → Validate → LLM
                      └ Memory (SQLite, read+write) · Cache (semantic) wrap it ┘
```

## Quick start

```python
from contextx import ContextEngine, Request, Document

engine = ContextEngine()
engine.ingest([Document(text="A refresh token is a long-lived credential ...")])   # once
result = engine.run(Request(user_message="Explain JWT refresh tokens"))            # per request
print(result.answer)
print(result.trace.report())
```

```bash
python examples/demo.py     # ingest a corpus, answer a query, print the trace
pytest tests/               # unit + integration
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...   # real answers instead of the mock
```

Runs with **zero installs** via fallbacks and upgrades transparently as each
backend appears. The trace footer shows which backend each stage actually used.

| Concern       | Production backend         | Fallback                          |
|---------------|----------------------------|-----------------------------------|
| Embeddings    | `sentence-transformers`    | deterministic hash embedder       |
| Vector index  | `faiss-cpu` HNSW (ANN)     | persisted numpy brute-force cosine|
| Reranker      | cross-encoder              | identity (keep bi-encoder order)  |
| Generation    | Claude (`anthropic` + key) | extractive mock                   |
| Token count   | `tiktoken` (+ safety margin)| ~4-chars/token heuristic         |

## What makes this not a toy

| Concern | Toy (v0.1) | Now (v0.2) |
|---------|------------|------------|
| **Scaling** | re-embedded the whole pool + rebuilt a flat index every request | ingest once → **persistent HNSW ANN**; query embeds only the query |
| **Long docs** | one vector per document | **chunking** with overlap at ingest |
| **Retrieval quality** | bi-encoder cosine only | **recall → cross-encoder rerank** (precision stage) |
| **Memory** | JSON file rewritten per turn (race, unbounded) | **SQLite** WAL, thread-safe, atomic, **bounded** w/ eviction |
| **Tokenizer** | OpenAI cl100k as if it were Claude | local estimate **+ safety margin**; exact Claude `count_tokens` hook |
| **Cost** | no caching | **prompt caching** breakpoints + **semantic** response cache |
| **Security** | none | untrusted content **fenced + instructed**; **injection scan** |
| **Resilience** | one bare API call | **retries + exp backoff + jitter + timeout** |
| **Config** | magic numbers everywhere | one `Config` dataclass |

## Stage map

| # | Stage | File | Notes |
|---|-------|------|-------|
| — | Ingest | `store.py`, `chunking.py` | chunk → embed → persist (HNSW + SQLite sidecar) |
| 1 | Collect | `collect.py` | ephemeral only (conversation, tool outputs); durable is pre-indexed |
| 2 | Retrieve | `retrieve.py` | durable ANN recall + inline ephemeral + BM25 hybrid |
| 2b| Rerank | `rerank.py` | cross-encoder precision over the recall set |
| 3 | Rank | `rank.py` | weighted blend (rerank + sim + bm25 + recency + importance …) |
| 4 | Filter | `filter.py` | semantic dedup, stale, contradiction resolution |
| 5 | Memory | `memory.py` | SQLite, 5 types, lifecycle, bounded |
| 6 | Compress | `compress.py` | extractive default / abstractive optional |
| 7 | Budget | `budget.py` | plan window, safety margin, trim tail |
| 8 | Build | `build.py` | trust-fenced blocks + prompt-cache markers |
| 9 | Validate | `validate.py` | token ceiling, delimiters, injection scan |
| 10| LLM | `llm.py` | resilient Claude client (or mock) |
| 11| Observability | `observability.py` | per-stage Trace + metrics |
| 12| Cache | `cache.py` | TTL + LRU + semantic response cache |

## Evaluation — prove it works, don't assert it

An unmeasured retrieval system is a toy. `contextx/eval/` is an offline harness
with a labeled golden set (32 docs incl. hard negatives, 12 queries), standard
IR metrics with **95% bootstrap CIs**, and ablations:

```bash
python examples/run_eval.py
```

```
config                recall@5          nDCG@5            mrr
random                0.28 [0.06,0.53]  0.21 [0.04,0.41]  0.26   <- floor
bi-encoder            1.00 [1.00,1.00]  0.98 [0.96,1.00]  1.00
+reranker (full)      1.00 [1.00,1.00]  0.97 [0.95,0.99]  1.00
null (shuffled gold)  0.04 [0.00,0.12]  0.03 [0.00,0.10]  0.04   <- sanity control
```

There are two benchmarks over the same corpus. The **EASY** set (topic-separated
queries) establishes the controls: the null collapses to the floor (metric is
valid), and real embeddings beat the hash fallback ~7pp recall@5 (backend earns
its place). But everything saturates (bi-encoder recall@5 = MRR = 1.0), so it
can't measure the reranker.

The **HARD** set is built from lexical traps + multi-hop queries — a decoy doc
outranks the true answer on surface similarity (e.g. "full-text search" → the
Elasticsearch decoy, when the answer is a Postgres GIN index). This breaks
bi-encoder saturation and lets the reranker show its value:

```
HARD (k=5)      MRR               nDCG@5            mean rank of 1st answer
bi-encoder      0.72 [0.53,0.88]  0.80 [0.66,0.92]  1.70
+reranker       0.85 [0.70,1.00]  0.88 [0.77,0.98]  1.30   (pulls answers up)
null (shuffled) 0.16 [0.04,0.36]  0.10 [0.00,0.30]  —
```

Reading it honestly:
- **the reranker earns its place when surface similarity lies** — +0.13 MRR,
  +0.09 nDCG, first correct answer rises from rank ~1.7 to ~1.3. This is the
  regime real queries live in.
- **but it is not yet statistically significant**: n=10, so the CIs overlap. The
  direction is consistent across all three metrics; proving significance needs a
  bigger labeled set (~50-100 queries). That is the concrete next task.

Bring your own data: `load_jsonl` a file of `{"query", "relevant":[doc_id,...]}`.

## Not done yet (honest roadmap)

- **Async collect/retrieve** — sources are still fetched synchronously.
- **Learned rank weights** — currently fixed in `Config`; wire stage-11 feedback.
- **LLM-based fact extraction** in memory — still a heuristic.
- **Deployment**: managed vector DB option, horizontal scaling, load test, an
  offline retrieval-quality eval harness (precision/recall vs a labeled set).
- **Streaming** responses and tool use in the LLM stage.
```
