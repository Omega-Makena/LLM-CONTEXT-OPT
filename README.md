# contextx — a context-optimization engine

Turns a raw user request plus scattered sources into **one tight, validated,
in-budget prompt** for an LLM — then calls the model. Built around the
**ingest/query split** so retrieval cost is independent of corpus size.

> **Domain packs:** a finance preset (capital markets · payments · lending ·
> compliance/KYC-AML) ships in `contextx/domains/` — see [FINANCE.md](FINANCE.md).

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
python examples/demo.py         # ingest a corpus, answer a query, print the trace
python examples/eval_custom.py  # eval + tune retrieval on your own JSONL data
pytest tests/                   # unit + integration
pip install -e ".[full]"
```

**LLM backends** — pluggable: **Claude**, **OpenAI**, local **Ollama**, or a
mock. `Config.llm_provider="auto"` resolves Claude (`ANTHROPIC_API_KEY`) →
OpenAI (`OPENAI_API_KEY`) → Ollama (local server reachable) → mock; force any
with `llm_provider="anthropic"|"openai"|"ollama"|"mock"`. The OpenAI backend is
OpenAI-compatible, so `Config(openai_base_url=...)` also targets Azure OpenAI,
OpenRouter, vLLM, or LM Studio. Ollama: `ollama serve && ollama pull llama3.1` —
no key, no cost, fully local.

Runs with **zero installs** via fallbacks and upgrades transparently as each
backend appears. The trace footer shows which backend each stage actually used.

| Concern       | Production backend         | Fallback                          |
|---------------|----------------------------|-----------------------------------|
| Embeddings    | `sentence-transformers`    | deterministic hash embedder       |
| Vector index  | `faiss-cpu` HNSW (ANN)     | persisted numpy brute-force cosine|
| Reranker      | cross-encoder              | identity (keep bi-encoder order)  |
| Generation    | Claude, or local **Ollama**| extractive mock                   |
| Token count   | `tiktoken` (+ safety margin)| ~4-chars/token heuristic         |

**Bring your own data:** point the eval at your labeled queries —
`python examples/eval_custom.py --corpus mycorpus.jsonl --queries myqueries.jsonl`
(`{"text","doc_id"}` and `{"query","relevant":[doc_id]}` per line) — to get IR
metrics + nDCG-tuned weights on *your* domain.

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

## Production capabilities (v0.3)

| Area | Capability |
|------|-----------|
| **Correctness** | document **delete/update** with index compaction; **embedding-model version stamping** (refuses a mismatched index); **real hybrid search** (SQLite FTS5 + reciprocal-rank fusion); **citations** + retrieval **abstention** |
| **Multi-tenancy** | tenant isolation + **permission-filtered retrieval** (ACLs enforced in both recall channels); memory namespaced per user |
| **Service** | **FastAPI** app (`/ingest`, `/query`, `/documents`, `/health`, `/stats`) running the engine in a threadpool; `Dockerfile` |
| **Ingestion** | loaders (txt/md/html/pdf); **structure-aware chunking** (headings + atomic code blocks); **incremental sync** (content-hash diff) |
| **Quality/trust** | **faithfulness/groundedness eval**; **learned rank weights** (nDCG search on labeled data) + feedback capture; **PII redaction** + **audit log** |
| **Ops** | per-request **cost in USD**, request IDs, structured JSON logs |
| **Packaging** | `pip install contextx` (extras: `retrieval`/`llm`/`serve`/`ingest`/`full`); MIT license; **GitHub Actions CI** with a retrieval **regression gate** |

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

The **HARD** set (68 queries) is built from lexical traps + multi-hop queries —
a decoy doc outranks the true answer on surface similarity (e.g. "full-text
search" → the Elasticsearch decoy, when the answer is a Postgres GIN index).

```
HARD (k=5)      MRR               nDCG@5            mean rank of 1st answer
bi-encoder      0.94 [0.89,0.98]  0.95 [0.91,0.98]  1.24
+reranker       0.97 [0.93,0.99]  0.97 [0.95,0.99]  1.07
null (shuffled) 0.07 [0.03,0.12]  0.06 [0.02,0.12]  —
```

Because bi-encoder and reranker are scored on the *same* queries, the correct
test is a **paired** one, not a comparison of marginal CIs:

```
+reranker vs bi-encoder      delta      95% CI            p(1-sided)  win rate
mrr                          +0.030   [-0.020,+0.084]    0.125        9%
ndcg@5                       +0.025   [-0.015,+0.070]    0.124        9%
```

Reading it honestly — and this is the interesting part:

- **Overall, the reranker's lift is NOT significant** (p≈0.13). Not because it
  fails, but because on 61 of 68 queries the bi-encoder already puts the answer
  at rank 1 — there is nothing to fix, and the effect gets diluted to +0.03 MRR.
- **Conditioned on the 7 queries the bi-encoder gets wrong, it is decisive:**

  ```
  bi-encoder already perfect on 61/68; 7 queries are genuinely hard
  on those 7: reranker improved 6, degraded 0, tied 1
  paired MRR delta on hard stratum: +0.532 [+0.333, +0.706]  p < 0.001
  ```

- So the reranker is a **low-risk, always-on component: it never regresses and it
  rescues the hard ~10%.** That is the honest answer to "is it worth the latency?"

Two caveats stated plainly: the hard stratum is small (n=7) and selected on the
baseline, so treat p<0.001 there as strong *directional* evidence plus a clean
"never regresses" safety property — confirming it needs a larger set sampled
*toward difficulty*, not just more queries (scaling n from 10→68 actually lowered
the aggregate effect by adding easy queries). And this is retrieval quality on a
synthetic corpus; the method transfers to real data via `load_jsonl`
(`{"query", "relevant":[doc_id,...]}`), which is where it should next be run.

## Not done yet (honest roadmap)

- **A larger, difficulty-stratified eval set** — enrich toward hard queries so
  the reranker's effect is significant on the *aggregate*, not just the stratum.
- **LLM-judge faithfulness at scale** — the harness supports an LLM judge; wiring
  it into CI needs an API key + budget. The offline embedding-overlap proxy ships.
- **True async I/O** — endpoints run the sync engine in a threadpool; a fully
  async retrieval path (async source fetchers, batching) is future work.
- **LLM-based fact extraction** in memory — still a heuristic extractor.
- **Managed vector DB backend** — FAISS HNSW is in-process; the `VectorStore`
  interface is the seam to swap in pgvector/Qdrant for horizontal scale.
- **Streaming** responses and tool use in the LLM stage.
```
