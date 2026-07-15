# How context optimization works

This is the heart of the system: the method by which contextx turns a raw user
request plus a pile of scattered sources into a single, tight, high-signal prompt
for a large language model.

## 1. The problem

An LLM is a function of its input. It has no memory of your data and a **fixed
context window** (e.g. 32k–200k tokens). Everything it "knows" for a given
request is whatever text you place in that window. So the quality of an answer is
bounded by the quality of the context you assemble.

Naively, you might dump everything relevant into the prompt. That fails for four
reasons:

1. **Size** — real corpora are far larger than any context window.
2. **Signal** — irrelevant text *dilutes* the model's attention and degrades
   answers (the "lost in the middle" effect).
3. **Cost & latency** — you pay per token and wait per token.
4. **Trust** — dumping raw retrieved text invites prompt injection, PII leakage,
   stale data, and un-attributable claims.

Context optimization is the discipline of selecting, ordering, compressing, and
framing exactly the right context. contextx implements it as a pipeline.

## 2. The core idea: split ingest from query

The single most important design decision. A toy RAG system re-embeds its whole
corpus on every request — O(N) per query, impossible at scale. contextx separates
two phases:

```
INGEST (amortized, once per document)
    document ──▶ chunk ──▶ embed ──▶ persistent ANN index + SQLite sidecar

QUERY (per request)
    request ──▶ embed the QUERY only ──▶ search the pre-built index ──▶ ... ──▶ LLM
```

At query time we embed **only the query** and search a persistent index whose
cost is independent of corpus size. Ingest is where the expensive per-document
work happens, and it survives process restarts.

See `contextx/store.py` (the index) and `contextx/pipeline.py` (`ingest()` vs
`run()`).

## 3. The unit of flow: `ContextItem`

Everything that moves through the pipeline is a `ContextItem` (`contextx/types.py`).
Each stage reads some fields and writes others, so an item accumulates provenance
and scores as it travels:

| Field | Set by | Meaning |
|---|---|---|
| `text`, `source`, `timestamp`, `metadata` | collection / ingest | the content and where it came from |
| `embedding` | retrieval | its vector |
| `similarity` | retrieval | cosine to the query (bi-encoder) |
| `rerank_score`, `raw_rerank_score` | rerank | cross-encoder relevance (normalized / raw) |
| `rrf_score` | retrieval | hybrid reciprocal-rank-fusion score |
| `score` | ranking | final blended rank score |
| `importance` | ranking / memory | pinned / critical weight |
| `tokens`, `included` | budget | token count; whether it made the cut |
| `trusted` | source | false for externally-authored (untrusted) content |

Reading the trace (stage 11) shows these counts shrink as items are filtered out.

## 4. The pipeline, stage by stage

The numbers match the trace output. Cache (12) and Memory (5) are cross-cutting.

### Stage 1 — Collect (`collect.py`)
Gather the per-request **ephemeral** context into `ContextItem`s: the user
message, this turn's conversation, freshly-fetched tool/API outputs. Durable
knowledge (documents/KB) is *not* collected here — it lives in the pre-built
index and is pulled at retrieval. Nothing is selected yet; we only normalize,
attach metadata, and record source + timestamp.

### Stage 2 — Retrieve (`retrieve.py`, `store.py`)
Find the relevant durable context, and score the ephemeral context.

- **Durable recall** queries the persistent index. Two channels run:
  - *Semantic* (`store.search`): embed the query, ANN-search the HNSW index
    (inner product over L2-normalized vectors = cosine).
  - *Lexical* (`store.lexical_search`): BM25 over a SQLite FTS5 index — catches
    exact tokens that embeddings miss (IDs, error codes, tickers, rare terms).
- The two channels are fused with **Reciprocal Rank Fusion (RRF)**:
  `score(doc) = Σ 1/(k + rank_in_channel)`. This is *real* hybrid search — a
  doc found only lexically still surfaces.
- **Ephemeral items** (small N) are embedded inline and scored against the query.
- **Adaptive over-fetch**: because tenant/ACL filtering is applied *after* the
  ANN search, a small tenant's docs could all sit beyond the first `recall_k`
  neighbours. The search expands the fetch (×4 each round) until it has `k`
  authorized hits or the index is exhausted — so a small tenant is never starved.

### Stage 2b — Rerank (`rerank.py`)
The bi-encoder cosine from stage 2 is a cheap **recall** filter: it embeds query
and document *independently*, so it is fooled by surface overlap. A
**cross-encoder** scores the (query, document) pair *jointly* with full
attention — far more accurate, but too slow to run over the whole corpus, so we
only run it over the recall set (`recall_k` → `rerank_k`). This recall-then-rerank
pattern is where most real-world retrieval quality comes from.

It also produces the **abstention** signal: if the best *raw* cross-encoder score
is below `abstain_below`, the retrieval is flagged `low_confidence` so the answer
path can decline to guess instead of hallucinating.

### Stage 3 — Rank (`rank.py`)
Blend the signals into one score (weights live in `Config`, tunable/learnable):

```
score = w_rerank * rerank_score       (joint relevance — dominant term)
      + w_similarity * similarity      (bi-encoder recall signal)
      + w_bm25 * keyword_score         (lexical / exact-term match)
      + w_recency * recency            (exponential half-life — fresher wins)
      + w_importance * importance      (pinned / critical facts)
      + w_conversation * conv          (discussed this turn)
      + w_preference * pref            (matches user preferences)
```

All component signals are in 0..1 so the linear blend is comparable.

### Stage 4 — Filter & Dedup (`filter.py`)
Ranking is not enough; the top list still contains noise. This stage removes:
- **near-duplicates** (embedding cosine ≥ `dup_threshold`, greedy keep-the-best),
- **stale** items (`metadata['expires_at']` in the past),
- and resolves **contradictions** on structured facts (`fact_key`/`fact_value` —
  highest score wins, tie-broken by recency).

### Stage 5 — Memory (`memory.py`)  *(cross-cutting)*
A durable, per-user store (SQLite, WAL, thread-safe, bounded). Read at stage 1
(relevant memories join the candidate pool) and written after the answer
(extract facts from the exchange). Five memory types (working/session/long-term/
episodic/semantic), a lifecycle (store → score → retrieve → expire/forget), and
eviction by importance × recency (pinned facts survive). Scoped per `tenant:user`
so one user's facts never leak to another.

### Stage 6 — Compress (`compress.py`)
Shrink oversized items so we keep meaning per token. Extractive by default (keep
the sentences most similar to the query); abstractive optionally (an LLM rewrite).
Only items over `max_item_tokens` are touched.

### Stage 7 — Budget (`budget.py`)
Everything must fit the context window with room reserved for the response. Count
tokens, subtract the system prompt + user message + reserved output, apply a
safety margin (tokenizer drift), then greedily include the highest-ranked items
until the budget is spent — trimming the lowest-ranked tail.

> Note: `tiktoken` (cl100k) is OpenAI's tokenizer, not Claude's, so counts drift
> ~10–20%; the `budget_safety_margin` absorbs that.

### Stage 8 — Build (`build.py`)
Assemble the final prompt with three production concerns:
- **Trust boundary** — content from untrusted sources is fenced in an
  `<untrusted_context>` block with an explicit instruction that it is DATA, not
  instructions (the primary prompt-injection mitigation: delimiting + instruction).
- **Citations** — retrieved knowledge is numbered `[1] … [n]`, the model is told
  to cite, and the sources map is returned with the answer.
- **Prompt caching** — the system prompt is emitted as a cacheable block so the
  provider can reuse the prefix across calls (a large cost saver).

### Stage 9 — Validate (`validate.py`)
The last gate before the model call: token ceiling, non-empty essentials,
balanced delimiters, leftover duplicates, and a **prompt-injection scan** over
untrusted content. A hard failure blocks the LLM call.

### Stage 10 — LLM (`llm.py`)
Call the model. Pluggable backends — **Claude**, **OpenAI** (and OpenAI-compatible
gateways), local **Ollama**, or a **mock** — with retries + exponential backoff +
jitter + timeout, prompt-cache passthrough, and **streaming** (`stream()`).

### Stage 11 — Observability (`observability.py`)  *(cross-cutting)*
A `Trace` threaded through every stage records per-stage timing, in/out counts,
token usage, **estimated dollar cost**, a request id, and a structured JSON log
line. This is the single best way to *see* the engine working.

### Stage 12 — Cache (`cache.py`)  *(cross-cutting)*
Bounded (LRU) + TTL cache with a **semantic response cache**: a new request is
matched against prior ones by embedding cosine, so *paraphrases* hit — exact
string caching almost never hits in real traffic. Invalidated on ingest/update/
delete so an edited document can't serve a stale answer.

## 5. Cross-cutting properties

- **Multi-tenancy** — tenant isolation + per-document ACLs enforced in *both*
  recall channels; memory namespaced per user; tenant-scoped delete.
- **Graceful degradation** — every heavy backend has a fallback (hash embeddings,
  numpy index, identity reranker, mock LLM, char-based tokenizer), so the whole
  pipeline runs with zero installs and upgrades transparently.
- **Honesty** — the engine reports confidence (abstention), cites sources, and is
  measured by an eval harness rather than assumed to work.

## 6. Why this order?

The stages are ordered to spend expensive work only on survivors:
cheap recall (2) → expensive rerank on the recall set (2b) → cheap blends and
filters (3, 4) → compression and budgeting on what's left (6, 7) → assembly and
one model call (8–10). Each stage reduces the item count, so the costly stages
operate on the smallest possible set.
