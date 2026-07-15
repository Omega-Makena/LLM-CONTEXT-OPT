# Evaluation

`contextx/eval/` measures retrieval and answer quality.

## Retrieval evaluation

```bash
python examples/run_eval.py                 # golden + hard benchmarks
python examples/eval_custom.py --corpus c.jsonl --queries q.jsonl   # your data
```

### Method
Each query is labeled with the relevant `doc_id`s (ground truth). The harness
(`eval/harness.py`) runs retrieval under several configurations and scores the
ranking against the labels.

### Metrics (`eval/metrics.py`)
Binary relevance, reported as mean with a 95% bootstrap confidence interval over
queries:

| Metric | Meaning |
|---|---|
| `recall@k` | fraction of relevant docs found in the top k |
| `precision@k` | fraction of the top k that are relevant |
| `hit@k` | 1 if any relevant doc is in the top k |
| `mrr` | mean reciprocal rank of the first relevant doc |
| `ndcg@k` | position-discounted gain, normalized to the ideal ranking |

### Configurations (ablations)
Each run compares:
- **random** — the floor; every real config must beat it.
- **bi-encoder** — embedding cosine only (recall stage).
- **+reranker** — bi-encoder recall then cross-encoder rerank (precision stage).
- **null (shuffled gold)** — the +reranker rankings scored against *shuffled*
  labels; a sanity control that must collapse toward the floor, proving the
  metric measures true relevance and not an artifact.

**Paired significance** (`metrics.paired_delta`): bi-encoder and reranker are
scored on the same queries, so it uses a paired bootstrap on per-query
differences. Reports mean delta, 95% CI, one-sided p-value, and win-rate.

### Reading results
- If the null control does not collapse, the benchmark is too small to
  discriminate — add distractors and hard negatives.
- If a component shows no aggregate lift, check whether the bi-encoder already
  saturates; the effect may be concentrated in a hard stratum.

### Difficulty-stratified set
`contextx/eval/hard_plus.py` is a set of confusable clusters (option greeks,
liquidity/leverage ratios, order types, risk measures, bond types, payment
rails) where docs share vocabulary and differ by one detail.

```bash
python examples/eval_hard_plus.py
```

On this set the reranker's lift is significant on the aggregate (MRR +0.046,
p=0.041; nDCG@5 +0.034, p=0.041), where a less-confusable set gives p≈0.13.
recall@5 still saturates; the significant lift is in ordering.

## Bring your own data

Two JSONL files, one object per line:

```
corpus.jsonl:   {"text": "...", "doc_id": "acme-10k-2024", "metadata": {...}}
queries.jsonl:  {"query": "what was revenue?", "relevant": ["acme-10k-2024"]}
```

```bash
python examples/eval_custom.py --corpus corpus.jsonl --queries queries.jsonl --k 5
```

This prints the metrics table and the nDCG-tuned rank weights for your domain.
Loaders: `contextx.eval.load_corpus_jsonl`, `contextx.eval.load_jsonl`.

## Tuning rank weights (`tune.py`)

Don't hand-guess weights. `tune_weights` fits `w_rerank / w_similarity / w_bm25`
to maximize mean nDCG@k on labeled data via random search over the weight simplex
(candidate signals are computed once per query and reused across trials):

```python
from contextx.tune import tune_weights
from contextx.eval import GOLDEN_QUERIES

res = tune_weights(engine, GOLDEN_QUERIES, k=5, trials=300)
print(res.baseline_ndcg, "->", res.ndcg, res.weights)
engine.cfg.w_rerank, engine.cfg.w_similarity, engine.cfg.w_bm25 = (
    res.weights["w_rerank"], res.weights["w_similarity"], res.weights["w_bm25"])
```

`FeedbackStore` (SQLite) captures thumbs/relevance labels from production so the
tuning set can grow from real usage.

## Answer-quality evaluation (`eval/faithfulness.py`)

Retrieval metrics say nothing about whether the *answer* is supported by the
context (hallucination). `FaithfulnessScorer` splits the answer into claims and
checks each is backed by a source:

- **offline (default)**: embedding overlap — a claim is supported if its max
  cosine to any source chunk clears a threshold. No API key needed.
- **LLM judge (optional)**: pass an `llm` callable; each claim is judged
  SUPPORTED / UNSUPPORTED.

```python
from contextx.eval.faithfulness import FaithfulnessScorer
s = FaithfulnessScorer()
r = s.score(answer_text, [src["preview"] for src in result.sources])
print(r.groundedness, r.unsupported_claims)
```

Low groundedness with high retrieval recall points at **generation**
(hallucination); low recall points at **retrieval**.

## CI regression gate (`scripts/eval_gate.py`)

Run in CI on every push. It executes the retrieval eval on the golden set and
**fails the build** if:
- `+reranker` recall@5 drops below a floor (well above random), or
- the shuffled-gold null control rises above a ceiling (metric invalid), or
- the reranker fails to beat random.

Thresholds are conservative so they hold on CI's fallback backends (hash
embeddings, identity reranker) — they catch *regressions*, not absolute quality.
