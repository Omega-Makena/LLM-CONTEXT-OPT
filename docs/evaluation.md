# Evaluation

An unmeasured retrieval system is a toy. `contextx/eval/` measures retrieval and
answer quality so you can defend and tune the engine rather than assume it works.

## Retrieval evaluation

```bash
python examples/run_eval.py                 # golden + hard benchmarks
python examples/eval_custom.py --corpus c.jsonl --queries q.jsonl   # your data
```

### The method
Given a labeled set — queries each annotated with the `doc_id`s that are
*relevant* (ground truth) — the harness (`eval/harness.py`) runs retrieval under
several configurations and scores the returned ranking against the labels.

### Metrics (`eval/metrics.py`)
Binary relevance. Reported as **mean with a 95% bootstrap confidence interval**
over queries (point estimates lie):

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

There is also a **paired significance** test (`metrics.paired_delta`): because
bi-encoder and reranker are scored on the *same* queries, the correct comparison
is a paired bootstrap on per-query differences (far more powerful than comparing
two marginal CIs). It reports the mean delta, 95% CI, one-sided p-value, and
win-rate.

### Reading results honestly
- If the **null control** does not collapse, your benchmark is too small/easy to
  discriminate — add distractors and hard negatives.
- If a component (e.g. the reranker) shows no aggregate lift, check whether the
  bi-encoder already saturates; the value may be concentrated in a hard stratum.
  Report that, don't hide it.

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
