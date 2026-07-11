"""Standard information-retrieval metrics + bootstrap confidence intervals.

Binary relevance (a doc is relevant or not). All metrics take the ground-truth
relevant doc_ids and the ranked list of retrieved doc_ids.

  * recall@k     — fraction of relevant docs found in the top k
  * precision@k  — fraction of the top k that are relevant
  * hit@k        — 1 if any relevant doc is in the top k (a.k.a. success@k)
  * MRR          — mean reciprocal rank of the first relevant doc
  * nDCG@k       — position-discounted gain, normalized to the ideal ranking

Point estimates lie. `aggregate` reports the mean with a 95% bootstrap CI over
queries so you can see whether a difference between configs is real or noise.
"""

from __future__ import annotations

import numpy as np


def recall_at_k(relevant: list[str], ranked: list[str], k: int) -> float:
    rel = set(relevant)
    if not rel:
        return 0.0
    return len(rel & set(ranked[:k])) / len(rel)


def precision_at_k(relevant: list[str], ranked: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    rel = set(relevant)
    return len(rel & set(ranked[:k])) / k


def hit_at_k(relevant: list[str], ranked: list[str], k: int) -> float:
    return 1.0 if set(relevant) & set(ranked[:k]) else 0.0


def reciprocal_rank(relevant: list[str], ranked: list[str]) -> float:
    rel = set(relevant)
    for i, d in enumerate(ranked):
        if d in rel:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(relevant: list[str], ranked: list[str], k: int) -> float:
    rel = set(relevant)
    dcg = sum(1.0 / np.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in rel)
    ideal = min(len(rel), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def paired_delta(
    baseline: list[float], treatment: list[float], n_boot: int = 5000, seed: int = 0
) -> dict[str, float]:
    """Paired bootstrap on (treatment - baseline), same queries.

    Because both configs are scored on identical queries, the per-query
    difference removes query-difficulty variance — far more powerful than
    comparing two marginal CIs. Returns the mean delta, its 95% CI, the share
    of queries where treatment strictly wins, and a one-sided bootstrap p-value
    for H1: treatment > baseline (fraction of resamples with mean delta <= 0).
    """
    a = np.asarray(baseline, dtype=float)
    b = np.asarray(treatment, dtype=float)
    d = b - a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "delta": float(d.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "p_one_sided": float((boots <= 0).mean()),
        "win_rate": float((d > 0).mean()),
        "n": int(d.size),
    }


def aggregate(
    per_query: dict[str, list[float]], n_boot: int = 2000, seed: int = 0
) -> dict[str, tuple[float, float, float]]:
    """mean + 95% bootstrap CI (resampling queries) for each metric."""
    rng = np.random.default_rng(seed)
    out: dict[str, tuple[float, float, float]] = {}
    for name, vals in per_query.items():
        arr = np.asarray(vals, dtype=float)
        if arr.size == 0:
            out[name] = (0.0, 0.0, 0.0)
            continue
        idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
        boots = arr[idx].mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out[name] = (float(arr.mean()), float(lo), float(hi))
    return out
