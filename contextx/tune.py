"""Learned rank weights + feedback capture (#12).

The rank weights were hand-set in `Config`. This tunes the retrieval weights
(w_rerank / w_similarity / w_bm25) against LABELED data by maximizing nDCG@k —
turning a guess into a fitted parameter. It's a random search over the weight
simplex (cheap, no gradients, robust for 3 knobs); each query's candidate items
and their signals are computed once and reused across trials.

`FeedbackStore` captures thumbs / relevance labels from production so the tuning
set can grow from real usage, not just an offline golden set.

    from contextx.tune import tune_weights
    from contextx.eval import GOLDEN_QUERIES
    res = tune_weights(engine, GOLDEN_QUERIES)
    engine.cfg.w_rerank, engine.cfg.w_similarity, engine.cfg.w_bm25 = (
        res.weights["w_rerank"], res.weights["w_similarity"], res.weights["w_bm25"])
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass

import numpy as np

from .config import Config
from .eval import metrics as M


@dataclass
class TuneResult:
    weights: dict          # {w_rerank, w_similarity, w_bm25}
    ndcg: float
    baseline_ndcg: float


def tune_weights(engine, examples, k: int = 5, trials: int = 300, seed: int = 0) -> TuneResult:
    """Fit (w_rerank, w_similarity, w_bm25) to maximize mean nDCG@k on `examples`
    (each has `.query` and `.relevant` doc_ids)."""
    rng = np.random.default_rng(seed)

    per_query = []
    for ex in examples:
        cands = engine.recall_candidates(ex.query, rerank=True)
        if not cands:
            continue
        bm25 = engine.retriever.hybrid_scores(ex.query, cands)
        rows = [
            (c.rerank_score, c.similarity, bm25.get(c.item_id, 0.0), c.metadata.get("doc_id"))
            for c in cands
        ]
        per_query.append((rows, set(ex.relevant)))

    def mean_ndcg(wr: float, ws: float, wb: float) -> float:
        out = []
        for rows, rel in per_query:
            scored = sorted(rows, key=lambda r: wr * r[0] + ws * r[1] + wb * r[2], reverse=True)
            seen, ranked = set(), []
            for r in scored:
                if r[3] and r[3] not in seen:
                    seen.add(r[3])
                    ranked.append(r[3])
            out.append(M.ndcg_at_k(list(rel), ranked, k))
        return float(np.mean(out)) if out else 0.0

    d = Config()
    baseline = mean_ndcg(d.w_rerank, d.w_similarity, d.w_bm25)
    best = baseline
    best_w = (d.w_rerank, d.w_similarity, d.w_bm25)
    for _ in range(trials):
        wr, ws, wb = rng.dirichlet([1.0, 1.0, 1.0])  # sum to 1
        s = mean_ndcg(wr, ws, wb)
        if s > best:
            best, best_w = s, (wr, ws, wb)

    return TuneResult(
        weights={"w_rerank": float(best_w[0]), "w_similarity": float(best_w[1]),
                 "w_bm25": float(best_w[2])},
        ndcg=round(best, 4),
        baseline_ndcg=round(baseline, 4),
    )


class FeedbackStore:
    """Capture relevance/thumbs feedback for growing the tuning set."""

    def __init__(self, path: str) -> None:
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS feedback (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   ts REAL, request_id TEXT, query TEXT, relevant TEXT, rating INTEGER
               )"""
        )
        self._db.commit()

    def record(self, query: str, relevant: list[str], rating: int = 1,
               request_id: str = "") -> None:
        self._db.execute(
            "INSERT INTO feedback (ts, request_id, query, relevant, rating) VALUES (?,?,?,?,?)",
            (time.time(), request_id, query, json.dumps(relevant), rating),
        )
        self._db.commit()

    def examples(self):
        from .eval.dataset import EvalExample
        rows = self._db.execute(
            "SELECT query, relevant FROM feedback WHERE rating > 0"
        ).fetchall()
        return [EvalExample(q, json.loads(r)) for q, r in rows]
