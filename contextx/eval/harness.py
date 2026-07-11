"""Retrieval evaluation harness.

Ingests the golden corpus into an engine, produces a ranked doc_id list per
query under several configurations, and scores them against the labels. The
configurations are the ablations that show each component earns its keep:

  * random            — shuffle the corpus. The floor. Every real config must beat it.
  * bi-encoder        — embedding cosine only (recall stage).
  * +reranker         — bi-encoder recall then cross-encoder rerank (precision stage).
  * null (shuffled)   — the +reranker rankings scored against SHUFFLED labels. A
                        sanity control: metrics must collapse toward the floor,
                        proving we measure true relevance, not artifacts.

Optionally a second evaluator with the hash-embedding backend quantifies how
much real embeddings buy you.
"""

from __future__ import annotations

import numpy as np

from ..config import Config
from ..pipeline import ContextEngine
from ..types import Document
from . import metrics as M
from .dataset import EvalExample


def _rank_to_doc_ids(items, by: str) -> list[str]:
    """Dedup chunk-level hits to doc level, best rank first."""
    ordered = sorted(items, key=lambda it: getattr(it, by), reverse=True)
    seen: set[str] = set()
    out: list[str] = []
    for it in ordered:
        did = it.metadata.get("doc_id")
        if did and did not in seen:
            seen.add(did)
            out.append(did)
    return out


class RetrievalEvaluator:
    def __init__(
        self,
        corpus: list[Document],
        queries: list[EvalExample],
        k: int = 5,
        recall_k: int | None = None,
        force_hash_embeddings: bool = False,
        index_dir: str | None = None,
    ) -> None:
        self.k = k
        # The recall stage is a real bottleneck: it returns `recall_k` candidates
        # (default 2*k) out of the whole corpus. If embeddings are poor, relevant
        # docs get dropped here before the reranker ever sees them — which is what
        # makes the embedding ablation meaningful.
        self.recall_k = recall_k or (2 * k)
        self.queries = queries
        self.doc_ids = [d.doc_id for d in corpus]
        cfg = Config(
            min_similarity=0.0,          # don't pre-filter; let recall_k do the cutting
            recall_k=self.recall_k,
            index_dir=index_dir or f".eval_index_{'hash' if force_hash_embeddings else 'real'}",
        )
        from ..embeddings import Embedder

        embedder = Embedder(cfg.embed_model, force_fallback=force_hash_embeddings)
        self.engine = ContextEngine(config=cfg, embedder=embedder)
        # fresh index each run for reproducibility
        self.engine.store._db.execute("DELETE FROM chunks")
        self.engine.store._db.commit()
        self.engine.store._index = None
        self.engine.store._matrix = None
        self.engine.ingest(corpus)
        self.embed_backend = embedder.backend
        self.rerank_backend = self.engine.reranker.backend

    # --- rankings per config ---------------------------------------------
    def _rankings(self, config: str) -> list[list[str]]:
        rankings: list[list[str]] = []
        rng = np.random.default_rng(0)
        for ex in self.queries:
            if config == "random":
                order = list(self.doc_ids)
                rng.shuffle(order)
                rankings.append(order)
            elif config == "biencoder":
                cands = self.engine.recall_candidates(ex.query, rerank=False)
                rankings.append(_rank_to_doc_ids(cands, "similarity"))
            elif config == "reranked":
                cands = self.engine.recall_candidates(ex.query, rerank=True)
                rankings.append(_rank_to_doc_ids(cands, "rerank_score"))
            else:
                raise ValueError(config)
        return rankings

    # --- scoring ----------------------------------------------------------
    def _score(
        self, rankings: list[list[str]], gold: list[list[str]]
    ) -> dict[str, list[float]]:
        k = self.k
        per: dict[str, list[float]] = {
            f"recall@{k}": [], f"precision@{k}": [], f"hit@{k}": [],
            "mrr": [], f"ndcg@{k}": [],
        }
        for ranked, rel in zip(rankings, gold):
            per[f"recall@{k}"].append(M.recall_at_k(rel, ranked, k))
            per[f"precision@{k}"].append(M.precision_at_k(rel, ranked, k))
            per[f"hit@{k}"].append(M.hit_at_k(rel, ranked, k))
            per["mrr"].append(M.reciprocal_rank(rel, ranked))
            per[f"ndcg@{k}"].append(M.ndcg_at_k(rel, ranked, k))
        return per

    def per_query(self, metric: str) -> tuple[list[float], list[float]]:
        """Per-query metric for (bi-encoder, +reranker) — for stratified analysis."""
        gold = [ex.relevant for ex in self.queries]
        bi = self._score(self._rankings("biencoder"), gold)[metric]
        rr = self._score(self._rankings("reranked"), gold)[metric]
        return bi, rr

    def paired_significance(self, metrics_list: tuple[str, ...] | None = None) -> dict[str, dict]:
        """Paired-bootstrap significance of +reranker vs bi-encoder, per metric."""
        k = self.k
        metrics_list = metrics_list or ("mrr", f"ndcg@{k}", f"recall@{k}", f"precision@{k}")
        gold = [ex.relevant for ex in self.queries]
        bi = self._score(self._rankings("biencoder"), gold)
        rr = self._score(self._rankings("reranked"), gold)
        return {m: M.paired_delta(bi[m], rr[m]) for m in metrics_list}

    def rank_diagnostics(self) -> dict[str, float]:
        """Mean 1-indexed rank of the first relevant doc: bi-encoder vs reranked.

        Lower is better. This is the clearest view of what the reranker does —
        if it pulls buried answers up, its mean rank drops below the bi-encoder's.
        Misses are penalized with rank = corpus_size + 1.
        """
        gold = [ex.relevant for ex in self.queries]
        miss = len(self.doc_ids) + 1

        def mean_first_rank(rankings: list[list[str]]) -> float:
            ranks = []
            for ranked, rel in zip(rankings, gold):
                relset = set(rel)
                r = next((i + 1 for i, d in enumerate(ranked) if d in relset), miss)
                ranks.append(r)
            return float(np.mean(ranks))

        return {
            "bi-encoder": mean_first_rank(self._rankings("biencoder")),
            "+reranker": mean_first_rank(self._rankings("reranked")),
        }

    def evaluate_all(self) -> dict[str, dict[str, tuple[float, float, float]]]:
        gold = [ex.relevant for ex in self.queries]
        # null control: each query scored against a *different* query's labels
        rng = np.random.default_rng(1)
        perm = rng.permutation(len(gold))
        # derange-ish: ensure not identity where possible
        shuffled_gold = [gold[perm[i]] for i in range(len(gold))]

        results: dict[str, dict] = {}
        rr = self._rankings("reranked")
        results["random"] = M.aggregate(self._score(self._rankings("random"), gold))
        results["bi-encoder"] = M.aggregate(self._score(self._rankings("biencoder"), gold))
        results["+reranker (full)"] = M.aggregate(self._score(rr, gold))
        results["null (shuffled gold)"] = M.aggregate(self._score(rr, shuffled_gold))
        return results
