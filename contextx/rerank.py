"""Cross-encoder reranking — the precision stage.

Bi-encoder cosine (stage 2) is a cheap *recall* filter: it embeds query and doc
independently, so it misses fine-grained relevance. A cross-encoder scores the
(query, doc) pair *jointly* and is far more accurate — but too slow to run over
a whole corpus, so we only run it over the recall set (top `recall_k`).

This recall-then-rerank pattern is where most real-world retrieval quality comes
from. Falls back to identity (keep bi-encoder order) if the model isn't
installed, so the pipeline still runs.
"""

from __future__ import annotations

from .config import Config
from .types import ContextItem


class Reranker:
    def __init__(self, config: Config | None = None) -> None:
        self.cfg = config or Config()
        self._model = None
        self.backend = "identity"
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.cfg.rerank_model)
            self.backend = "cross-encoder"
        except Exception:
            self._model = None

    def rerank(self, query: str, items: list[ContextItem]) -> list[ContextItem]:
        """Set `.rerank_score` on each item; return sorted best-first."""
        if not items:
            return items
        if self._model is None:
            # identity fallback: carry cosine similarity through as the score
            for it in items:
                it.rerank_score = it.similarity
            return sorted(items, key=lambda it: it.rerank_score, reverse=True)

        pairs = [(query, it.text) for it in items]
        scores = self._model.predict(pairs)
        # min-max normalize to 0..1 so it blends with other signals
        lo, hi = float(min(scores)), float(max(scores))
        span = (hi - lo) or 1.0
        for it, s in zip(items, scores):
            it.rerank_score = (float(s) - lo) / span
        return sorted(items, key=lambda it: it.rerank_score, reverse=True)
