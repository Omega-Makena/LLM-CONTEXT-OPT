"""Stage 3 — Context Ranking.

Blend the signals into one score. The dominant term is the cross-encoder
`rerank_score` (precision); the rest are tie-breakers/priors:

    score = w_rerank * rerank_score          (joint query-doc relevance)
          + w_sim    * semantic_similarity    (bi-encoder recall signal)
          + w_bm25   * keyword_similarity      (lexical / exact-term match)
          + w_recent * recency                 (exponential half-life)
          + w_imp    * importance              (pinned / critical facts)
          + w_conv   * conversation_relevance
          + w_pref   * preference_match

All component signals are in 0..1 so the linear blend is comparable. Weights
live in `Config` (tunable / learnable), not scattered as literals.
"""

from __future__ import annotations

import math

from .config import Config
from .types import SOURCE_PRIOR, ContextItem, Request, Source


class Ranker:
    def __init__(self, config: Config | None = None) -> None:
        self.cfg = config or Config()

    def rank(
        self,
        request: Request,
        items: list[ContextItem],
        bm25: dict[str, float] | None = None,
    ) -> list[ContextItem]:
        c = self.cfg
        bm25 = bm25 or {}
        pref_terms = _preference_terms(request.preferences)

        for it in items:
            recency = math.exp(-it.age_seconds / c.recency_half_life_s)
            keyword = bm25.get(it.item_id, 0.0)

            pinned = float(it.metadata.get("pinned", False))
            imp = max(it.importance, pinned, SOURCE_PRIOR.get(it.source, 0.5) * 0.5)
            it.importance = imp

            conv = 1.0 if it.source == Source.CURRENT_CONVERSATION else 0.0
            pref = _preference_match(it, pref_terms)

            it.score = (
                c.w_rerank * it.rerank_score
                + c.w_similarity * it.similarity
                + c.w_bm25 * keyword
                + c.w_recency * recency
                + c.w_importance * imp
                + c.w_conversation * conv
                + c.w_preference * pref
            )

        return sorted(items, key=lambda it: it.score, reverse=True)


def _preference_terms(preferences: dict) -> set[str]:
    terms: set[str] = set()
    for v in preferences.values():
        if isinstance(v, str):
            terms.add(v.lower())
        elif isinstance(v, (list, tuple)):
            terms.update(str(x).lower() for x in v)
    return terms


def _preference_match(item: ContextItem, pref_terms: set[str]) -> float:
    if not pref_terms:
        return 0.0
    text = item.text.lower()
    return 1.0 if any(t in text for t in pref_terms) else 0.0
