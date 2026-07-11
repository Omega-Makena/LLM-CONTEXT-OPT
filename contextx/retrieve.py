"""Stage 2 — Retrieval (query time).

Two kinds of context, handled correctly:

  * DURABLE knowledge (KB, documents) — pre-indexed at ingest. We query the
    persistent HNSW index; cost is independent of corpus size. This is the fix
    for the toy's "re-embed the whole pool every request".
  * EPHEMERAL context (this turn's conversation, freshly-fetched tool outputs,
    retrieved memories) — small N, not worth indexing; embed inline.

We merge both into one candidate set, attach BM25 (lexical) scores for hybrid
search, and hand off to the reranker. Only the query embedding is computed fresh
per request.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from .cache import Cache
from .config import Config
from .embeddings import Embedder
from .store import VectorStore
from .types import ContextItem

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        config: Config | None = None,
        cache: Cache | None = None,
    ) -> None:
        self.cfg = config or Config()
        self.embedder = embedder
        self.store = store
        self.cache = cache
        self.backend = store.backend

    def retrieve(
        self, query: str, ephemeral: list[ContextItem], metadata_filter: dict | None = None
    ) -> list[ContextItem]:
        # 1) durable recall from the persistent index
        durable = self.store.search(query, self.cfg.recall_k, metadata_filter)

        # 2) ephemeral items: embed inline (small N) and score against the query
        ephemeral = self._score_ephemeral(query, ephemeral)

        # keep the strongest ephemeral items so they don't drown the recall set
        ephemeral.sort(key=lambda it: it.similarity, reverse=True)
        ephemeral = ephemeral[: self.cfg.ephemeral_k]

        candidates = durable + ephemeral
        # drop obvious non-matches early
        candidates = [c for c in candidates if c.similarity >= self.cfg.min_similarity]
        return candidates

    def _score_ephemeral(self, query: str, items: list[ContextItem]) -> list[ContextItem]:
        to_embed = [it for it in items if it.embedding is None]
        if to_embed:
            vecs = self.embedder.encode([it.text for it in to_embed])
            for it, v in zip(to_embed, vecs):
                it.embedding = v
        if not items:
            return []
        qvec = self.embedder.encode_one(query)
        for it in items:
            it.similarity = float(it.embedding @ qvec)
        return items

    def hybrid_scores(self, query: str, items: list[ContextItem]) -> dict[str, float]:
        """BM25 lexical score per item id — the keyword half of hybrid search."""
        docs = [_tokens(it.text) for it in items]
        n = len(docs)
        if n == 0:
            return {}
        avgdl = sum(len(d) for d in docs) / n
        df: Counter = Counter()
        for d in docs:
            for term in set(d):
                df[term] += 1
        q_terms = _tokens(query)
        k1, b = 1.5, 0.75
        out: dict[str, float] = {}
        for it, d in zip(items, docs):
            tf = Counter(d)
            dl = len(d)
            score = 0.0
            for term in q_terms:
                if term not in tf:
                    continue
                idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
                denom = tf[term] + k1 * (1 - b + b * dl / avgdl) if avgdl else 1.0
                score += idf * (tf[term] * (k1 + 1)) / denom
            out[it.item_id] = score
        mx = max(out.values(), default=0.0)
        if mx > 0:
            out = {kk: vv / mx for kk, vv in out.items()}
        return out
