"""Stage 4 — Filtering & Deduplication.

Ranking is not enough: the top list still contains near-duplicates, stale data,
low-confidence retrievals, and occasionally *contradictions*. This stage prunes
them, keeping the highest-ranked representative of each cluster.

  * dedup:        drop items whose embedding cosine to an already-kept item
                  exceeds `dup_threshold` (semantic near-duplicate).
  * low signal:   drop items whose similarity < `min_similarity`.
  * expired:      drop items with metadata['expires_at'] in the past.
  * contradiction: items sharing metadata['fact_key'] with different
                  metadata['fact_value'] conflict; a policy picks the winner
                  (default: highest score, tie-broken by recency).

Contradiction handling here is a deliberately simple, honest heuristic — real
conflict detection is an open problem; this catches the structured case.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .types import ContextItem


@dataclass
class FilterStats:
    duplicates: int = 0
    low_signal: int = 0
    expired: int = 0
    contradictions: int = 0
    conflicts_resolved: list[str] = field(default_factory=list)


class Filter:
    def __init__(self, dup_threshold: float = 0.92, min_similarity: float = 0.0) -> None:
        self.dup_threshold = dup_threshold
        self.min_similarity = min_similarity

    def apply(
        self, items: list[ContextItem]
    ) -> tuple[list[ContextItem], FilterStats]:
        stats = FilterStats()
        now = time.time()

        # 1) drop expired + low-signal (items assumed already rank-sorted)
        survivors: list[ContextItem] = []
        for it in items:
            exp = it.metadata.get("expires_at")
            if exp is not None and exp < now:
                stats.expired += 1
                continue
            if it.similarity < self.min_similarity:
                stats.low_signal += 1
                continue
            survivors.append(it)

        # 2) contradiction resolution on structured facts
        survivors = self._resolve_conflicts(survivors, stats)

        # 3) semantic dedup — greedily keep the best, drop near-duplicates
        kept: list[ContextItem] = []
        kept_vecs: list[np.ndarray] = []
        for it in survivors:
            if it.embedding is not None and kept_vecs:
                sims = np.array([float(v @ it.embedding) for v in kept_vecs])
                if sims.max() >= self.dup_threshold:
                    stats.duplicates += 1
                    continue
            kept.append(it)
            if it.embedding is not None:
                kept_vecs.append(it.embedding)

        return kept, stats

    def _resolve_conflicts(
        self, items: list[ContextItem], stats: FilterStats
    ) -> list[ContextItem]:
        by_key: dict[str, list[ContextItem]] = {}
        passthrough: list[ContextItem] = []
        for it in items:
            key = it.metadata.get("fact_key")
            if key is None:
                passthrough.append(it)
            else:
                by_key.setdefault(key, []).append(it)

        winners: list[ContextItem] = []
        for key, group in by_key.items():
            values = {it.metadata.get("fact_value") for it in group}
            if len(values) > 1:
                stats.contradictions += 1
                # policy: highest score wins, then most recent
                winner = max(group, key=lambda it: (it.score, it.timestamp))
                stats.conflicts_resolved.append(
                    f"{key}: kept '{winner.metadata.get('fact_value')}'"
                )
                winners.append(winner)
            else:
                winners.extend(group)

        # preserve original (rank) ordering
        keep_ids = {id(x) for x in passthrough + winners}
        return [it for it in items if id(it) in keep_ids]
