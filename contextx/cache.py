"""Stage 12 — Cache layer (cross-cutting), production-shaped.

The toy cache was an unbounded dict keyed on exact strings — it leaked memory
and, for LLM responses, almost never hit (prompts are rarely byte-identical).
This version adds:

  * bounded LRU eviction (`cache_max_entries`)
  * per-entry TTL (`cache_ttl_s`)
  * a SEMANTIC response cache: match a new request against prior ones by
    embedding cosine, so paraphrases hit.

Swap the in-memory backing for Redis/diskcache by keeping `get_or_compute`.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Callable

import numpy as np

from .config import Config


class Cache:
    def __init__(self, config: Config | None = None) -> None:
        self.cfg = config or Config()
        self._store: "OrderedDict[str, tuple[float, Any]]" = OrderedDict()
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        # semantic response cache: parallel arrays of (embedding, key)
        self._sem_vecs: list[np.ndarray] = []
        self._sem_keys: list[str] = []

    def get_or_compute(self, namespace: str, key: str, compute: Callable[[], Any]) -> Any:
        full = f"{namespace}:{key}"
        with self._lock:
            hit = self._get(full)
            if hit is not None:
                self.hits += 1
                return hit
            self.misses += 1
        value = compute()  # compute outside the lock
        with self._lock:
            self._put(full, value)
        return value

    def semantic_get_or_compute(
        self, namespace: str, embedding: np.ndarray, key: str, compute: Callable[[], Any]
    ) -> Any:
        """Response-cache lookup by embedding similarity, not exact string."""
        with self._lock:
            if self._sem_vecs:
                mat = np.vstack(self._sem_vecs)
                sims = mat @ embedding
                j = int(np.argmax(sims))
                if sims[j] >= self.cfg.semantic_cache_threshold:
                    hit = self._get(self._sem_keys[j])
                    if hit is not None:
                        self.hits += 1
                        return hit
                    # matched a key whose store entry was evicted/expired —
                    # prune the dead semantic entry instead of leaking it.
                    self._sem_vecs.pop(j)
                    self._sem_keys.pop(j)
            self.misses += 1
        value = compute()
        with self._lock:
            full = f"{namespace}:{key}"
            self._put(full, value)
            self._sem_vecs.append(embedding.astype(np.float32))
            self._sem_keys.append(full)
            # bound the semantic cache like the exact cache (drop oldest)
            while len(self._sem_vecs) > self.cfg.cache_max_entries:
                self._sem_vecs.pop(0)
                self._sem_keys.pop(0)
        return value

    # --- internals --------------------------------------------------------
    def _get(self, full: str) -> Any:
        entry = self._store.get(full)
        if entry is None:
            return None
        ts, value = entry
        if time.time() - ts > self.cfg.cache_ttl_s:
            del self._store[full]
            return None
        self._store.move_to_end(full)  # LRU touch
        return value

    def _put(self, full: str, value: Any) -> None:
        self._store[full] = (time.time(), value)
        self._store.move_to_end(full)
        while len(self._store) > self.cfg.cache_max_entries:
            self._store.popitem(last=False)  # evict LRU

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._sem_vecs.clear()
            self._sem_keys.clear()

    def invalidate_responses(self) -> None:
        """Drop cached LLM responses (exact + semantic). Call when the corpus
        changes, so an edited/deleted document can't serve a stale answer."""
        with self._lock:
            for k in [k for k in self._store if k.startswith("llm:")]:
                del self._store[k]
            self._sem_vecs.clear()
            self._sem_keys.clear()

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
