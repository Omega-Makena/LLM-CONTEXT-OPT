"""Stage 6 — Compression / Summarization.

Large items are shrunk before budgeting so we keep meaning per token. Two modes:

  * extractive (default): keep the sentences most similar to the query, drop the
    rest. Fast, lossless-ish, no model call. Used automatically for any item
    over `max_item_tokens`.
  * abstractive (optional): pass an `llm` callable to rewrite an item into a
    shorter summary. Higher quality, costs a model call; cached by the pipeline.

Only oversized items are touched; short, high-signal items pass through intact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .budget import count_tokens
from .config import Config
from .embeddings import Embedder
from .types import ContextItem

_SENT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class CompressStats:
    items_compressed: int = 0
    tokens_before: int = 0
    tokens_after: int = 0

    @property
    def ratio(self) -> float:
        return self.tokens_after / self.tokens_before if self.tokens_before else 1.0


class Compressor:
    def __init__(self, embedder: Embedder, config: Config | None = None) -> None:
        cfg = config or Config()
        self.embedder = embedder
        self.max_item_tokens = cfg.max_item_tokens
        self.target_ratio = cfg.compress_target_ratio

    def compress(
        self, query: str, items: list[ContextItem], llm=None
    ) -> tuple[list[ContextItem], CompressStats]:
        stats = CompressStats()
        qvec = self.embedder.encode_one(query)
        for it in items:
            toks = count_tokens(it.text)
            if toks <= self.max_item_tokens:
                continue
            stats.tokens_before += toks
            if llm is not None:
                it.text = self._abstractive(it.text, llm)
            else:
                it.text = self._extractive(it.text, qvec)
            it.tokens = count_tokens(it.text)
            it.embedding = None  # invalidate; text changed
            stats.tokens_after += it.tokens
            stats.items_compressed += 1
        return items, stats

    def _extractive(self, text: str, qvec: np.ndarray) -> str:
        sentences = [s.strip() for s in _SENT.split(text) if s.strip()]
        if len(sentences) <= 1:
            return text
        vecs = self.embedder.encode(sentences)
        sims = vecs @ qvec
        keep_n = max(1, int(len(sentences) * self.target_ratio))
        top_idx = sorted(np.argsort(sims)[-keep_n:])  # keep original order
        return " ".join(sentences[i] for i in top_idx)

    def _abstractive(self, text: str, llm) -> str:
        prompt = (
            "Summarize the following context in 2-3 sentences, preserving "
            "concrete facts, names, and numbers:\n\n" + text
        )
        return llm(prompt).strip()
