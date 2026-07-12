"""Answer-quality eval — faithfulness / groundedness (#10).

Retrieval metrics say nothing about whether the *answer* is supported by the
context (hallucination). This scores groundedness: split the answer into claims
(sentences) and check each is backed by at least one retrieved source.

  * default (offline): embedding overlap — a claim is "supported" if its max
    cosine to any source chunk clears `threshold`. No API key needed.
  * optional (`judge=llm`): an LLM judges each claim as SUPPORTED / UNSUPPORTED —
    higher fidelity, costs calls.

groundedness = fraction of claims supported. Low groundedness with high retrieval
recall points at generation (hallucination); low recall points at retrieval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from ..embeddings import Embedder

_SENT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class FaithfulnessResult:
    groundedness: float
    supported: int
    total: int
    unsupported_claims: list[str] = field(default_factory=list)
    method: str = "embedding-overlap"


class FaithfulnessScorer:
    def __init__(self, embedder: Embedder | None = None, threshold: float = 0.5) -> None:
        self.embedder = embedder or Embedder()
        self.threshold = threshold

    def _claims(self, answer: str) -> list[str]:
        return [s.strip() for s in _SENT.split(answer.strip()) if len(s.strip()) > 8]

    def score(self, answer: str, sources: list[str], judge=None) -> FaithfulnessResult:
        claims = self._claims(answer)
        if not claims:
            return FaithfulnessResult(1.0, 0, 0, method="empty")
        if not sources:
            return FaithfulnessResult(0.0, 0, len(claims), unsupported_claims=claims)

        if judge is not None:
            return self._llm_judge(claims, sources, judge)

        src = self.embedder.encode(sources)
        cl = self.embedder.encode(claims)
        sims = cl @ src.T                      # (claims, sources)
        best = sims.max(axis=1)
        supported_mask = best >= self.threshold
        unsupported = [c for c, ok in zip(claims, supported_mask) if not ok]
        return FaithfulnessResult(
            groundedness=float(supported_mask.mean()),
            supported=int(supported_mask.sum()),
            total=len(claims),
            unsupported_claims=unsupported,
        )

    def _llm_judge(self, claims, sources, judge) -> FaithfulnessResult:
        context = "\n".join(f"- {s}" for s in sources)
        supported = 0
        unsupported: list[str] = []
        for c in claims:
            prompt = (
                f"Context:\n{context}\n\nClaim: {c}\n\n"
                "Is the claim fully supported by the context? Answer only YES or NO."
            )
            verdict = judge(prompt).strip().upper()
            if verdict.startswith("YES"):
                supported += 1
            else:
                unsupported.append(c)
        return FaithfulnessResult(
            groundedness=supported / len(claims),
            supported=supported,
            total=len(claims),
            unsupported_claims=unsupported,
            method="llm-judge",
        )
