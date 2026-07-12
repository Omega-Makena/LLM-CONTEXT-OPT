"""Stage 11 — Feedback & Observability.

A `Trace` is threaded through the whole pipeline. Every stage records how many
items it saw, how many it kept, how long it took, and any notable numbers. Each
trace carries a `request_id` for correlation, can estimate the dollar cost of the
LLM call, and can emit a structured (JSON) log line for ingestion by a real log
pipeline — the difference between a dev print and production observability.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field

# USD per 1M tokens (input, output). Cache reads are billed at ~10% of input.
# Update as pricing changes; unknown models fall back to a mid estimate.
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-4-8": (15.0, 75.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (0.80, 4.0),
    "claude-fable-5": (3.0, 15.0),
}
_DEFAULT_PRICE = (3.0, 15.0)


def estimate_cost(
    model: str, input_tokens: int, output_tokens: int,
    cache_read_tokens: int = 0, cache_write_tokens: int = 0,
) -> float:
    """Estimated USD for one call. Cache reads ~0.1x input, writes ~1.25x."""
    p_in, p_out = MODEL_PRICES.get(model, _DEFAULT_PRICE)
    fresh_in = max(0, input_tokens - cache_read_tokens - cache_write_tokens)
    cost = (
        fresh_in * p_in
        + cache_write_tokens * p_in * 1.25
        + cache_read_tokens * p_in * 0.10
        + output_tokens * p_out
    ) / 1_000_000
    return round(cost, 6)


@dataclass
class StageRecord:
    name: str
    ms: float = 0.0
    items_in: int = 0
    items_out: int = 0
    notes: dict = field(default_factory=dict)


@dataclass
class Trace:
    stages: list[StageRecord] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    @contextmanager
    def stage(self, name: str, items_in: int = 0):
        rec = StageRecord(name=name, items_in=items_in)
        start = time.perf_counter()
        try:
            yield rec
        finally:
            rec.ms = (time.perf_counter() - start) * 1000.0
            self.stages.append(rec)

    def to_log(self) -> dict:
        """Structured record for a JSON log pipeline (one line per request)."""
        return {
            "request_id": self.request_id,
            "total_ms": round(sum(s.ms for s in self.stages), 1),
            "stages": {s.name: {"in": s.items_in, "out": s.items_out,
                                "ms": round(s.ms, 1)} for s in self.stages},
            "metrics": self.metrics,
        }

    def log_line(self) -> str:
        return json.dumps(self.to_log(), default=str)

    def report(self) -> str:
        lines = ["", "=" * 68, f"CONTEXTX PIPELINE TRACE  ({self.request_id})", "=" * 68]
        lines.append(f"{'stage':<26}{'in':>5}{'out':>6}{'ms':>9}   notes")
        lines.append("-" * 68)
        for s in self.stages:
            note = ", ".join(f"{k}={v}" for k, v in s.notes.items())
            lines.append(
                f"{s.name:<26}{s.items_in:>5}{s.items_out:>6}{s.ms:>9.1f}   {note}"
            )
        lines.append("-" * 68)
        total_ms = sum(s.ms for s in self.stages)
        lines.append(f"{'TOTAL':<26}{'':>5}{'':>6}{total_ms:>9.1f}")
        if self.metrics:
            lines.append("-" * 68)
            for k, v in self.metrics.items():
                lines.append(f"  {k}: {v}")
        lines.append("=" * 68)
        return "\n".join(lines)
