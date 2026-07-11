"""Stage 11 — Feedback & Observability.

A `Trace` is threaded through the whole pipeline. Every stage records how many
items it saw, how many it kept, how long it took, and any notable numbers
(tokens saved, cache hits). At the end we can print a legible per-stage report,
which is the single best way to *see* the engine working.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field


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

    @contextmanager
    def stage(self, name: str, items_in: int = 0):
        rec = StageRecord(name=name, items_in=items_in)
        start = time.perf_counter()
        try:
            yield rec
        finally:
            rec.ms = (time.perf_counter() - start) * 1000.0
            self.stages.append(rec)

    def report(self) -> str:
        lines = ["", "=" * 68, "CONTEXTX PIPELINE TRACE", "=" * 68]
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
