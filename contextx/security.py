"""Security & privacy — PII redaction and an audit log (#13).

  * `redact_pii` scrubs emails, phones, SSNs, credit-card numbers, and IPs from
    text. Applied (optionally) to retrieved context before it reaches the model,
    so third-party documents don't leak PII into the prompt or the LLM provider.
  * `AuditLog` is an append-only JSONL record of every request: who asked what,
    which chunks were retrieved, and what was sent — the provenance trail needed
    for compliance and incident debugging.

Regex PII detection is necessarily imperfect (a floor, not a guarantee); pair it
with a real DLP service for production.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("PHONE", re.compile(r"\b(?:\+?\d{1,2}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b")),
    ("IP", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
]


def redact_pii(text: str) -> tuple[str, dict[str, int]]:
    """Return (redacted_text, {type: count})."""
    counts: dict[str, int] = {}
    for label, pat in _PATTERNS:
        n = 0

        def _sub(_m, _label=label):
            nonlocal n
            n += 1
            return f"[REDACTED_{_label}]"

        text = pat.sub(_sub, text)
        if n:
            counts[label] = n
    return text, counts


class AuditLog:
    """Append-only JSONL audit trail. Thread-safe."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(self, entry: dict) -> None:
        entry = {"ts": time.time(), **entry}
        line = json.dumps(entry, default=str)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def tail(self, n: int = 20) -> list[dict]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return [json.loads(x) for x in lines[-n:]]
