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

def luhn_ok(s: str) -> bool:
    """Luhn checksum — real card numbers pass; random digit runs almost never do.
    Prevents redacting arbitrary long numbers (timestamps, quantities, financial
    figures) as if they were credit cards."""
    digits = [int(c) for c in s if c.isdigit()]
    if len(digits) < 13:
        return False
    total, parity = 0, len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# (label, pattern, validator|None). A pattern may expose a named group `v` to
# redact only that span (keeping a keyword prefix); otherwise the whole match
# is redacted. Validators gate redaction (e.g. Luhn for cards).
_PATTERNS: list[tuple] = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), None),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), None),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b"), luhn_ok),
    ("PHONE", re.compile(r"\b(?:\+?\d{1,2}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b"), None),
    ("IP", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), None),
]


def redact_pii(
    text: str, extra_patterns: list[tuple] | None = None
) -> tuple[str, dict[str, int]]:
    """Return (redacted_text, {type: count}). `extra_patterns` lets a domain add
    its own sensitive identifiers (e.g. finance: IBAN/SWIFT/routing/account)."""
    counts: dict[str, int] = {}
    for entry in [*(extra_patterns or []), *_PATTERNS]:
        label, pat = entry[0], entry[1]
        validator = entry[2] if len(entry) > 2 else None
        n = 0

        def _sub(m, _label=label, _val=validator):
            nonlocal n
            has_group = "v" in m.re.groupindex
            target = m.group("v") if has_group else m.group(0)
            if _val is not None and not _val(target):
                return m.group(0)  # failed validation — leave untouched
            n += 1
            token = f"[REDACTED_{_label}]"
            return m.group(0).replace(target, token) if has_group else token

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
