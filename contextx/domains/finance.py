"""Finance domain pack — tune contextx for financial retrieval.

Covers capital markets, payments, lending, and compliance/KYC-AML. Public,
generic financial knowledge only — load your proprietary corpus/labels
separately.

What it changes vs the generic defaults:
  * recency matters more (filings/quotes go stale) -> higher recency weight,
    shorter half-life.
  * safety on by default: PII/sensitive redaction + injection scan, and a
    system prompt that forbids investment advice and demands exact figures +
    citations or abstention.
  * financial sensitive-data patterns (IBAN, SWIFT/BIC, routing, account #) on
    top of the base PII set.
  * entity anchors (ticker / CUSIP / ISIN) — the hybrid lexical channel makes
    these exact-match retrievable.
"""

from __future__ import annotations

import re

from ..config import Config

FINANCE_SYSTEM_PROMPT = (
    "You are a financial information assistant answering strictly from retrieved "
    "context.\n"
    "Rules:\n"
    "1. Use ONLY the provided context; cite each fact inline as [n].\n"
    "2. Be exact with numbers, units, currencies, percentages, and as-of dates. "
    "Never round or estimate silently.\n"
    "3. If the context does not contain the answer, say you do not have it — do "
    "not guess.\n"
    "4. Provide factual information only. Do NOT give personalized investment, "
    "legal, tax, or trading advice.\n"
    "5. Flag when figures may be stale or conflict across sources."
)

# Financial sensitive-data patterns (imperfect regex floor; pair with real DLP).
FINANCE_PII_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")),
    ("SWIFT_BIC", re.compile(r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b")),
    ("ROUTING", re.compile(r"\b\d{9}\b")),
    ("ACCOUNT", re.compile(r"\b\d{8,17}\b")),
]

# Entity anchors — tag docs/queries so the lexical channel can exact-match them.
_TICKER = re.compile(r"\b[A-Z]{1,5}(?:\.[A-Z])?\b")
_CUSIP = re.compile(r"\b[0-9A-Z]{9}\b")
_ISIN = re.compile(r"\b[A-Z]{2}[0-9A-Z]{9}\d\b")

# Uppercase words to keep the ticker heuristic from over-firing: English stop
# words + common financial acronyms that are not tickers.
_STOP_UPPER = {
    "A", "I", "THE", "AND", "OR", "OF", "TO", "IN", "IS", "IT", "US", "EU", "AI",
    # financial acronyms
    "AML", "KYC", "APR", "EPS", "ETF", "IBAN", "SWIFT", "BIC", "ISIN", "CUSIP",
    "SAR", "CTR", "PEP", "SEC", "DTI", "LTV", "ACH", "PCI", "DSS", "MIFID",
    "OFAC", "FICO", "YTM", "T1", "SA", "IPO", "NAV", "AUM", "GDPR", "PII",
}


def finance_config(**overrides) -> Config:
    """Config preset tuned for finance. Override any field via kwargs."""
    cfg = Config(
        w_rerank=0.50,
        w_similarity=0.15,
        w_bm25=0.15,          # exact terms (tickers, codes) matter in finance
        w_recency=0.15,       # freshness matters more than the generic default
        w_importance=0.03,
        w_conversation=0.01,
        w_preference=0.01,
        recency_half_life_s=90 * 24 * 3600.0,  # ~a quarter
        enable_hybrid=True,
        injection_scan=True,
        redact_pii=True,
        abstain_below=-2.0,   # decline sooner rather than guess a number
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def redact_financial(text: str):
    """Redact base PII + financial identifiers. Returns (text, counts)."""
    from ..security import redact_pii

    return redact_pii(text, extra_patterns=FINANCE_PII_PATTERNS)


def extract_entities(text: str) -> dict[str, list[str]]:
    """Best-effort ticker / CUSIP / ISIN extraction for metadata tagging."""
    isins = _ISIN.findall(text)
    cusips = [c for c in _CUSIP.findall(text) if c not in isins]
    tickers = [
        t for t in _TICKER.findall(text)
        if t not in _STOP_UPPER and t not in cusips and len(t) <= 5
    ]
    out: dict[str, list[str]] = {}
    if tickers:
        out["tickers"] = sorted(set(tickers))
    if cusips:
        out["cusips"] = sorted(set(cusips))
    if isins:
        out["isins"] = sorted(set(isins))
    return out
