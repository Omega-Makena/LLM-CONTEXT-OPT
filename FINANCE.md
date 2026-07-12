# Finance domain pack

Tunes the generic engine for financial retrieval across **capital markets,
payments, lending, and compliance/KYC-AML**. Public, generic financial knowledge
— load your proprietary corpus/labels separately (they never enter this repo).

```python
from contextx import ContextEngine
from contextx.domains.finance import finance_config, FINANCE_SYSTEM_PROMPT
from contextx.domains.finance_data import FINANCE_CORPUS

engine = ContextEngine(config=finance_config(), system_prompt=FINANCE_SYSTEM_PROMPT)
engine.ingest(FINANCE_CORPUS)                 # or your own filings/transcripts/docs
result = engine.run(Request(user_message="difference between a CUSIP and an ISIN?"))
```

Run the demo:  `python examples/finance_demo.py`

## What it changes vs. the generic defaults

| Concern | Finance tuning |
|---|---|
| **Freshness** | higher recency weight, ~90-day half-life — filings/quotes go stale |
| **Exact terms** | higher BM25 weight; the **hybrid FTS5 lexical channel** makes tickers, CUSIP, ISIN, routing numbers exact-match retrievable |
| **Numbers** | system prompt forbids silent rounding/estimates; demands as-of dates |
| **Guardrails** | "no personalized investment/legal/tax advice"; cite `[n]` or abstain |
| **Safety** | PII **+ financial identifiers** redacted (IBAN, SWIFT/BIC, routing, account #); injection scan on; audit log for compliance |
| **Entities** | `extract_entities` tags ticker / CUSIP / ISIN for metadata filtering |

## Measured on the finance eval set

41-doc corpus, **28 lexical-trap queries** (CUSIP vs ISIN, KYC vs AML, IBAN vs
SWIFT vs routing, EPS vs P/E, DTI vs LTV, 10-K vs 10-Q, SAR vs CTR):

```
FINANCE (k=5)   recall@5          MRR               nDCG@5
random          0.04 [0.00,0.11]  0.08 [0.04,0.15]  0.04
bi-encoder      1.00 [1.00,1.00]  0.96 [0.91,1.00]  0.97
+reranker       1.00 [1.00,1.00]  1.00 [1.00,1.00]  1.00
null (shuffled) 0.14 [0.04,0.29]  0.08 [0.03,0.17]  0.08
```

- crushes the random floor; **null control collapses** → the metric is valid.
- the **reranker fixes the trap-ordering** the bi-encoder gets wrong (MRR
  0.96 → 1.00), which is exactly the CUSIP/ISIN-style near-synonym confusion.

## Honest caveats

- This is a **generic starter benchmark**, not Innova's data. The saturation
  (recall@5 = 1.0) means it's not hard enough to separate methods on the
  aggregate — the value shows on real, ambiguous queries. Point the harness at
  your labeled data (`contextx.eval.load_jsonl`) to get numbers that matter.
- Financial-identifier regexes are a **redaction floor**, not a DLP guarantee.
- Weight tuning is a no-op here because the defaults already hit nDCG 1.0 on this
  easy set; it earns its keep on harder/real corpora.
