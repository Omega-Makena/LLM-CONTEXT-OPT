"""Finance domain demo: ingest a financial corpus, answer with citations, and
measure + tune retrieval on the finance eval set.

Run:  python examples/finance_demo.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contextx import ContextEngine, Request  # noqa: E402
from contextx.domains.finance import (  # noqa: E402
    FINANCE_SYSTEM_PROMPT, extract_entities, finance_config, redact_financial,
)
from contextx.domains.finance_data import FINANCE_CORPUS, FINANCE_QUERIES  # noqa: E402
from contextx.eval import RetrievalEvaluator  # noqa: E402
from contextx.eval.report import format_results  # noqa: E402
from contextx.tune import tune_weights  # noqa: E402

K = 5


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="contextx_fin_"))
    cfg = finance_config(index_dir=str(tmp / "idx"), memory_db_path=str(tmp / "m.db"),
                         max_context_tokens=4000, reserve_output_tokens=1000)
    engine = ContextEngine(config=cfg, system_prompt=FINANCE_SYSTEM_PROMPT)
    n = engine.ingest(FINANCE_CORPUS)
    print(f"[ingest] {n} finance chunks  (recency_weight={cfg.w_recency}, "
          f"redact_pii={cfg.redact_pii})")

    # sensitive-data + entity handling
    red, counts = redact_financial(
        "Wire to IBAN GB29NWBK60161331926819 via SWIFT NWBKGB2L, routing 021000021.")
    print(f"[redaction] {counts} -> {red}")
    print(f"[entities] {extract_entities('Buy 100 AAPL and MSFT; ISIN US0378331005')}")

    # a finance query: citations + abstention
    q = Request(user_message="what is the difference between a CUSIP and an ISIN?",
                max_context_tokens=4000, reserve_output_tokens=1000)
    res = engine.run(q)
    print("\n" + "#" * 68 + f"\n# ANSWER  [backend: {res.llm.backend}, "
          f"low_confidence={res.low_confidence}]\n" + "#" * 68)
    print(res.answer)
    print("sources:", [(s["n"], s["doc_id"]) for s in res.sources])

    # off-topic query should abstain
    off = engine.run(Request(user_message="what is the boiling point of helium?",
                             max_context_tokens=4000, reserve_output_tokens=1000))
    print(f"\n[abstention] off-topic query -> low_confidence={off.low_confidence}")

    # retrieval eval on the finance set
    print("\n" + "=" * 60 + "\nFINANCE RETRIEVAL EVAL")
    ev = RetrievalEvaluator(FINANCE_CORPUS, FINANCE_QUERIES, k=K, recall_k=15,
                            index_dir=str(tmp / "eval"))
    print(f"[backends] {ev.embed_backend} / {ev.rerank_backend}")
    print(format_results(ev.evaluate_all(), title=f"FINANCE (k={K})"))

    # tune the rank weights on the finance labels
    tuned = tune_weights(engine, FINANCE_QUERIES, k=K, trials=200)
    print(f"\n[tune] nDCG@{K}: baseline {tuned.baseline_ndcg} -> tuned {tuned.ndcg}")
    print(f"[tune] weights: {tuned.weights}")

    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
