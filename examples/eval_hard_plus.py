"""Run the difficulty-stratified HARD_PLUS eval: metrics, paired significance,
and the hard-stratum breakdown.

Run:  python examples/eval_hard_plus.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contextx.eval import RetrievalEvaluator  # noqa: E402
from contextx.eval.hard_plus import HARD_PLUS_CORPUS, HARD_PLUS_QUERIES  # noqa: E402
from contextx.eval.report import format_results  # noqa: E402

K = 5


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="contextx_hardplus_"))
    ev = RetrievalEvaluator(HARD_PLUS_CORPUS, HARD_PLUS_QUERIES, k=K, recall_k=15,
                            index_dir=str(tmp))
    print(f"corpus: {len(HARD_PLUS_CORPUS)} docs   queries: {len(HARD_PLUS_QUERIES)}   k={K}")
    print(f"[backends] {ev.embed_backend} / {ev.rerank_backend}\n")
    print(format_results(ev.evaluate_all(), title=f"HARD_PLUS (k={K})"))

    print("\nPAIRED significance (+reranker vs bi-encoder, same queries):")
    sig = ev.paired_significance()
    for m, s in sig.items():
        star = " *" if s["p_one_sided"] < 0.05 else ""
        print(f"  {m:<12} delta {s['delta']:+.3f}  95% CI [{s['lo']:+.3f},{s['hi']:+.3f}]"
              f"  p={s['p_one_sided']:.3f}  win {s['win_rate']:.0%}{star}")

    diag = ev.rank_diagnostics()
    print(f"\nmean rank of first correct answer: bi-encoder {diag['bi-encoder']:.2f} "
          f"-> +reranker {diag['+reranker']:.2f}")
    bi, _ = ev.per_query("mrr")
    hard = sum(1 for x in bi if x < 1.0)
    print(f"bi-encoder imperfect on {hard}/{len(bi)} queries "
          f"({hard / len(bi):.0%} genuinely hard)")

    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
