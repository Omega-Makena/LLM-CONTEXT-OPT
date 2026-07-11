"""Run the retrieval evaluation and print the results tables.

Run:  python examples/run_eval.py

Two benchmarks over the same corpus:
  * EASY (topic-separated queries) — everything saturates; establishes the null
    control and the embedding ablation.
  * HARD (lexical traps + multi-hop) — a decoy outranks the true answer on
    surface similarity, so the bi-encoder misses and the cross-encoder reranker
    has room to prove its value. This is where rerank lift becomes measurable.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contextx.eval import GOLDEN_CORPUS, GOLDEN_QUERIES, RetrievalEvaluator  # noqa: E402
from contextx.eval.dataset import HARD_QUERIES  # noqa: E402
from contextx.eval.report import format_results  # noqa: E402

K = 5


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="contextx_eval_"))

    # ---------- EASY ----------
    print(f"\n### EASY benchmark: {len(GOLDEN_QUERIES)} topic-separated queries, k={K}")
    ev = RetrievalEvaluator(GOLDEN_CORPUS, GOLDEN_QUERIES, k=K, index_dir=str(tmp / "easy"))
    print(f"[backends] embeddings={ev.embed_backend}  reranker={ev.rerank_backend}")
    easy = ev.evaluate_all()
    print(format_results(easy, title=f"EASY (k={K})"))

    # ---------- HARD ----------
    print(f"\n\n### HARD benchmark: {len(HARD_QUERIES)} lexical-trap / multi-hop queries, "
          f"k={K}, recall_k=15")
    evh = RetrievalEvaluator(GOLDEN_CORPUS, HARD_QUERIES, k=K, recall_k=15,
                             index_dir=str(tmp / "hard"))
    hard = evh.evaluate_all()
    print(format_results(hard, title=f"HARD (k={K})"))

    diag = evh.rank_diagnostics()
    print(f"\nmean rank of first correct answer: bi-encoder {diag['bi-encoder']:.2f} "
          f"-> +reranker {diag['+reranker']:.2f}  (lower is better)")

    print("\nPAIRED significance  (+reranker vs bi-encoder, same queries):")
    print(f"  {'metric':<12}{'delta':>8}{'95% CI':>18}{'p(1-sided)':>12}{'win rate':>10}")
    sig = evh.paired_significance()
    for m, s in sig.items():
        star = " *" if s["p_one_sided"] < 0.05 else ""
        ci = f"[{s['lo']:+.3f},{s['hi']:+.3f}]"
        print(f"  {m:<12}{s['delta']:>+8.3f}{ci:>18}{s['p_one_sided']:>12.3f}"
              f"{s['win_rate']:>9.0%}{star}")
    print(f"  (n={sig[next(iter(sig))]['n']} queries;  * = significant at p<0.05)")

    # stratified: the reranker can only help where the bi-encoder isn't already
    # perfect. Condition on those queries and report the recovery rate + a paired
    # test on that stratum.
    from contextx.eval.metrics import paired_delta  # noqa: E402

    bi, rr = evh.per_query("mrr")
    hard_idx = [i for i, x in enumerate(bi) if x < 1.0]
    n_total, n_hard = len(bi), len(hard_idx)
    recovered = sum(1 for i in hard_idx if rr[i] > bi[i])
    degraded = sum(1 for i in hard_idx if rr[i] < bi[i])
    print("\nStratified on the queries the bi-encoder gets WRONG (RR < 1):")
    print(f"  bi-encoder already perfect on {n_total - n_hard}/{n_total}; "
          f"{n_hard} queries are hard")
    print(f"  on those {n_hard}: reranker improved {recovered}, degraded {degraded}, "
          f"tied {n_hard - recovered - degraded}")
    if n_hard:
        s = paired_delta([bi[i] for i in hard_idx], [rr[i] for i in hard_idx])
        star = " *" if s["p_one_sided"] < 0.05 else ""
        print(f"  paired MRR delta on hard stratum: {s['delta']:+.3f} "
              f"[{s['lo']:+.3f},{s['hi']:+.3f}]  p={s['p_one_sided']:.3f}{star}")

    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
