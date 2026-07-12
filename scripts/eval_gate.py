"""CI regression gate (#11) — fail the build if retrieval quality drops.

Runs the retrieval eval on the golden set and asserts:
  * +reranker recall@k clears a floor well above random,
  * the shuffled-label null control collapses (metric is valid).

Thresholds are conservative so they hold on CI's fallback backends (hash
embeddings, identity reranker) — they catch *regressions*, not absolute quality.
Exit code 1 on failure so CI blocks.

    python scripts/eval_gate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contextx.eval import GOLDEN_CORPUS, GOLDEN_QUERIES, RetrievalEvaluator  # noqa: E402

K = 5
RECALL_FLOOR = 0.60      # +reranker recall@5 must exceed this
NULL_CEILING = 0.30      # shuffled-gold recall@5 must stay below this


def main() -> int:
    ev = RetrievalEvaluator(GOLDEN_CORPUS, GOLDEN_QUERIES, k=K)
    res = ev.evaluate_all()
    recall = res["+reranker (full)"][f"recall@{K}"][0]
    null = res["null (shuffled gold)"][f"recall@{K}"][0]
    random_floor = res["random"][f"recall@{K}"][0]

    print(f"[eval-gate] backend={ev.embed_backend}/{ev.rerank_backend}")
    print(f"[eval-gate] recall@{K}: reranker={recall:.2f}  random={random_floor:.2f}  null={null:.2f}")

    ok = True
    if recall < RECALL_FLOOR:
        print(f"FAIL: recall@{K} {recall:.2f} < floor {RECALL_FLOOR}")
        ok = False
    if null > NULL_CEILING:
        print(f"FAIL: null recall@{K} {null:.2f} > ceiling {NULL_CEILING} (metric invalid)")
        ok = False
    if recall <= random_floor:
        print("FAIL: reranker no better than random")
        ok = False

    print("[eval-gate] PASS" if ok else "[eval-gate] FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
