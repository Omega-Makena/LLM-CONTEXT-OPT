"""contextx.eval — offline retrieval evaluation.

Turns "looks good" into defensible numbers: a labeled golden set, IR metrics
with bootstrap CIs, and ablations (random floor, bi-encoder vs +reranker, real
vs hash embeddings, shuffled-label null control).

    from contextx.eval import RetrievalEvaluator, GOLDEN_CORPUS, GOLDEN_QUERIES
    from contextx.eval.report import format_results

    ev = RetrievalEvaluator(GOLDEN_CORPUS, GOLDEN_QUERIES, k=5)
    print(format_results(ev.evaluate_all()))
"""

from .dataset import (
    GOLDEN_CORPUS, GOLDEN_QUERIES, EvalExample,
    load_corpus_jsonl, load_jsonl, save_corpus_jsonl, save_jsonl,
)
from .harness import RetrievalEvaluator

__all__ = [
    "RetrievalEvaluator",
    "GOLDEN_CORPUS",
    "GOLDEN_QUERIES",
    "EvalExample",
    "load_jsonl",
    "save_jsonl",
    "load_corpus_jsonl",
    "save_corpus_jsonl",
]
