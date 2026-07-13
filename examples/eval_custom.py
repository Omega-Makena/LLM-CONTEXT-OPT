"""Evaluate + tune retrieval on YOUR data — drop in JSONL, get domain numbers.

    python examples/eval_custom.py --corpus mycorpus.jsonl --queries myqueries.jsonl

Formats (one JSON object per line):
    corpus.jsonl:   {"text": "...", "doc_id": "acme-10k-2024", "metadata": {...}}
    queries.jsonl:  {"query": "what was revenue?", "relevant": ["acme-10k-2024"]}

With no args it runs on the bundled sample (examples/data/*.jsonl) so you can see
the format and a working run immediately. Prints IR metrics with bootstrap CIs
(random / bi-encoder / +reranker / null control) and the nDCG-tuned rank weights.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contextx.eval import RetrievalEvaluator, load_corpus_jsonl, load_jsonl  # noqa: E402
from contextx.eval.report import format_results  # noqa: E402
from contextx.tune import tune_weights  # noqa: E402

_HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate + tune retrieval on your JSONL data.")
    ap.add_argument("--corpus", default=str(_HERE / "data" / "sample_corpus.jsonl"))
    ap.add_argument("--queries", default=str(_HERE / "data" / "sample_queries.jsonl"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--recall-k", type=int, default=15)
    ap.add_argument("--trials", type=int, default=300)
    args = ap.parse_args()

    corpus = load_corpus_jsonl(args.corpus)
    queries = load_jsonl(args.queries)
    print(f"corpus: {len(corpus)} docs ({args.corpus})")
    print(f"queries: {len(queries)} labeled ({args.queries})\n")

    ev = RetrievalEvaluator(corpus, queries, k=args.k, recall_k=args.recall_k)
    print(f"[backends] embeddings={ev.embed_backend}  reranker={ev.rerank_backend}\n")
    print(format_results(ev.evaluate_all(), title=f"RETRIEVAL EVAL (k={args.k})"))

    tuned = tune_weights(ev.engine, queries, k=args.k, trials=args.trials)
    print(f"\n[tune] nDCG@{args.k}: baseline {tuned.baseline_ndcg} -> tuned {tuned.ndcg}")
    print(f"[tune] weights: {tuned.weights}")
    print("\nApply with: Config(w_rerank=..., w_similarity=..., w_bm25=...)")


if __name__ == "__main__":
    main()
