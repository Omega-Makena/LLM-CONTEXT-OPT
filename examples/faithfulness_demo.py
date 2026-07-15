"""Faithfulness / groundedness demo.

Runs a query, then scores whether the answer is supported by the retrieved
sources — with the offline embedding-overlap proxy, and with the LLM judge when
a real backend is configured. Loads .env for a key.

Run:  python examples/faithfulness_demo.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
env = REPO / ".env"
if env.exists():
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            if v.strip():
                os.environ.setdefault(k.strip(), v.strip())

sys.path.insert(0, str(REPO))
from contextx import Config, ContextEngine, Document, Request  # noqa: E402
from contextx.eval.faithfulness import FaithfulnessScorer  # noqa: E402


def main() -> None:
    d = tempfile.mkdtemp()
    engine = ContextEngine(config=Config(index_dir=d + "/i", memory_db_path=d + "/m.db"))
    engine.ingest([
        Document(text="Acme Corp FY2024 revenue was $5.0 million, up 12% year over "
                      "year. Net income was $0.8 million.", doc_id="fin"),
    ])
    res = engine.run(Request(user_message="What was Acme's FY2024 revenue and growth?"))
    sources = [s["preview"] for s in res.sources]

    print(f"backend: {res.llm.backend}")
    print(f"answer: {res.answer}\n")

    scorer = FaithfulnessScorer(engine.embedder)
    off = scorer.score(res.answer, sources)
    print(f"[offline proxy] groundedness={off.groundedness:.2f} "
          f"({off.supported}/{off.total} claims)")
    if off.unsupported_claims:
        print("  unsupported:", off.unsupported_claims)

    if res.llm.backend != "mock":
        j = scorer.score(res.answer, sources, judge=engine.llm)
        print(f"[llm judge]     groundedness={j.groundedness:.2f} "
              f"({j.supported}/{j.total} claims)")
        if j.unsupported_claims:
            print("  unsupported:", j.unsupported_claims)
    else:
        print("[llm judge]     skipped (mock backend — set a key for the judge)")


if __name__ == "__main__":
    main()
