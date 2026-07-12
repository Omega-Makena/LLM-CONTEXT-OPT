"""Tests for Batch 5: faithfulness eval, PII/audit, observability, learned weights."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contextx import Config, ContextEngine  # noqa: E402
from contextx.embeddings import Embedder  # noqa: E402
from contextx.eval import GOLDEN_CORPUS, GOLDEN_QUERIES  # noqa: E402
from contextx.eval.faithfulness import FaithfulnessScorer  # noqa: E402
from contextx.observability import Trace, estimate_cost  # noqa: E402
from contextx.security import AuditLog, redact_pii  # noqa: E402
from contextx.tune import FeedbackStore, tune_weights  # noqa: E402


# --- #13 security ---------------------------------------------------------
def test_redact_pii():
    text = "Reach bob@acme.com, call 555-123-4567, SSN 123-45-6789, ip 10.0.0.1"
    red, counts = redact_pii(text)
    assert "bob@acme.com" not in red and "[REDACTED_EMAIL]" in red
    assert counts["EMAIL"] == 1 and counts["SSN"] == 1
    assert "123-45-6789" not in red


def test_audit_log(tmp_path):
    log = AuditLog(str(tmp_path / "audit.jsonl"))
    log.record({"request_id": "abc", "query": "hi", "tenant_id": "t"})
    tail = log.tail()
    assert tail[-1]["request_id"] == "abc" and "ts" in tail[-1]


def test_pipeline_redacts_and_audits(tmp_path):
    cfg = Config(index_dir=str(tmp_path / "idx"), memory_db_path=str(tmp_path / "m.db"),
                 redact_pii=True, audit_log_path=str(tmp_path / "audit.jsonl"))
    from contextx.types import Document
    engine = ContextEngine(config=cfg)
    engine.ingest([Document(text="Contact the vendor at vendor@example.com for details.",
                            doc_id="v")])
    res = engine.run(__import__("contextx").Request(
        user_message="how do I contact the vendor?",
        max_context_tokens=3000, reserve_output_tokens=800))
    # PII scrubbed from the prompt; audit trail written
    assert "vendor@example.com" not in res.prompt.user
    assert engine.audit.tail()[-1]["query"].startswith("how do I contact")


# --- #10 faithfulness -----------------------------------------------------
def test_faithfulness_grounded_vs_hallucinated():
    scorer = FaithfulnessScorer(Embedder(), threshold=0.5)
    sources = ["PostgreSQL uses MVCC for concurrency control.",
               "Refresh tokens are long-lived credentials."]
    grounded = scorer.score("PostgreSQL uses MVCC for concurrency control.", sources)
    hallucinated = scorer.score("The moon is made of green cheese and orbits Jupiter.",
                                sources)
    assert grounded.groundedness > hallucinated.groundedness
    assert grounded.groundedness >= 0.5


def test_faithfulness_no_sources_is_zero():
    scorer = FaithfulnessScorer(Embedder())
    r = scorer.score("Some confident claim about things.", [])
    assert r.groundedness == 0.0


# --- #14 observability ----------------------------------------------------
def test_estimate_cost_and_caching():
    full = estimate_cost("claude-sonnet-5", 1000, 500)
    cached = estimate_cost("claude-sonnet-5", 1000, 500, cache_read_tokens=900)
    assert full > 0 and cached < full


def test_trace_log_line_is_json():
    t = Trace()
    with t.stage("x", 3) as rec:
        rec.items_out = 2
    t.metrics["cost_usd"] = 0.01
    obj = json.loads(t.log_line())
    assert obj["request_id"] == t.request_id
    assert "x" in obj["stages"]


# --- #12 learned weights --------------------------------------------------
def test_tune_weights_not_worse_than_baseline(tmp_path):
    cfg = Config(index_dir=str(tmp_path / "idx"), memory_db_path=str(tmp_path / "m.db"))
    engine = ContextEngine(config=cfg)
    engine.ingest(GOLDEN_CORPUS)
    res = tune_weights(engine, GOLDEN_QUERIES, k=5, trials=80)
    assert res.ndcg >= res.baseline_ndcg  # search includes the baseline
    assert abs(sum(res.weights.values()) - 1.0) < 1e-6


def test_feedback_store(tmp_path):
    fb = FeedbackStore(str(tmp_path / "fb.db"))
    fb.record("what is redis?", ["redis"], rating=1)
    ex = fb.examples()
    assert len(ex) == 1 and ex[0].relevant == ["redis"]
