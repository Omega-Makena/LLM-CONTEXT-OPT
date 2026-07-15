"""Tests for fixed naive assumptions: cache bounding, PII precision, memory dims."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from contextx.cache import Cache  # noqa: E402
from contextx.config import Config  # noqa: E402
from contextx.domains.finance import redact_financial  # noqa: E402
from contextx.embeddings import Embedder  # noqa: E402
from contextx.memory import MemoryManager, MemoryRecord, MemoryType  # noqa: E402
from contextx.security import luhn_ok, redact_pii  # noqa: E402


# --- semantic cache no longer leaks ---------------------------------------
def test_semantic_cache_is_bounded():
    # threshold impossible to hit -> every call inserts; must stay bounded
    c = Cache(Config(cache_max_entries=3, semantic_cache_threshold=9e9))
    for i in range(12):
        v = np.random.RandomState(i).randn(4).astype(np.float32)
        c.semantic_get_or_compute("llm", v, f"k{i}", lambda i=i: f"a{i}")
    assert len(c._sem_vecs) == 3
    assert len(c._sem_keys) == 3


def test_semantic_cache_prunes_dead_match():
    c = Cache(Config(cache_max_entries=100, semantic_cache_threshold=0.9))
    v = np.ones(4, dtype=np.float32)
    c.semantic_get_or_compute("llm", v, "k0", lambda: "answer")
    # evict the underlying store entry but leave the semantic vector
    c._store.clear()
    # a near-identical query matches the vec, finds the store gone, prunes it
    c.semantic_get_or_compute("llm", v, "k1", lambda: "fresh")
    assert "llm:k0" not in c._store
    assert len(c._sem_vecs) == len(c._sem_keys)


# --- PII redaction precision ----------------------------------------------
def test_luhn_only_redacts_real_cards():
    red, counts = redact_pii("card 4242 4242 4242 4242 ok")       # valid Luhn
    assert "4242" not in red and counts.get("CREDIT_CARD") == 1
    red2, counts2 = redact_pii("ref 4242424242424241 ok")          # invalid Luhn
    assert "4242424242424241" in red2 and "CREDIT_CARD" not in counts2


def test_luhn_ok():
    assert luhn_ok("4242424242424242")
    assert not luhn_ok("4242424242424241")


def test_finance_redaction_keeps_plain_figures():
    # a bare financial figure must survive (naive \d{8,17} would have scrubbed it)
    red, counts = redact_financial("Q2 revenue was 123456789 dollars, up 12%.")
    assert "123456789" in red
    assert "ACCOUNT" not in counts and "ROUTING" not in counts
    # but a labeled account/routing number is redacted
    red2, _ = redact_financial("account 12345678, routing 021000021")
    assert "12345678" not in red2 and "021000021" not in red2


# --- small tenant is not starved by post-filter ANN -----------------------
def test_small_tenant_not_starved(tmp_path):
    from contextx.store import VectorStore
    from contextx.types import Document
    s = VectorStore(Embedder(force_fallback=True), Config(index_dir=str(tmp_path / "i")))
    # 60 big-tenant docs all outrank the small tenant's single doc on the query
    s.add_documents([Document(text=f"big markets trading strategy alpha {i}",
                              doc_id=f"b{i}", tenant_id="big") for i in range(60)])
    s.add_documents([Document(text="small markets note", doc_id="s1", tenant_id="small")])
    # adaptive over-fetch must still surface the small tenant's doc (not [])
    hits = s.search("markets trading", 5, tenant_id="small")
    assert [h.metadata["doc_id"] for h in hits] == ["s1"]


# --- LLM-based memory fact extraction -------------------------------------
def test_llm_fact_extraction(tmp_path):
    cfg = Config(memory_db_path=str(tmp_path / "m.db"), llm_memory_extraction=True)
    mem = MemoryManager(Embedder(force_fallback=True), cfg)
    mem.llm = lambda p: '[{"fact":"User uses PostgreSQL","key":"db","value":"PostgreSQL","importance":0.8}]'
    new = mem.extract_and_store("we run everything on postgres", "noted", scope="u")
    facts = [r for r in new if r.mtype == MemoryType.SEMANTIC]
    assert any(f.fact_value == "PostgreSQL" for f in facts)


def test_llm_extraction_falls_back_on_bad_json(tmp_path):
    cfg = Config(memory_db_path=str(tmp_path / "m.db"), llm_memory_extraction=True)
    mem = MemoryManager(Embedder(force_fallback=True), cfg)
    mem.llm = lambda p: "sorry, no json here"
    new = mem.extract_and_store("the user uses Rust", "ok", scope="u")
    assert any(r.mtype == MemoryType.SEMANTIC for r in new)  # heuristic fallback fired


# --- memory survives an embedding-model change ----------------------------
def test_memory_dim_mismatch_degrades(tmp_path):
    cfg = Config(memory_db_path=str(tmp_path / "m.db"))
    mem = MemoryManager(Embedder(force_fallback=True, dim=8), cfg)
    mem.store(MemoryRecord(text="hello world", mtype=MemoryType.LONG_TERM))
    # simulate switching to a different embedding model (different dim)
    mem.embedder = Embedder(force_fallback=True, dim=16)
    assert mem.retrieve("hello", scope="global") == []  # no crash, graceful empty
