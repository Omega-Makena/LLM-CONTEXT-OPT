"""Tests for the Ollama backend and custom-data (JSONL) eval loading."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contextx import Config  # noqa: E402
from contextx.eval import load_corpus_jsonl, load_jsonl, save_corpus_jsonl  # noqa: E402
from contextx.llm import LLM, _system_text  # noqa: E402
from contextx.types import Document  # noqa: E402


def test_system_text_flatten():
    assert _system_text("plain") == "plain"
    assert _system_text([{"type": "text", "text": "A"}, {"type": "text", "text": "B"}]) == "A\nB"


def test_llm_mock_provider():
    llm = LLM(Config(llm_provider="mock"))
    assert llm.backend == "mock"
    r = llm.complete("sys", "<user_request>\nhi\n</user_request>")
    assert r.backend == "mock"


def test_llm_ollama_selected_and_degrades():
    # force ollama at a dead host; must select ollama, then degrade to mock on call
    llm = LLM(Config(llm_provider="ollama", ollama_host="http://127.0.0.1:1",
                     llm_max_retries=0))
    assert llm.backend == "ollama"
    r = llm.complete("sys", "<user_request>\nhi\n</user_request>")
    assert r.backend == "mock"
    assert "ollama unavailable" in r.text


def test_openai_backend_selection(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # no key -> forced openai degrades to mock
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert LLM(Config(llm_provider="openai")).backend == "mock"
    # with key -> selects openai (no call made)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert LLM(Config(llm_provider="openai")).backend == "openai"
    # auto prefers a key-based provider over local ollama
    assert LLM(Config(llm_provider="auto")).backend == "openai"


def test_cache_invalidate_responses():
    import numpy as np
    from contextx.cache import Cache
    c = Cache()
    c.get_or_compute("llm", "k1", lambda: "old-answer")
    c.semantic_get_or_compute("llm", np.ones(4, dtype=np.float32) / 2, "k2", lambda: "x")
    assert any(k.startswith("llm:") for k in c._store)
    c.invalidate_responses()
    assert not any(k.startswith("llm:") for k in c._store)
    assert c._sem_vecs == []


def test_engine_update_invalidates_response_cache(tmp_path):
    from contextx import ContextEngine, Request
    cfg = Config(index_dir=str(tmp_path / "idx"), memory_db_path=str(tmp_path / "m.db"))
    engine = ContextEngine(config=cfg)
    engine.ingest([Document(text="The vault code is ALPHA.", doc_id="v")])
    engine.run(Request(user_message="what is the vault code?",
                       max_context_tokens=3000, reserve_output_tokens=800))
    assert any(k.startswith("llm:") for k in engine.cache._store)  # a response cached
    engine.update([Document(text="The vault code is OMEGA now.", doc_id="v")])
    assert not any(k.startswith("llm:") for k in engine.cache._store)  # invalidated


def test_load_corpus_jsonl_roundtrip(tmp_path):
    docs = [Document(text="alpha content", doc_id="a"),
            Document(text="beta content", doc_id="b")]
    p = tmp_path / "corpus.jsonl"
    save_corpus_jsonl(docs, str(p))
    loaded = load_corpus_jsonl(str(p))
    assert [d.doc_id for d in loaded] == ["a", "b"]
    assert loaded[0].text == "alpha content"


def test_bundled_sample_jsonl_is_valid():
    base = Path(__file__).resolve().parents[1] / "examples" / "data"
    corpus = load_corpus_jsonl(str(base / "sample_corpus.jsonl"))
    queries = load_jsonl(str(base / "sample_queries.jsonl"))
    assert len(corpus) >= 6 and len(queries) >= 5
    corpus_ids = {d.doc_id for d in corpus}
    for q in queries:
        assert q.relevant
        for did in q.relevant:
            assert did in corpus_ids  # every label points at a real doc
