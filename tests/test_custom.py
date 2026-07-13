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
