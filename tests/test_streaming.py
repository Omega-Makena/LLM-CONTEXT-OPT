"""Tests for streaming responses (LLM stream, engine.run_stream, SSE endpoint)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from contextx import Config, ContextEngine, Request  # noqa: E402
from contextx.llm import LLM  # noqa: E402
from contextx.types import Document  # noqa: E402


def test_mock_stream_reconstructs_complete():
    llm = LLM(Config(llm_provider="mock"))
    system = "sys"
    user = "<user_request>\nhi\n</user_request>\n- fact one\n- fact two"
    full = llm.complete(system, user).text
    streamed = "".join(llm.stream(system, user))
    assert streamed == full  # streaming is lossless vs the blocking call


def test_engine_run_stream(tmp_path):
    cfg = Config(index_dir=str(tmp_path / "i"), memory_db_path=str(tmp_path / "m.db"))
    engine = ContextEngine(config=cfg)
    engine.ingest([Document(text="Redis is an in-memory data store.", doc_id="r")])
    sr = engine.run_stream(Request(user_message="what is redis?"), write_memory=False)
    assert sr.sources and sr.sources[0]["doc_id"] == "r"   # citations known up front
    chunks = list(sr.stream)
    assert chunks and "".join(chunks).strip()               # got streamed text
    assert isinstance(sr.low_confidence, bool)


def test_service_stream_endpoint(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from contextx.service import create_app
    cfg = Config(index_dir=str(tmp_path / "i"), memory_db_path=str(tmp_path / "m.db"))
    engine = ContextEngine(config=cfg)
    engine.ingest([Document(text="Redis is an in-memory data store.", doc_id="r")])
    client = TestClient(create_app(engine))  # open/dev mode (no auth configured)
    resp = client.post("/query/stream", json={"user_message": "what is redis?"})
    assert resp.status_code == 200
    body = resp.text
    assert "event: meta" in body          # sources/meta sent first
    assert "data:" in body                # streamed chunks
    assert "event: done" in body          # terminated
