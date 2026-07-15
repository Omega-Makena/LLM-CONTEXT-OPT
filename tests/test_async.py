"""Tests for the async entry points (arun / arun_stream)."""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from contextx import Config, ContextEngine, Request  # noqa: E402
from contextx.types import Document  # noqa: E402


@pytest.fixture(scope="module")
def engine():
    d = tempfile.mkdtemp(prefix="contextx_async_")
    e = ContextEngine(config=Config(index_dir=d + "/i", memory_db_path=d + "/m.db"))
    e.ingest([Document(text="Redis is an in-memory data store.", doc_id="r")])
    return e


def test_arun(engine):
    res = asyncio.run(engine.arun(Request(user_message="what is redis?")))
    assert res.answer
    assert any(s["doc_id"] == "r" for s in res.sources)


def test_arun_stream(engine):
    async def go():
        sr = await engine.arun_stream(
            Request(user_message="what is redis?"), write_memory=False)
        chunks = [c async for c in sr.stream]     # async iteration, off the loop
        return sr, chunks

    sr, chunks = asyncio.run(go())
    assert chunks and "".join(chunks).strip()
    assert any(s["doc_id"] == "r" for s in sr.sources)


def test_arun_with_sources_concurrent(engine):
    import time

    async def fetch_a(req):
        await asyncio.sleep(0.3)
        return ["Fetched note: the sky is blue today."]

    async def fetch_b(req):
        await asyncio.sleep(0.3)
        return ["Fetched note: unrelated weather trivia."]

    async def go():
        t = time.perf_counter()
        res = await engine.arun_with_sources(
            Request(user_message="what color is the sky?"),
            [fetch_a, fetch_b], write_memory=False)
        return res, time.perf_counter() - t

    res, elapsed = asyncio.run(go())
    assert elapsed < 0.6                        # 2x0.3s ran concurrently, not 0.6s+
    assert "sky is blue" in res.prompt.user      # fetched source reached retrieval
