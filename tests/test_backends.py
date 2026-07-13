"""Tests for the pluggable vector-index backends."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from contextx.backends import FaissBackend, NumpyBackend, make_backend  # noqa: E402
from contextx.config import Config  # noqa: E402


def test_backend_selection():
    assert make_backend(Config(vector_backend="numpy")).name == "numpy"
    assert make_backend(Config(vector_backend="faiss")).name == "faiss"
    assert make_backend(Config(vector_backend="auto")).name in ("faiss", "numpy")
    with pytest.raises(ValueError):
        make_backend(Config(vector_backend="bogus"))


def test_numpy_backend_roundtrip():
    b = NumpyBackend(Config())
    b.add([0, 1, 2], np.eye(3, dtype=np.float32))
    assert b.count() == 3
    rows, sims = b.search(np.array([1, 0, 0], dtype=np.float32), 2)
    assert int(rows[0]) == 0            # identity row 0 nearest to [1,0,0]
    b.reset()
    assert b.count() == 0


def test_faiss_backend_roundtrip():
    b = FaissBackend(Config())
    b.add([0, 1, 2, 3], np.eye(4, dtype=np.float32))
    assert b.count() == 4
    rows, _ = b.search(np.array([0, 1, 0, 0], dtype=np.float32), 2)
    assert 1 in [int(r) for r in rows]  # identity row 1 nearest to [0,1,0,0]


def test_pgvector_unavailable_fails_cleanly():
    # no psycopg / no Postgres here -> must raise, not silently misbehave
    with pytest.raises(Exception):
        make_backend(Config(vector_backend="pgvector"))


def test_store_honours_selected_backend(tmp_path):
    from contextx.embeddings import Embedder
    from contextx.store import VectorStore
    from contextx.types import Document
    s = VectorStore(Embedder(force_fallback=True),
                    Config(index_dir=str(tmp_path / "i"), vector_backend="numpy"))
    s.add_documents([Document(text="hello world", doc_id="h")])
    assert s.backend == "numpy"
    assert s.search("hello", 1)[0].metadata["doc_id"] == "h"
