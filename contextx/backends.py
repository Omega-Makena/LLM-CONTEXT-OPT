"""Pluggable vector-index backends — the seam to scale off single-node FAISS.

`VectorStore` owns chunk text/metadata/tenant/ACL/FTS in SQLite; the *vector
index* (the part with the single-node RAM ceiling) lives behind this narrow
interface so it can be swapped:

  * FaissBackend  — HNSW ANN, in-process (default when faiss is installed).
  * NumpyBackend  — brute-force cosine fallback, in-process.
  * PgVectorBackend — Postgres + pgvector, for horizontal scale. EXPERIMENTAL:
    requires a running Postgres with the `vector` extension and `psycopg`; it is
    NOT exercised by CI (no DB here), so treat it as unverified until you run it
    against your own Postgres.

Contract: the store assigns each vector an integer `row` id and guarantees rows
are dense/sequential (rebuild renumbers 0..n-1), so FAISS/numpy can ignore the
explicit ids (position == row); pgvector stores them. `search` returns
(row_ids, cosine_sims).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from .config import Config


class VectorBackend(Protocol):
    name: str

    def add(self, rows: list[int], vecs: np.ndarray) -> None: ...
    def search(self, qvec: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]: ...
    def count(self) -> int: ...
    def reset(self) -> None: ...
    def save(self, directory: Path) -> None: ...
    def load(self, directory: Path) -> None: ...


class NumpyBackend:
    name = "numpy"

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._m: np.ndarray | None = None

    def add(self, rows: list[int], vecs: np.ndarray) -> None:
        vecs = vecs.astype(np.float32)
        self._m = vecs if self._m is None else np.vstack([self._m, vecs])

    def search(self, qvec: np.ndarray, k: int):
        sims = self._m @ qvec.astype(np.float32)
        top = np.argsort(sims)[::-1][:k]
        return top, sims[top]

    def count(self) -> int:
        return self._m.shape[0] if self._m is not None else 0

    def reset(self) -> None:
        self._m = None

    def save(self, directory: Path) -> None:
        if self._m is not None:
            np.save(Path(directory) / "matrix.npy", self._m)

    def load(self, directory: Path) -> None:
        p = Path(directory) / "matrix.npy"
        if p.exists():
            self._m = np.load(p)


class FaissBackend:
    name = "faiss"

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._index = None

    def _ensure(self, dim: int) -> None:
        if self._index is None:
            import faiss

            idx = faiss.IndexHNSWFlat(dim, self.cfg.hnsw_M, faiss.METRIC_INNER_PRODUCT)
            idx.hnsw.efConstruction = self.cfg.hnsw_ef_construction
            idx.hnsw.efSearch = self.cfg.hnsw_ef_search
            self._index = idx

    def add(self, rows: list[int], vecs: np.ndarray) -> None:
        vecs = vecs.astype(np.float32)
        self._ensure(vecs.shape[1])
        self._index.add(vecs)  # position == row (store guarantees dense rows)

    def search(self, qvec: np.ndarray, k: int):
        self._index.hnsw.efSearch = max(self.cfg.hnsw_ef_search, k)
        sims, idxs = self._index.search(qvec.reshape(1, -1).astype(np.float32), k)
        return idxs[0], sims[0]

    def count(self) -> int:
        return self._index.ntotal if self._index is not None else 0

    def reset(self) -> None:
        self._index = None

    def save(self, directory: Path) -> None:
        if self._index is not None:
            import faiss

            faiss.write_index(self._index, str(Path(directory) / "index.faiss"))

    def load(self, directory: Path) -> None:
        p = Path(directory) / "index.faiss"
        if p.exists():
            import faiss

            self._index = faiss.read_index(str(p))


def _vec_literal(v: np.ndarray) -> str:
    return "[" + ",".join(f"{float(x):.7f}" for x in v) + "]"


class PgVectorBackend:
    """EXPERIMENTAL — Postgres + pgvector. Requires psycopg and a running DB;
    not covered by CI. Vectors are L2-normalized, so cosine distance `<=>` maps
    to (1 - similarity)."""

    name = "pgvector"

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        try:
            import psycopg
        except Exception as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "pgvector backend needs psycopg: pip install 'psycopg[binary]' "
                "(and a Postgres with the pgvector extension)."
            ) from exc
        self._psycopg = psycopg
        self._conn = psycopg.connect(cfg.pg_dsn)
        self._table = cfg.pg_table
        self._ready = False

    def _ensure(self, dim: int) -> None:  # pragma: no cover - needs a DB
        if self._ready:
            return
        with self._conn.cursor() as c:
            c.execute("CREATE EXTENSION IF NOT EXISTS vector")
            c.execute(
                f"CREATE TABLE IF NOT EXISTS {self._table} "
                f"(row BIGINT PRIMARY KEY, embedding vector({dim}))"
            )
        self._conn.commit()
        self._ready = True

    def add(self, rows: list[int], vecs: np.ndarray) -> None:  # pragma: no cover
        self._ensure(vecs.shape[1])
        with self._conn.cursor() as c:
            for r, v in zip(rows, vecs):
                c.execute(
                    f"INSERT INTO {self._table}(row, embedding) VALUES (%s, %s) "
                    f"ON CONFLICT (row) DO UPDATE SET embedding = EXCLUDED.embedding",
                    (int(r), _vec_literal(v)),
                )
        self._conn.commit()

    def search(self, qvec: np.ndarray, k: int):  # pragma: no cover - needs a DB
        lit = _vec_literal(qvec)
        with self._conn.cursor() as c:
            c.execute(
                f"SELECT row, 1 - (embedding <=> %s) FROM {self._table} "
                f"ORDER BY embedding <=> %s LIMIT %s",
                (lit, lit, k),
            )
            rows = c.fetchall()
        return (np.array([r[0] for r in rows]),
                np.array([r[1] for r in rows], dtype=np.float32))

    def count(self) -> int:  # pragma: no cover - needs a DB
        with self._conn.cursor() as c:
            c.execute(f"SELECT count(*) FROM {self._table}")
            return c.fetchone()[0]

    def reset(self) -> None:  # pragma: no cover - needs a DB
        with self._conn.cursor() as c:
            c.execute(f"TRUNCATE {self._table}")
        self._conn.commit()

    def save(self, directory: Path) -> None:
        pass  # Postgres persists

    def load(self, directory: Path) -> None:
        pass


class QdrantBackend:
    """Qdrant vector database. `qdrant_location` is ":memory:" (ephemeral), a
    local path (persistent), or an http(s) URL (server). Vectors are
    L2-normalized, so cosine distance gives cosine similarity directly."""

    name = "qdrant"

    def __init__(self, cfg: Config) -> None:
        try:
            from qdrant_client import QdrantClient, models
        except Exception as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "qdrant backend needs qdrant-client: pip install qdrant-client"
            ) from exc
        self.cfg = cfg
        self._models = models
        loc = cfg.qdrant_location
        if loc == ":memory:":
            self._client = QdrantClient(location=":memory:")
        elif loc.startswith("http"):
            self._client = QdrantClient(url=loc)
        else:
            Path(loc).mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=loc)
        self._coll = cfg.qdrant_collection
        self._ready = self._client.collection_exists(self._coll)

    def _ensure(self, dim: int) -> None:
        if not self._ready:
            if not self._client.collection_exists(self._coll):
                self._client.create_collection(
                    self._coll,
                    vectors_config=self._models.VectorParams(
                        size=dim, distance=self._models.Distance.COSINE),
                )
            self._ready = True

    def add(self, rows: list[int], vecs: np.ndarray) -> None:
        self._ensure(vecs.shape[1])
        points = [
            self._models.PointStruct(id=int(r), vector=v.astype(float).tolist())
            for r, v in zip(rows, vecs)
        ]
        self._client.upsert(self._coll, points=points)

    def search(self, qvec: np.ndarray, k: int):
        if not self._ready:
            return np.array([]), np.array([])
        pts = self._client.query_points(
            self._coll, query=qvec.astype(float).tolist(), limit=int(k)).points
        return (np.array([p.id for p in pts]),
                np.array([p.score for p in pts], dtype=np.float32))

    def count(self) -> int:
        return self._client.count(self._coll).count if self._ready else 0

    def reset(self) -> None:
        if self._client.collection_exists(self._coll):
            self._client.delete_collection(self._coll)
        self._ready = False

    def save(self, directory: Path) -> None:
        pass  # Qdrant persists (path / server); :memory: is ephemeral

    def load(self, directory: Path) -> None:
        pass


def make_backend(cfg: Config) -> VectorBackend:
    choice = cfg.vector_backend
    if choice == "auto":
        try:
            import faiss  # noqa: F401

            return FaissBackend(cfg)
        except Exception:
            return NumpyBackend(cfg)
    if choice == "faiss":
        return FaissBackend(cfg)
    if choice == "numpy":
        return NumpyBackend(cfg)
    if choice == "pgvector":
        return PgVectorBackend(cfg)
    if choice == "qdrant":
        return QdrantBackend(cfg)
    raise ValueError(f"unknown vector_backend: {choice!r}")
