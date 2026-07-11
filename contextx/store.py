"""Persistent vector store — the ingest/query split.

The toy re-embedded the entire candidate pool and built a flat index on every
request (O(N) per query). Production separates:

  * INGEST (once, or incrementally): chunk -> embed -> add to a persistent
    FAISS HNSW index; chunk text + metadata live in a SQLite sidecar.
  * QUERY (per request): embed only the query, ANN-search the pre-built index,
    hydrate hits from SQLite. Cost is independent of corpus size.

HNSW gives sub-linear ANN search that scales to millions of vectors. If FAISS
is unavailable we fall back to a persisted numpy matrix + brute-force cosine,
so the pipeline still runs (just linear).

The index and SQLite store are saved under `Config.index_dir` and reloaded on
construction, so ingest survives process restarts.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

import numpy as np

from .chunking import chunk_text
from .config import Config
from .embeddings import Embedder
from .types import ContextItem, Document, Source


class VectorStore:
    def __init__(self, embedder: Embedder, config: Config | None = None) -> None:
        self.cfg = config or Config()
        self.embedder = embedder
        self.dir = Path(self.cfg.index_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

        self._index = None                 # faiss index (or None -> numpy path)
        self._matrix: np.ndarray | None = None  # numpy fallback store
        self.backend = "numpy"
        try:
            import faiss  # noqa: F401

            self.backend = "faiss"
        except Exception:
            self.backend = "numpy"

        self._db = sqlite3.connect(
            str(self.dir / "chunks.db"), check_same_thread=False
        )
        self._db.execute("PRAGMA journal_mode=WAL;")
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS chunks (
                   row INTEGER PRIMARY KEY,
                   chunk_id TEXT UNIQUE, doc_id TEXT, source TEXT,
                   text TEXT, metadata TEXT, timestamp REAL, embedding BLOB
               )"""
        )
        self._db.commit()
        self._load()

    # --- ingest -----------------------------------------------------------
    def add_documents(self, docs: list[Document]) -> int:
        """Chunk, embed, and persist. Idempotent per chunk_id. Returns #chunks."""
        with self._lock:
            existing = {r[0] for r in self._db.execute("SELECT chunk_id FROM chunks")}
            new_texts: list[str] = []
            rows: list[tuple] = []
            for doc in docs:
                pieces = chunk_text(
                    doc.text,
                    self.cfg.chunk_target_tokens,
                    self.cfg.chunk_overlap_tokens,
                )
                for i, piece in enumerate(pieces):
                    cid = f"{doc.doc_id}:{i}"
                    if cid in existing:
                        continue
                    existing.add(cid)
                    rows.append(
                        (
                            cid,
                            doc.doc_id,
                            doc.source.value,
                            piece,
                            json.dumps(doc.metadata),
                            doc.timestamp,
                        )
                    )
                    new_texts.append(piece)
            if not new_texts:
                return 0

            vecs = self.embedder.encode(new_texts)  # batched inside Embedder
            self._ensure_index(vecs.shape[1])
            start_row = self._count()
            for offset, (row, vec) in enumerate(zip(rows, vecs)):
                r = start_row + offset
                self._db.execute(
                    "INSERT INTO chunks (row, chunk_id, doc_id, source, text, metadata,"
                    " timestamp, embedding) VALUES (?,?,?,?,?,?,?,?)",
                    (r, *row, vec.astype(np.float32).tobytes()),
                )
            self._db.commit()
            self._add_vectors(vecs)
            self._save()
            return len(new_texts)

    # --- query ------------------------------------------------------------
    def search(
        self, query: str, k: int, metadata_filter: dict[str, Any] | None = None
    ) -> list[ContextItem]:
        with self._lock:
            n = self._count()
            if n == 0:
                return []
            qvec = self.embedder.encode_one(query).astype(np.float32)
            # over-fetch when filtering so the post-filter still yields ~k
            fetch = k * 4 if metadata_filter else k
            idxs, sims = self._ann_search(qvec, min(fetch, n))
            out: list[ContextItem] = []
            for row, sim in zip(idxs, sims):
                rec = self._db.execute(
                    "SELECT chunk_id, doc_id, source, text, metadata, timestamp, embedding"
                    " FROM chunks WHERE row=?",
                    (int(row),),
                ).fetchone()
                if rec is None:
                    continue
                cid, doc_id, source, text, meta_json, ts, emb_blob = rec
                meta = json.loads(meta_json)
                if metadata_filter and not _matches(meta, metadata_filter):
                    continue
                src = Source(source)
                item = ContextItem(
                    text=text,
                    source=src,
                    timestamp=ts,
                    similarity=float(sim),
                    embedding=np.frombuffer(emb_blob, dtype=np.float32) if emb_blob else None,
                    metadata={**meta, "chunk_id": cid, "doc_id": doc_id},
                    trusted=src not in _UNTRUSTED,
                )
                out.append(item)
                if len(out) >= k:
                    break
            return out

    def stats(self) -> dict:
        return {"backend": self.backend, "chunks": self._count()}

    # --- index internals --------------------------------------------------
    def _ensure_index(self, dim: int) -> None:
        if self.backend == "faiss" and self._index is None:
            import faiss

            index = faiss.IndexHNSWFlat(dim, self.cfg.hnsw_M, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = self.cfg.hnsw_ef_construction
            index.hnsw.efSearch = self.cfg.hnsw_ef_search
            self._index = index

    def _add_vectors(self, vecs: np.ndarray) -> None:
        vecs = vecs.astype(np.float32)
        if self.backend == "faiss":
            self._index.add(vecs)
        else:
            self._matrix = vecs if self._matrix is None else np.vstack([self._matrix, vecs])

    def _ann_search(self, qvec: np.ndarray, k: int):
        if self.backend == "faiss":
            self._index.hnsw.efSearch = max(self.cfg.hnsw_ef_search, k)
            sims, idxs = self._index.search(qvec.reshape(1, -1), k)
            return idxs[0], sims[0]
        sims = self._matrix @ qvec
        top = np.argsort(sims)[::-1][:k]
        return top, sims[top]

    def _count(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    # --- persistence ------------------------------------------------------
    def _save(self) -> None:
        if self.backend == "faiss" and self._index is not None:
            import faiss

            faiss.write_index(self._index, str(self.dir / "index.faiss"))
        elif self._matrix is not None:
            np.save(self.dir / "matrix.npy", self._matrix)

    def _load(self) -> None:
        if self.backend == "faiss":
            p = self.dir / "index.faiss"
            if p.exists():
                import faiss

                self._index = faiss.read_index(str(p))
        else:
            p = self.dir / "matrix.npy"
            if p.exists():
                self._matrix = np.load(p)


# untrusted set imported lazily to avoid a cycle at module import time
from .types import UNTRUSTED_SOURCES as _UNTRUSTED  # noqa: E402


def _matches(meta: dict, flt: dict) -> bool:
    return all(meta.get(k) == v for k, v in flt.items())
