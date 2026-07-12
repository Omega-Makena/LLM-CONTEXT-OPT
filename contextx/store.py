"""Persistent vector store — the ingest/query split, with a lexical channel.

  * INGEST: chunk -> embed -> persistent FAISS HNSW index; chunk text + metadata
    + the embedding blob live in a SQLite sidecar, plus an FTS5 full-text index.
  * QUERY: `search` does semantic ANN; `lexical_search` does BM25 over FTS5. The
    retriever fuses the two (real hybrid search).

Production concerns handled here that the earlier version missed:
  * document delete / update (with index compaction) — the index no longer goes
    stale when the corpus changes.
  * embedding-model version stamping — refuses to query an index built with a
    different embedding model instead of returning silent garbage.

If FAISS is unavailable we fall back to a persisted numpy matrix + brute-force
cosine; if the SQLite build lacks FTS5, lexical search degrades to empty and the
retriever runs vector-only. Everything still works, just with fewer guarantees.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any

import numpy as np

from .chunking import chunk_text
from .config import Config
from .embeddings import Embedder
from .types import ContextItem, Document, Source

_WORD = re.compile(r"[a-z0-9]+")


class ModelMismatchError(RuntimeError):
    """Raised when the index was built with a different embedding model."""


class VectorStore:
    def __init__(self, embedder: Embedder, config: Config | None = None) -> None:
        self.cfg = config or Config()
        self.embedder = embedder
        self.dir = Path(self.cfg.index_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

        self._index = None
        self._matrix: np.ndarray | None = None
        self.backend = "numpy"
        try:
            import faiss  # noqa: F401

            self.backend = "faiss"
        except Exception:
            self.backend = "numpy"

        self._db = sqlite3.connect(str(self.dir / "chunks.db"), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL;")
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS chunks (
                   row INTEGER PRIMARY KEY,
                   chunk_id TEXT UNIQUE, doc_id TEXT, source TEXT,
                   text TEXT, metadata TEXT, timestamp REAL, embedding BLOB
               )"""
        )
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_doc ON chunks(doc_id)")
        self._db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        self.fts_enabled = self._init_fts()
        self._db.commit()
        self._load()
        self._check_model()

    def _init_fts(self) -> bool:
        try:
            self._db.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, row UNINDEXED)"
            )
            return True
        except sqlite3.OperationalError:
            return False  # sqlite built without FTS5

    # --- model versioning -------------------------------------------------
    def _check_model(self) -> None:
        row = self._db.execute("SELECT value FROM meta WHERE key='embed_model'").fetchone()
        if row and self._count() > 0 and row[0] != self.embedder.model_name:
            raise ModelMismatchError(
                f"Index at {self.dir} was built with embedding model '{row[0]}', but the "
                f"engine is now using '{self.embedder.model_name}'. Query results would be "
                f"meaningless. Rebuild the index (delete {self.dir}) or restore the original "
                f"model."
            )

    def _stamp_model(self) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO meta(key,value) VALUES('embed_model',?)",
            (self.embedder.model_name,),
        )
        self._db.execute(
            "INSERT OR REPLACE INTO meta(key,value) VALUES('embed_dim',?)",
            (str(self.embedder.dim),),
        )

    # --- ingest -----------------------------------------------------------
    def add_documents(self, docs: list[Document]) -> int:
        """Chunk, embed, and persist. Idempotent per chunk_id. Returns #chunks."""
        with self._lock:
            existing = {r[0] for r in self._db.execute("SELECT chunk_id FROM chunks")}
            new_texts: list[str] = []
            rows: list[tuple] = []
            for doc in docs:
                for i, piece in enumerate(
                    chunk_text(doc.text, self.cfg.chunk_target_tokens, self.cfg.chunk_overlap_tokens)
                ):
                    cid = f"{doc.doc_id}:{i}"
                    if cid in existing:
                        continue
                    existing.add(cid)
                    rows.append((cid, doc.doc_id, doc.source.value, piece,
                                 json.dumps(doc.metadata), doc.timestamp))
                    new_texts.append(piece)
            if not new_texts:
                return 0

            vecs = self.embedder.encode(new_texts)
            self._ensure_index(vecs.shape[1])
            start_row = self._vector_count()  # next append position (survives deletes)
            for offset, (row, vec) in enumerate(zip(rows, vecs)):
                r = start_row + offset
                self._db.execute(
                    "INSERT INTO chunks (row, chunk_id, doc_id, source, text, metadata,"
                    " timestamp, embedding) VALUES (?,?,?,?,?,?,?,?)",
                    (r, *row, vec.astype(np.float32).tobytes()),
                )
                if self.fts_enabled:
                    self._db.execute("INSERT INTO chunks_fts(text, row) VALUES(?,?)", (row[3], r))
            self._stamp_model()
            self._db.commit()
            self._add_vectors(vecs)
            self._save()
            return len(new_texts)

    # --- delete / update --------------------------------------------------
    def delete_document(self, doc_id: str, auto_rebuild: bool = True) -> int:
        """Remove all chunks of a document. Vectors are tombstoned in the index
        and skipped at query time; the index is compacted once tombstones pile
        up (or call `rebuild()` explicitly). Returns #chunks removed."""
        with self._lock:
            rows = [r[0] for r in self._db.execute(
                "SELECT row FROM chunks WHERE doc_id=?", (doc_id,))]
            if not rows:
                return 0
            self._db.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            if self.fts_enabled:
                self._db.executemany("DELETE FROM chunks_fts WHERE row=?", [(r,) for r in rows])
            self._db.commit()
            if auto_rebuild and self._tombstone_ratio() > 0.2:
                self.rebuild()
            return len(rows)

    def update_documents(self, docs: list[Document]) -> int:
        """Replace documents by doc_id (delete existing chunks, then re-ingest)."""
        with self._lock:
            for d in docs:
                self.delete_document(d.doc_id, auto_rebuild=False)
            n = self.add_documents(docs)
            if self._tombstone_ratio() > 0.2:
                self.rebuild()
            return n

    def rebuild(self) -> None:
        """Compact: rebuild the vector index and row numbering from live chunks
        only, dropping tombstones left by deletes."""
        with self._lock:
            live = self._db.execute(
                "SELECT chunk_id, doc_id, source, text, metadata, timestamp, embedding"
                " FROM chunks ORDER BY row"
            ).fetchall()
            self._index = None
            self._matrix = None
            self._db.execute("DELETE FROM chunks")
            if self.fts_enabled:
                self._db.execute("DELETE FROM chunks_fts")
            vecs: list[np.ndarray] = []
            for new_row, rec in enumerate(live):
                cid, doc_id, source, text, meta, ts, emb = rec
                self._db.execute(
                    "INSERT INTO chunks (row, chunk_id, doc_id, source, text, metadata,"
                    " timestamp, embedding) VALUES (?,?,?,?,?,?,?,?)",
                    (new_row, cid, doc_id, source, text, meta, ts, emb),
                )
                if self.fts_enabled:
                    self._db.execute("INSERT INTO chunks_fts(text, row) VALUES(?,?)", (text, new_row))
                vecs.append(np.frombuffer(emb, dtype=np.float32))
            self._db.commit()
            if vecs:
                mat = np.vstack(vecs).astype(np.float32)
                self._ensure_index(mat.shape[1])
                self._add_vectors(mat)
            self._save()

    # --- query: semantic --------------------------------------------------
    def search(
        self, query: str, k: int, metadata_filter: dict[str, Any] | None = None
    ) -> list[ContextItem]:
        with self._lock:
            vcount = self._vector_count()
            if vcount == 0:
                return []
            qvec = self.embedder.encode_one(query).astype(np.float32)
            tombs = vcount - self._count()
            fetch = min(vcount, max(k * 4 if metadata_filter else k, k + tombs))
            idxs, sims = self._ann_search(qvec, fetch)
            out: list[ContextItem] = []
            for row, sim in zip(idxs, sims):
                item = self._hydrate(int(row), similarity=float(sim))
                if item is None:
                    continue
                if metadata_filter and not _matches(item.metadata, metadata_filter):
                    continue
                out.append(item)
                if len(out) >= k:
                    break
            return out

    # --- query: lexical (BM25 over FTS5) ----------------------------------
    def lexical_search(
        self, query: str, k: int, metadata_filter: dict[str, Any] | None = None
    ) -> list[ContextItem]:
        if not self.fts_enabled:
            return []
        with self._lock:
            match = _fts_query(query)
            if not match:
                return []
            try:
                rows = self._db.execute(
                    "SELECT row FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank"
                    " LIMIT ?",
                    (match, k * 4 if metadata_filter else k),
                ).fetchall()
            except sqlite3.OperationalError:
                return []
            out: list[ContextItem] = []
            for (row,) in rows:
                item = self._hydrate(int(row))
                if item is None:
                    continue
                if metadata_filter and not _matches(item.metadata, metadata_filter):
                    continue
                out.append(item)
                if len(out) >= k:
                    break
            return out

    def stats(self) -> dict:
        return {
            "backend": self.backend,
            "chunks": self._count(),
            "vectors": self._vector_count(),
            "fts": self.fts_enabled,
        }

    # --- internals --------------------------------------------------------
    def _hydrate(self, row: int, similarity: float = 0.0) -> ContextItem | None:
        rec = self._db.execute(
            "SELECT chunk_id, doc_id, source, text, metadata, timestamp, embedding"
            " FROM chunks WHERE row=?",
            (row,),
        ).fetchone()
        if rec is None:
            return None  # tombstoned
        cid, doc_id, source, text, meta_json, ts, emb_blob = rec
        meta = json.loads(meta_json)
        src = Source(source)
        return ContextItem(
            text=text,
            source=src,
            timestamp=ts,
            similarity=similarity,
            embedding=np.frombuffer(emb_blob, dtype=np.float32) if emb_blob else None,
            metadata={**meta, "chunk_id": cid, "doc_id": doc_id},
            trusted=src not in _UNTRUSTED,
        )

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

    def _vector_count(self) -> int:
        if self.backend == "faiss":
            return self._index.ntotal if self._index is not None else 0
        return self._matrix.shape[0] if self._matrix is not None else 0

    def _tombstone_ratio(self) -> float:
        vc = self._vector_count()
        return (vc - self._count()) / vc if vc else 0.0

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


from .types import UNTRUSTED_SOURCES as _UNTRUSTED  # noqa: E402


def _matches(meta: dict, flt: dict) -> bool:
    return all(meta.get(k) == v for k, v in flt.items())


def _fts_query(query: str) -> str | None:
    terms = [t for t in _WORD.findall(query.lower()) if len(t) > 1]
    if not terms:
        return None
    return " OR ".join(f'"{t}"' for t in terms)
