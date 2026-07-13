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

from .backends import make_backend
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

        self.vindex = make_backend(self.cfg)
        self.backend = self.vindex.name

        self._db = sqlite3.connect(str(self.dir / "chunks.db"), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL;")
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS chunks (
                   row INTEGER PRIMARY KEY,
                   chunk_id TEXT UNIQUE, doc_id TEXT, source TEXT,
                   text TEXT, metadata TEXT, timestamp REAL, embedding BLOB,
                   tenant_id TEXT DEFAULT 'default', acl TEXT DEFAULT '[]'
               )"""
        )
        self._migrate_columns()  # add tenant_id/acl to pre-existing indexes
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_doc ON chunks(doc_id)")
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_tenant ON chunks(tenant_id)")
        self._db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        self.fts_enabled = self._init_fts()
        self._db.commit()
        self._load()
        self._check_model()

    def _migrate_columns(self) -> None:
        cols = {r[1] for r in self._db.execute("PRAGMA table_info(chunks)")}
        if "tenant_id" not in cols:
            self._db.execute("ALTER TABLE chunks ADD COLUMN tenant_id TEXT DEFAULT 'default'")
        if "acl" not in cols:
            self._db.execute("ALTER TABLE chunks ADD COLUMN acl TEXT DEFAULT '[]'")

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
                                 json.dumps(doc.metadata), doc.timestamp,
                                 doc.tenant_id, json.dumps(doc.acl)))
                    new_texts.append(piece)
            if not new_texts:
                return 0

            vecs = self.embedder.encode(new_texts)
            start_row = self._vector_count()  # next append position (survives deletes)
            row_ids: list[int] = []
            for offset, (row, vec) in enumerate(zip(rows, vecs)):
                r = start_row + offset
                row_ids.append(r)
                self._db.execute(
                    "INSERT INTO chunks (row, chunk_id, doc_id, source, text, metadata,"
                    " timestamp, tenant_id, acl, embedding) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (r, *row, vec.astype(np.float32).tobytes()),
                )
                if self.fts_enabled:
                    self._db.execute("INSERT INTO chunks_fts(text, row) VALUES(?,?)", (row[3], r))
            self._stamp_model()
            self._db.commit()
            self.vindex.add(row_ids, vecs)
            self._save()
            return len(new_texts)

    # --- delete / update --------------------------------------------------
    def delete_document(
        self, doc_id: str, auto_rebuild: bool = True, tenant_id: str | None = None
    ) -> int:
        """Remove all chunks of a document. If `tenant_id` is given, only that
        tenant's chunks are removed (so one tenant can't delete another's doc by
        id). Vectors are tombstoned and skipped at query time; the index compacts
        once tombstones pile up. Returns #chunks removed."""
        with self._lock:
            if tenant_id is None:
                sql, params = "SELECT row FROM chunks WHERE doc_id=?", (doc_id,)
                dele = "DELETE FROM chunks WHERE doc_id=?"
            else:
                sql = "SELECT row FROM chunks WHERE doc_id=? AND tenant_id=?"
                params = (doc_id, tenant_id)
                dele = "DELETE FROM chunks WHERE doc_id=? AND tenant_id=?"
            rows = [r[0] for r in self._db.execute(sql, params)]
            if not rows:
                return 0
            self._db.execute(dele, params)
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
                "SELECT chunk_id, doc_id, source, text, metadata, timestamp, tenant_id,"
                " acl, embedding FROM chunks ORDER BY row"
            ).fetchall()
            self.vindex.reset()
            self._db.execute("DELETE FROM chunks")
            if self.fts_enabled:
                self._db.execute("DELETE FROM chunks_fts")
            vecs: list[np.ndarray] = []
            for new_row, rec in enumerate(live):
                cid, doc_id, source, text, meta, ts, tenant, acl, emb = rec
                self._db.execute(
                    "INSERT INTO chunks (row, chunk_id, doc_id, source, text, metadata,"
                    " timestamp, tenant_id, acl, embedding) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (new_row, cid, doc_id, source, text, meta, ts, tenant, acl, emb),
                )
                if self.fts_enabled:
                    self._db.execute("INSERT INTO chunks_fts(text, row) VALUES(?,?)", (text, new_row))
                vecs.append(np.frombuffer(emb, dtype=np.float32))
            self._db.commit()
            if vecs:
                mat = np.vstack(vecs).astype(np.float32)
                self.vindex.add(list(range(len(vecs))), mat)
            self._save()

    # --- query: semantic --------------------------------------------------
    def search(
        self,
        query: str,
        k: int,
        metadata_filter: dict[str, Any] | None = None,
        tenant_id: str = "default",
        principals: set[str] | None = None,
    ) -> list[ContextItem]:
        principals = principals or set()
        with self._lock:
            vcount = self._vector_count()
            if vcount == 0:
                return []
            qvec = self.embedder.encode_one(query).astype(np.float32)
            tombs = vcount - self._count()
            fetch = min(vcount, max(k * 8, k + tombs))
            # Adaptive over-fetch: tenant/ACL/metadata filtering is post-hoc, so a
            # small tenant's docs can all sit beyond the first `fetch` neighbours
            # in a large index. Expand until we have k authorized hits or we've
            # scanned the whole index — a small tenant is never starved.
            while True:
                idxs, sims = self.vindex.search(qvec, fetch)
                out: list[ContextItem] = []
                for row, sim in zip(idxs, sims):
                    item = self._hydrate(int(row), similarity=float(sim))
                    if item is None:
                        continue
                    if not self._authorized(item, tenant_id, principals):
                        continue
                    if metadata_filter and not _matches(item.metadata, metadata_filter):
                        continue
                    out.append(item)
                    if len(out) >= k:
                        break
                if len(out) >= k or fetch >= vcount:
                    return out
                fetch = min(vcount, fetch * 4)

    # --- query: lexical (BM25 over FTS5) ----------------------------------
    def lexical_search(
        self,
        query: str,
        k: int,
        metadata_filter: dict[str, Any] | None = None,
        tenant_id: str = "default",
        principals: set[str] | None = None,
    ) -> list[ContextItem]:
        principals = principals or set()
        if not self.fts_enabled:
            return []
        with self._lock:
            match = _fts_query(query)
            if not match:
                return []
            total = self._count()
            limit = k * 8
            while True:
                try:
                    rows = self._db.execute(
                        "SELECT row FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank"
                        " LIMIT ?",
                        (match, limit),
                    ).fetchall()
                except sqlite3.OperationalError:
                    return []
                out: list[ContextItem] = []
                for (row,) in rows:
                    item = self._hydrate(int(row))
                    if item is None:
                        continue
                    if not self._authorized(item, tenant_id, principals):
                        continue
                    if metadata_filter and not _matches(item.metadata, metadata_filter):
                        continue
                    out.append(item)
                    if len(out) >= k:
                        break
                # stop when satisfied, when FTS is exhausted, or the limit covers all
                if len(out) >= k or len(rows) < limit or limit >= total:
                    return out
                limit = min(total, limit * 4)

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
            "SELECT chunk_id, doc_id, source, text, metadata, timestamp, tenant_id,"
            " acl, embedding FROM chunks WHERE row=?",
            (row,),
        ).fetchone()
        if rec is None:
            return None  # tombstoned
        cid, doc_id, source, text, meta_json, ts, tenant, acl_json, emb_blob = rec
        meta = json.loads(meta_json)
        src = Source(source)
        return ContextItem(
            text=text,
            source=src,
            timestamp=ts,
            similarity=similarity,
            embedding=np.frombuffer(emb_blob, dtype=np.float32) if emb_blob else None,
            metadata={
                **meta, "chunk_id": cid, "doc_id": doc_id,
                "tenant_id": tenant, "acl": json.loads(acl_json or "[]"),
            },
            trusted=src not in _UNTRUSTED,
        )

    @staticmethod
    def _authorized(item: ContextItem, tenant_id: str, principals: set[str]) -> bool:
        if item.metadata.get("tenant_id", "default") != tenant_id:
            return False
        acl = item.metadata.get("acl") or []
        return not acl or bool(set(acl) & principals)

    def _count(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def _vector_count(self) -> int:
        return self.vindex.count()

    def _tombstone_ratio(self) -> float:
        vc = self._vector_count()
        return (vc - self._count()) / vc if vc else 0.0

    # --- persistence ------------------------------------------------------
    def _save(self) -> None:
        self.vindex.save(self.dir)

    def _load(self) -> None:
        self.vindex.load(self.dir)


from .types import UNTRUSTED_SOURCES as _UNTRUSTED  # noqa: E402


def _matches(meta: dict, flt: dict) -> bool:
    return all(meta.get(k) == v for k, v in flt.items())


def _fts_query(query: str) -> str | None:
    terms = [t for t in _WORD.findall(query.lower()) if len(t) > 1]
    if not terms:
        return None
    return " OR ".join(f'"{t}"' for t in terms)
