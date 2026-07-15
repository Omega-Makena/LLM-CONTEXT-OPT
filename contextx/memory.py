"""Stage 5 — Memory Manager (durable, concurrency-safe, bounded).

The toy stored memory in a JSON file rewritten on every turn — a data race
under concurrency and unbounded growth. This version uses SQLite:

  * atomic writes inside transactions; WAL mode for concurrent readers
  * a process-wide lock guards the write path
  * bounded: capped at `memory_max_records`; lowest value*recency evicted
  * fact merge on `fact_key` (update in place, no duplicates)

Memory types (working/session/long_term/episodic/semantic) and the lifecycle
(store -> score -> retrieve -> expire/forget) are unchanged in spirit; the
storage layer is what got hardened. Fact extraction is still a heuristic and is
flagged as the next thing to replace with an LLM extractor.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .config import Config
from .embeddings import Embedder
from .types import ContextItem, Source


class MemoryType(str, Enum):
    WORKING = "working"
    SESSION = "session"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


@dataclass
class MemoryRecord:
    text: str
    mtype: MemoryType
    importance: float = 0.5
    created: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    uses: int = 0
    ttl_s: float | None = None
    fact_key: str | None = None
    fact_value: str | None = None


class MemoryManager:
    def __init__(
        self,
        embedder: Embedder,
        config: Config | None = None,
        path: str | None = None,
    ) -> None:
        self.cfg = config or Config()
        self.embedder = embedder
        self.llm = None   # optional; set by the engine for LLM fact extraction
        self._lock = threading.RLock()
        self._db = sqlite3.connect(
            path or self.cfg.memory_db_path, check_same_thread=False
        )
        self._db.execute("PRAGMA journal_mode=WAL;")
        self._db.execute("PRAGMA synchronous=NORMAL;")
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS memory (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   text TEXT, mtype TEXT, importance REAL,
                   created REAL, last_used REAL, uses INTEGER,
                   ttl_s REAL, fact_key TEXT, fact_value TEXT,
                   embedding BLOB, scope TEXT DEFAULT 'global'
               )"""
        )
        if "scope" not in {r[1] for r in self._db.execute("PRAGMA table_info(memory)")}:
            self._db.execute("ALTER TABLE memory ADD COLUMN scope TEXT DEFAULT 'global'")
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_factkey ON memory(fact_key)")
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_scope ON memory(scope)")
        self._db.commit()

    # --- write side --------------------------------------------------------
    def store(self, record: MemoryRecord, scope: str = "global") -> None:
        with self._lock:
            emb = self.embedder.encode_one(record.text).astype(np.float32)
            if record.fact_key is not None:
                # fact_key uniqueness is per-scope, so users don't overwrite each other
                row = self._db.execute(
                    "SELECT id FROM memory WHERE fact_key=? AND scope=?",
                    (record.fact_key, scope),
                ).fetchone()
                if row is not None:
                    self._db.execute(
                        "UPDATE memory SET text=?, fact_value=?, importance=MAX(importance,?),"
                        " created=?, embedding=? WHERE id=?",
                        (record.text, record.fact_value, record.importance,
                         record.created, emb.tobytes(), row[0]),
                    )
                    self._db.commit()
                    return
            self._db.execute(
                "INSERT INTO memory (text, mtype, importance, created, last_used, uses,"
                " ttl_s, fact_key, fact_value, embedding, scope) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (record.text, record.mtype.value, record.importance, record.created,
                 record.last_used, record.uses, record.ttl_s, record.fact_key,
                 record.fact_value, emb.tobytes(), scope),
            )
            self._db.commit()
            self._enforce_bound()

    def extract_and_store(
        self, request_text: str, answer: str, scope: str = "global"
    ) -> list[MemoryRecord]:
        """Capture the exchange as episodic memory and extract durable semantic
        facts. Uses an LLM extractor when `llm` is set and enabled in config;
        otherwise a heuristic. Falls back to the heuristic on any LLM failure."""
        new: list[MemoryRecord] = []
        episodic = MemoryRecord(
            text=f"User asked: {request_text.strip()[:200]}",
            mtype=MemoryType.EPISODIC,
            importance=0.4,
        )
        self.store(episodic, scope=scope)
        new.append(episodic)

        facts = None
        if self.llm is not None and self.cfg.llm_memory_extraction:
            facts = self._llm_extract(request_text, answer)
        if facts is None:  # no LLM, disabled, or extraction failed
            facts = self._heuristic_extract(request_text)

        for rec in facts:
            self.store(rec, scope=scope)
            new.append(rec)
        return new

    def _heuristic_extract(self, request_text: str) -> list[MemoryRecord]:
        low = request_text.lower()
        for verb in (" uses ", " prefers ", " likes "):
            if verb in low:
                subj, _, obj = request_text.partition(verb)
                return [MemoryRecord(
                    text=request_text.strip(), mtype=MemoryType.SEMANTIC, importance=0.7,
                    fact_key=f"{subj.strip().lower()}{verb.strip()}",
                    fact_value=obj.strip().rstrip(".").split(".")[0][:60])]
        return []

    def _llm_extract(self, request_text: str, answer: str) -> list[MemoryRecord] | None:
        """Extract durable user facts via the LLM. Returns None on any failure so
        the caller can fall back to the heuristic."""
        prompt = (
            "From the exchange, extract durable facts about the USER worth "
            "remembering (preferences, identity, environment, decisions). Return "
            'ONLY a JSON array; each item {"fact": str, "key": snake_case str, '
            '"value": str, "importance": 0..1}. Return [] if nothing durable.\n\n'
            f"User: {request_text.strip()[:1000]}\nAssistant: {answer.strip()[:1000]}"
        )
        try:
            raw = self.llm(prompt)
            start, end = raw.find("["), raw.rfind("]")
            if start < 0 or end < 0:
                return None
            items = json.loads(raw[start:end + 1])
            out: list[MemoryRecord] = []
            for it in items:
                if not isinstance(it, dict) or not it.get("fact"):
                    continue
                out.append(MemoryRecord(
                    text=str(it["fact"])[:200], mtype=MemoryType.SEMANTIC,
                    importance=float(it.get("importance", 0.6)),
                    fact_key=(str(it["key"])[:60] if it.get("key") else None),
                    fact_value=(str(it["value"])[:120] if it.get("value") else None)))
            return out
        except Exception:
            return None

    def forget(self, where_sql: str, params: tuple = ()) -> int:
        with self._lock:
            cur = self._db.execute(f"DELETE FROM memory WHERE {where_sql}", params)
            self._db.commit()
            return cur.rowcount

    def expire(self) -> int:
        now = time.time()
        return self.forget("ttl_s IS NOT NULL AND (? - created) > ttl_s", (now,))

    # --- read side ---------------------------------------------------------
    def retrieve(
        self, query: str, top_k: int | None = None, scope: str = "global"
    ) -> list[ContextItem]:
        top_k = top_k or self.cfg.memory_read_k
        with self._lock:
            self.expire()
            rows = self._db.execute(
                "SELECT id, text, mtype, importance, created, last_used, fact_key,"
                " fact_value, embedding FROM memory WHERE scope=?",
                (scope,),
            ).fetchall()
            if not rows:
                return []
            mat = np.frombuffer(b"".join(r[8] for r in rows), dtype=np.float32)
            mat = mat.reshape(len(rows), -1)
            qvec = self.embedder.encode_one(query)
            if mat.shape[1] != qvec.shape[0]:
                # embedding model changed since these memories were written —
                # degrade gracefully instead of crashing on a dim mismatch.
                return []
            sims = mat @ qvec
            now = time.time()
            scored = []
            for r, s in zip(rows, sims):
                recency = 1.0 / (1.0 + (now - r[5]) / 86400.0)
                scored.append((0.6 * float(s) + 0.25 * r[3] + 0.15 * recency, r))
            scored.sort(key=lambda x: x[0], reverse=True)

            out: list[ContextItem] = []
            used_ids = []
            for _, r in scored[:top_k]:
                used_ids.append(r[0])
                out.append(
                    ContextItem(
                        text=r[1],
                        source=Source.LONG_TERM_MEMORY,
                        timestamp=r[4],
                        importance=r[3],
                        trusted=True,
                        metadata={
                            "memory_type": r[2],
                            "fact_key": r[6],
                            "fact_value": r[7],
                            "pinned": r[3] >= 0.9,
                        },
                    )
                )
            # bump usage stats atomically
            if used_ids:
                qmarks = ",".join("?" * len(used_ids))
                self._db.execute(
                    f"UPDATE memory SET uses=uses+1, last_used=? WHERE id IN ({qmarks})",
                    (now, *used_ids),
                )
                self._db.commit()
            return out

    def count(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM memory").fetchone()[0]

    # --- bounding ---------------------------------------------------------
    def _enforce_bound(self) -> None:
        n = self.count()
        over = n - self.cfg.memory_max_records
        if over <= 0:
            return
        now = time.time()
        # evict lowest importance * recency; never evict pinned (importance>=0.9)
        rows = self._db.execute(
            "SELECT id, importance, last_used FROM memory WHERE importance < 0.9"
        ).fetchall()
        ranked = sorted(
            rows,
            key=lambda r: r[1] * (1.0 / (1.0 + (now - r[2]) / 86400.0)),
        )
        victims = [r[0] for r in ranked[:over]]
        if victims:
            qmarks = ",".join("?" * len(victims))
            self._db.execute(f"DELETE FROM memory WHERE id IN ({qmarks})", victims)
            self._db.commit()
