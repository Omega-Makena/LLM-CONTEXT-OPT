"""Incremental sync — keep the index in step with a directory of files (#9).

Diffs the current files against a content-hash manifest and applies the minimal
set of changes: new files -> ingest, changed files -> update (delete+re-embed),
removed files -> delete. This is what keeps a live corpus from going stale
without re-embedding everything each run.

    from contextx import ContextEngine
    from contextx.sync import DirectorySync

    sync = DirectorySync(ContextEngine())
    print(sync.sync("docs/"))   # {'added': 3, 'updated': 1, 'deleted': 0}

doc_id is the file path, so update/delete line up with what was ingested.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .loaders import LOADERS, load_file
from .pipeline import ContextEngine


def _content_hash(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


class DirectorySync:
    def __init__(self, engine: ContextEngine, manifest_path: str | None = None) -> None:
        self.engine = engine
        self.manifest_path = Path(
            manifest_path or (Path(engine.store.dir) / "sync_manifest.json")
        )
        self.manifest: dict[str, str] = {}
        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def sync(
        self, directory, glob: str = "**/*",
        tenant_id: str = "default", acl: list[str] | None = None,
    ) -> dict[str, int]:
        added = updated = deleted = 0
        current: dict[str, str] = {}
        seen: set[str] = set()

        files = [
            p for p in sorted(Path(directory).glob(glob))
            if p.is_file() and p.suffix.lower() in LOADERS
        ]
        for p in files:
            key = str(p)
            seen.add(key)
            h = _content_hash(p)
            current[key] = h
            if self.manifest.get(key) == h:
                continue  # unchanged
            doc = load_file(p, tenant_id=tenant_id, acl=acl)
            if key in self.manifest:
                self.engine.update([doc])
                updated += 1
            else:
                self.engine.ingest([doc])
                added += 1

        # files that vanished from disk -> remove from the index
        for key in list(self.manifest):
            if key not in seen:
                self.engine.delete(key)
                deleted += 1

        self.manifest = current
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")
        return {"added": added, "updated": updated, "deleted": deleted}
