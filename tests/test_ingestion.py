"""Tests for ingestion: structure-aware chunking, loaders, incremental sync."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contextx import Config, ContextEngine  # noqa: E402
from contextx.chunking import chunk_text  # noqa: E402
from contextx.loaders import load_directory, load_file  # noqa: E402
from contextx.sync import DirectorySync  # noqa: E402

MD = """# Guide

Intro sentence about the setup.

## Install

Run the installer. Then configure the settings.

```
pip install foo
bar --init
```

## Usage

Call the API with your key.
"""


def test_structured_chunking_keeps_headings_and_code():
    chunks = chunk_text(MD, target_tokens=1000, overlap_tokens=0)
    # heading breadcrumb is prepended so a chunk keeps its context
    assert any(c.startswith("Guide > Install") for c in chunks)
    assert any(c.startswith("Guide > Usage") for c in chunks)
    # the fenced code block survives intact inside a single chunk
    assert any("pip install foo\nbar --init" in c for c in chunks)


def test_chunking_plaintext_fallback():
    long = ". ".join(f"Sentence {i} about tokens and things" for i in range(120)) + "."
    chunks = chunk_text(long, target_tokens=80, overlap_tokens=15)
    assert len(chunks) > 1


def test_loaders_txt_md_html(tmp_path):
    (tmp_path / "a.txt").write_text("plain text file", encoding="utf-8")
    (tmp_path / "b.md").write_text("# Title\n\nmarkdown body", encoding="utf-8")
    (tmp_path / "c.html").write_text(
        "<html><body><h1>Hi</h1><p>Hello world</p></body></html>", encoding="utf-8")
    docs = load_directory(str(tmp_path))
    assert len(docs) == 3
    html_doc = load_file(tmp_path / "c.html")
    assert "Hello world" in html_doc.text
    assert "<p>" not in html_doc.text  # tags stripped


def test_sync_add_update_delete(tmp_path):
    cfg = Config(index_dir=str(tmp_path / "idx"), memory_db_path=str(tmp_path / "m.db"))
    engine = ContextEngine(config=cfg)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# A\nalpha content about widgets", encoding="utf-8")
    (docs / "b.txt").write_text("beta content about gadgets", encoding="utf-8")

    sync = DirectorySync(engine, manifest_path=str(tmp_path / "manifest.json"))
    assert sync.sync(str(docs))["added"] == 2
    # unchanged -> no-op
    assert sync.sync(str(docs)) == {"added": 0, "updated": 0, "deleted": 0}
    # modify one file -> update
    (docs / "a.md").write_text("# A\nalpha content about sprockets now", encoding="utf-8")
    assert sync.sync(str(docs))["updated"] == 1
    # remove one file -> delete
    (docs / "b.txt").unlink()
    assert sync.sync(str(docs))["deleted"] == 1
