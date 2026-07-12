"""Document loaders — parse real file formats into `Document`s (#7).

The engine takes clean text; real corpora are files. These loaders turn
txt/markdown/html/pdf into `Document`s (stable doc_id = file path). PDF/HTML
parsers are optional deps with graceful fallbacks:

    pip install "contextx[ingest]"   # pypdf + beautifulsoup4

Unsupported types raise; `load_directory` skips them.
"""

from __future__ import annotations

import re
from pathlib import Path

from .types import Document, Source


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def load_html(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        from bs4 import BeautifulSoup

        return BeautifulSoup(raw, "html.parser").get_text(" ", strip=True)
    except Exception:
        # crude fallback: drop script/style, strip tags
        raw = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.S | re.I)
        return re.sub(r"<[^>]+>", " ", raw)


def load_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "PDF support needs pypdf. Install with: pip install 'contextx[ingest]'"
        ) from exc
    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


LOADERS = {
    ".txt": load_text, ".text": load_text, ".md": load_text, ".markdown": load_text,
    ".html": load_html, ".htm": load_html,
    ".pdf": load_pdf,
}


def load_file(
    path, tenant_id: str = "default", acl: list[str] | None = None,
    source: Source = Source.DOCUMENT,
) -> Document:
    p = Path(path)
    loader = LOADERS.get(p.suffix.lower())
    if loader is None:
        raise ValueError(f"unsupported file type: {p.suffix!r}")
    return Document(
        text=loader(p),
        doc_id=str(p),
        source=source,
        tenant_id=tenant_id,
        acl=acl or [],
        metadata={"path": str(p), "filename": p.name},
    )


def load_directory(
    path, glob: str = "**/*", tenant_id: str = "default", acl: list[str] | None = None,
) -> list[Document]:
    """Load every supported file under `path`. Unreadable/unsupported files skipped."""
    docs: list[Document] = []
    for p in sorted(Path(path).glob(glob)):
        if p.is_file() and p.suffix.lower() in LOADERS:
            try:
                docs.append(load_file(p, tenant_id=tenant_id, acl=acl))
            except Exception:
                continue
    return docs
