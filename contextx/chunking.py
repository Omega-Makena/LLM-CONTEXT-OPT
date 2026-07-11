"""Chunking — split documents into embeddable units at ingest time.

Whole-document vectors are semantically useless for anything longer than a
paragraph (the toy did this). We split on paragraph/sentence boundaries into
~`target_tokens` windows with overlap, so a chunk is a coherent, retrievable
unit and adjacent context isn't lost at the seam.
"""

from __future__ import annotations

import re

from .budget import count_tokens

_PARA = re.compile(r"\n\s*\n")
_SENT = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text: str, target_tokens: int = 320, overlap_tokens: int = 48) -> list[str]:
    """Greedy sentence-packing into ~target_tokens windows with token overlap."""
    text = text.strip()
    if not text:
        return []
    if count_tokens(text) <= target_tokens:
        return [text]

    # split into sentences, respecting paragraph breaks
    sentences: list[str] = []
    for para in _PARA.split(text):
        para = para.strip()
        if not para:
            continue
        sentences.extend(s.strip() for s in _SENT.split(para) if s.strip())

    chunks: list[str] = []
    cur: list[str] = []
    cur_tok = 0
    for sent in sentences:
        st = count_tokens(sent)
        if cur and cur_tok + st > target_tokens:
            chunks.append(" ".join(cur))
            # carry overlap: keep trailing sentences up to overlap_tokens
            back: list[str] = []
            back_tok = 0
            for s in reversed(cur):
                stk = count_tokens(s)
                if back_tok + stk > overlap_tokens:
                    break
                back.insert(0, s)
                back_tok += stk
            cur = back
            cur_tok = back_tok
        cur.append(sent)
        cur_tok += st
    if cur:
        chunks.append(" ".join(cur))
    return chunks
