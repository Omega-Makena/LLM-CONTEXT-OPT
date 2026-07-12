"""Chunking — split documents into embeddable units at ingest time (#8).

Whole-document vectors are useless; naive fixed-window splitting shreds
structure. This chunker is structure-aware:

  * splits on Markdown headings, and prepends the heading breadcrumb (e.g.
    "Title > Section") to every chunk in that section, so a chunk keeps its
    context even when retrieved in isolation.
  * treats fenced code blocks (```...```) as atomic — never split mid-code.
  * within a section, greedily packs prose sentences into ~target_tokens windows
    with token overlap so adjacent context isn't lost at the seam.

Plain text with no headings degrades gracefully to sentence-packing.
"""

from __future__ import annotations

import re

from .budget import count_tokens

_PARA = re.compile(r"\n\s*\n")
_SENT = re.compile(r"(?<=[.!?])\s+")
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_CODE = re.compile(r"(```.*?```)", re.S)


def chunk_text(text: str, target_tokens: int = 320, overlap_tokens: int = 48) -> list[str]:
    text = text.strip()
    if not text:
        return []
    chunks: list[str] = []
    for heading, body in _split_headings(text):
        prefix = f"{heading}\n" if heading else ""
        for packed in _pack(body, target_tokens, overlap_tokens):
            chunks.append(prefix + packed)
    return chunks


def _split_headings(text: str) -> list[tuple[str, str]]:
    """Split into (heading_breadcrumb, body) sections on Markdown headings."""
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    cur: list[str] = []

    def flush() -> None:
        if any(line.strip() for line in cur):
            trail = " > ".join(title for _, title in stack)
            sections.append((trail, "\n".join(cur).strip()))

    for line in text.split("\n"):
        m = _HEADING.match(line)
        if m:
            flush()
            cur = []
            level, title = len(m.group(1)), m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            cur.append(line)
    flush()
    return sections or [("", text)]


def _units(body: str) -> list[str]:
    """Break a section into atomic units: whole code blocks + prose sentences."""
    units: list[str] = []
    for part in _CODE.split(body):
        if part.startswith("```"):
            units.append(part.strip())
        else:
            for para in _PARA.split(part):
                para = para.strip()
                if para:
                    units.extend(s.strip() for s in _SENT.split(para) if s.strip())
    return units


def _pack(body: str, target_tokens: int, overlap_tokens: int) -> list[str]:
    body = body.strip()
    if not body:
        return []
    if count_tokens(body) <= target_tokens:
        return [body]

    units = _units(body)
    chunks: list[str] = []
    cur: list[str] = []
    cur_tok = 0
    for unit in units:
        ut = count_tokens(unit)
        if cur and cur_tok + ut > target_tokens:
            chunks.append(" ".join(cur))
            # carry trailing units up to overlap_tokens for continuity
            back: list[str] = []
            back_tok = 0
            for u in reversed(cur):
                utk = count_tokens(u)
                if back_tok + utk > overlap_tokens:
                    break
                back.insert(0, u)
                back_tok += utk
            cur, cur_tok = back, back_tok
        cur.append(unit)
        cur_tok += ut
    if cur:
        chunks.append(" ".join(cur))
    return chunks
