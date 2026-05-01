"""Hand-written sliding-window chunker.

Strategy: paragraph-aware sliding window over characters.
- We first split on blank lines so we never break inside a paragraph if avoidable.
- We then pack paragraphs greedily into windows of ~CHUNK_SIZE characters.
- A trailing slice of CHUNK_OVERLAP characters from the previous window is
  prepended to the next window so that semantic context (entity names,
  pronouns) survives across chunk boundaries.

This keeps the implementation transparent (no LangChain/TextSplitter wrapper)
and works well for Wikipedia articles which can be tens of thousands of chars.
"""
from __future__ import annotations

import re
from typing import Iterator

from config import CHUNK_SIZE, CHUNK_OVERLAP

_PARA_SPLIT = re.compile(r"\n\s*\n")
_WS = re.compile(r"[ \t]+")


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = _WS.sub(" ", text)
    return text.strip()


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = _normalize(text)
    if not text:
        return []

    paragraphs = [p.strip() for p in _PARA_SPLIT.split(text) if p.strip()]

    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        # If a single paragraph is itself larger than `size`, hard-split it.
        # Flush any pending buf ONCE, then append the hard-split pieces directly.
        if len(para) > size:
            if buf:
                chunks.append(buf.strip())
                buf = ""
            for piece in _hard_split(para, size, overlap):
                chunks.append(piece)
            continue

        if not buf:
            buf = para
        elif len(buf) + len(para) + 1 <= size:
            buf = f"{buf}\n{para}"
        else:
            chunks.append(buf.strip())
            tail = buf[-overlap:] if overlap else ""
            buf = f"{tail}\n{para}".strip() if tail else para

    if buf.strip():
        chunks.append(buf.strip())
    return [c for c in chunks if c]


def _hard_split(text: str, size: int, overlap: int) -> Iterator[str]:
    step = max(size - overlap, 1)
    for i in range(0, len(text), step):
        yield text[i : i + size]
