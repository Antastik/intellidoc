"""Document ingestion: PDF/text -> page-aware, overlapping text chunks.

Each chunk carries its source filename and page number so the pipeline can cite
exactly where an answer came from.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field


@dataclass
class Chunk:
    doc_id: str
    source: str
    page: int | None
    chunk_index: int
    text: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex)


_WS = re.compile(r"[ \t]+")
_MULTINL = re.compile(r"\n{3,}")


def _clean(text: str) -> str:
    text = _WS.sub(" ", text)
    text = _MULTINL.sub("\n\n", text)
    return text.strip()


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Greedy splitter that prefers paragraph/sentence boundaries and keeps a
    sliding overlap so context isn't cut mid-thought."""
    text = _clean(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    # Prefer to break on paragraph, then sentence, then hard cut.
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            window = text[start:end]
            for sep in ("\n\n", ". ", "\n", " "):
                idx = window.rfind(sep)
                if idx >= chunk_size * 0.5:  # don't break too early
                    end = start + idx + len(sep)
                    break
        chunks.append(text[start:end].strip())
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


def read_pdf_pages(data: bytes) -> list[str]:
    """Return text per page. Imported lazily so non-PDF tests don't need pypdf."""
    import io

    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    return [(page.extract_text() or "") for page in reader.pages]


def chunk_pages(
    source: str,
    pages: list[str],
    chunk_size: int,
    overlap: int,
    doc_id: str | None = None,
) -> list[Chunk]:
    """Chunk a list of page texts into page-aware Chunk objects."""
    doc_id = doc_id or uuid.uuid4().hex[:12]
    chunks: list[Chunk] = []
    idx = 0
    for page_no, page_text in enumerate(pages, start=1):
        for piece in split_text(page_text, chunk_size, overlap):
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    source=source,
                    page=page_no,
                    chunk_index=idx,
                    text=piece,
                )
            )
            idx += 1
    return chunks


def ingest_pdf_bytes(
    source: str, data: bytes, chunk_size: int, overlap: int
) -> list[Chunk]:
    pages = read_pdf_pages(data)
    return chunk_pages(source, pages, chunk_size, overlap)
