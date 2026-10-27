"""Shared test helpers: minimal PDF builder and an isolated app factory.

Everything here runs with the `testing` providers and a temporary DATA_DIR, so
no API keys, network access, or model downloads are needed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic_settings import SettingsConfigDict

from app.config import Settings


def make_pdf(pages: list[str]) -> bytes:
    """Build a tiny but valid PDF with one text line per page (Helvetica).

    pypdf's `extract_text` reads this back, which is all the upload tests need.
    An empty string yields a page with no text (simulates a scanned page).
    """
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    page_ids: list[int] = []
    pages_obj_num = len(objects) + 1 + 2 * len(pages)  # reserved after pages/streams
    for text in pages:
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = (
            f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1")
            if text
            else b""
        )
        content = add(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        page = add(
            (
                f"<< /Type /Page /Parent {pages_obj_num} 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 {font} 0 R >> >> /Contents {content} 0 R >>"
            ).encode()
        )
        page_ids.append(page)
    kids = " ".join(f"{p} 0 R" for p in page_ids)
    pages_obj = add(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode())
    assert pages_obj == pages_obj_num
    catalog = add(f"<< /Type /Catalog /Pages {pages_obj} 0 R >>".encode())

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog} 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n"
    ).encode()
    return bytes(out)


class IsolatedSettings(Settings):
    """Settings that ignore any local `.env` so tests never pick up real keys."""

    model_config = SettingsConfigDict(env_file=None, extra="ignore")


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    kwargs: dict[str, Any] = dict(
        llm_provider="testing",
        embedding_provider="testing",
        data_dir=str(tmp_path / "data"),
    )
    kwargs.update(overrides)
    return IsolatedSettings(**kwargs)


@pytest.fixture
def client(tmp_path: Path):
    from fastapi.testclient import TestClient

    from app.main import create_app

    app = create_app(make_settings(tmp_path))
    with TestClient(app) as c:
        yield c
