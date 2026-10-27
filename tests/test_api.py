"""End-to-end API regression tests via FastAPI's TestClient (no API keys)."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi.testclient import TestClient

from .conftest import make_pdf, make_settings


def _upload(client: TestClient, name: str, pages: list[str]):
    return client.post(
        "/api/upload", files=[("files", (name, make_pdf(pages), "application/pdf"))]
    )


def test_upload_list_query_roundtrip(client: TestClient):
    r = _upload(client, "leave.pdf", ["Employees get 20 days of paid annual leave.", "Second page."])
    assert r.status_code == 200, r.text
    doc = r.json()["documents"][0]
    assert doc["source"] == "leave.pdf" and doc["pages"] == 2 and doc["chunks"] == 2

    docs = client.get("/api/documents").json()
    assert [d["doc_id"] for d in docs] == [doc["doc_id"]]
    assert client.get("/api/health").json()["chunks"] == 2

    q = client.post("/api/query", json={"question": "How many days of annual leave?"})
    assert q.status_code == 200, q.text
    body = q.json()
    assert body["citations"][0]["source"] == "leave.pdf"
    assert body["citations"][0]["page"] == 1
    assert "20 days" in body["citations"][0]["snippet"]
    assert body["answer"].startswith("[echo]")


def test_invalid_and_scanned_pdfs_are_rejected_without_mutating_store(client: TestClient):
    r = client.post("/api/upload", files=[("files", ("bad.pdf", b"not a pdf", "application/pdf"))])
    assert r.status_code == 422

    r = _upload(client, "scanned.pdf", ["", ""])
    assert r.status_code == 422
    assert "No extractable text" in r.json()["detail"]

    r = client.post("/api/upload", files=[("files", ("notes.txt", b"hello", "text/plain"))])
    assert r.status_code == 400

    assert client.get("/api/health").json()["chunks"] == 0


def test_multipart_batch_is_all_or_nothing(client: TestClient):
    files = [
        ("files", ("good.pdf", make_pdf(["Good document text."]), "application/pdf")),
        ("files", ("bad.pdf", b"garbage", "application/pdf")),
    ]
    r = client.post("/api/upload", files=files)
    assert r.status_code == 422
    assert client.get("/api/health").json()["chunks"] == 0
    assert client.get("/api/documents").json() == []


def test_empty_corpus_vs_unmatched_doc_filter(client: TestClient):
    r = client.post("/api/query", json={"question": "anything?"})
    assert r.status_code == 200
    assert r.json()["citations"] == []
    assert "any documents indexed" in r.json()["answer"].lower()

    _upload(client, "a.pdf", ["Alpha content here."])
    r = client.post("/api/query", json={"question": "alpha?", "doc_ids": ["does-not-exist"]})
    assert r.status_code == 200
    assert r.json()["citations"] == []
    assert "selected document" in r.json()["answer"].lower()


def test_doc_filter_finds_results_beyond_overfetch_window(client: TestClient):
    question = "annual leave days policy"
    for i in range(30):
        _upload(client, f"decoy{i}.pdf", [question])
    target = _upload(
        client, "target.pdf", ["Annual leave: staff receive twenty days plus bank holidays."]
    ).json()["documents"][0]["doc_id"]

    r = client.post(
        "/api/query", json={"question": question, "top_k": 1, "doc_ids": [target]}
    )
    assert r.status_code == 200, r.text
    cits = r.json()["citations"]
    assert len(cits) == 1 and cits[0]["doc_id"] == target


class _ExplodingEmbeddings:
    name = "exploding"
    model = "boom"
    dim = 8

    def embed(self, texts):
        raise RuntimeError("upstream 401: invalid api key sk-SUPERSECRET123")

    def embed_query(self, text):
        return self.embed([text])[0]


class _EmptyLLM:
    name = "empty"
    model = "empty"

    def complete(self, system, user):
        return "   "


def test_provider_failure_returns_safe_json_502(tmp_path: Path):
    from app.main import create_app

    app = create_app(make_settings(tmp_path))
    with TestClient(app) as c:
        _upload(c, "a.pdf", ["Some text."])
        app.state.pipeline.embedder = _ExplodingEmbeddings()
        r = c.post("/api/query", json={"question": "hi"})
    assert r.status_code == 502
    body = r.json()
    assert body["error"] == "provider_error"
    assert body["stage"] == "query"
    assert isinstance(body["detail"], str)
    assert "SUPERSECRET" not in r.text and "RuntimeError" not in r.text


def test_upload_embedding_failure_returns_502_and_store_untouched(tmp_path: Path):
    from app.main import create_app

    app = create_app(make_settings(tmp_path))
    app.state.embedder = _ExplodingEmbeddings()
    with TestClient(app) as c:
        r = _upload(c, "a.pdf", ["Some text."])
        assert r.status_code == 502
        assert r.json()["stage"] == "embedding"
        assert "SUPERSECRET" not in r.text
        assert c.get("/api/health").json()["chunks"] == 0


def test_empty_llm_answer_is_not_a_silent_success(tmp_path: Path):
    from app.main import create_app

    app = create_app(make_settings(tmp_path))
    app.state.pipeline.llm = _EmptyLLM()
    with TestClient(app) as c:
        _upload(c, "a.pdf", ["Some text."])
        r = c.post("/api/query", json={"question": "what?"})
    assert r.status_code == 502
    assert r.json()["error"] == "provider_error"


def test_concurrent_uploads_and_queries(client: TestClient):
    _upload(client, "seed.pdf", ["Seed text about cats."])

    def up(i: int):
        return _upload(client, f"doc{i}.pdf", [f"Document {i} about dogs.", f"Page two {i}."]).status_code

    def ask(i: int):
        return client.post("/api/query", json={"question": f"dogs {i}?"}).status_code

    with ThreadPoolExecutor(max_workers=8) as ex:
        ups = list(ex.map(up, range(8)))
        asks = list(ex.map(ask, range(8)))
    assert ups == [200] * 8 and asks == [200] * 8
    h = client.get("/api/health").json()
    assert h["documents"] == 9 and h["chunks"] == 1 + 8 * 2
