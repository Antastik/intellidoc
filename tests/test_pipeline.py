import tempfile
from pathlib import Path

from app.providers.testing import EchoLLM, HashingEmbeddings
from app.rag import RAGPipeline, VectorStore
from app.rag.ingest import chunk_pages


def _build_pipeline(tmp):
    embedder = HashingEmbeddings(dim=128)
    store = VectorStore(dim=embedder.dim, data_dir=Path(tmp))
    corpus = [
        ("leave.txt", "Employees get 20 days of paid annual leave each year."),
        ("security.txt", "Hard-coding API keys in source code is strictly prohibited."),
    ]
    for i, (src, text) in enumerate(corpus):
        chunks = chunk_pages(src, [text], 900, 150, doc_id=f"d{i}")
        store.add(chunks, embedder.embed([c.text for c in chunks]))
    return RAGPipeline(store, embedder, EchoLLM(), top_k=2)


def test_retrieval_returns_relevant_citation():
    with tempfile.TemporaryDirectory() as tmp:
        pipe = _build_pipeline(tmp)
        resp = pipe.answer("How many days of annual leave?")
        assert resp.citations, "expected at least one citation"
        top = resp.citations[0]
        assert top.source == "leave.txt"
        assert top.score > 0
        assert "20 days" in top.snippet


def test_empty_store_is_handled():
    with tempfile.TemporaryDirectory() as tmp:
        embedder = HashingEmbeddings(dim=64)
        store = VectorStore(dim=embedder.dim, data_dir=Path(tmp))
        pipe = RAGPipeline(store, embedder, EchoLLM(), top_k=3)
        resp = pipe.answer("anything?")
        assert resp.citations == []


def test_persistence_reloads_index():
    with tempfile.TemporaryDirectory() as tmp:
        _build_pipeline(tmp)  # writes index to disk
        # New store over same dir should see the persisted chunks.
        store2 = VectorStore(dim=128, data_dir=Path(tmp))
        assert store2.total_chunks == 2
        assert len(store2.documents()) == 2
