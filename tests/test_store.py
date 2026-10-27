import math
import pickle
from pathlib import Path

import pytest

from app.providers.testing import HashingEmbeddings
from app.rag import VectorStore
from app.rag.ingest import chunk_pages


def _chunks(src: str, text: str, doc_id: str):
    return chunk_pages(src, [text], 900, 150, doc_id=doc_id)


def test_add_rejects_wrong_dimension_without_mutating(tmp_path: Path):
    store = VectorStore(dim=8, data_dir=tmp_path)
    with pytest.raises(ValueError, match="dimension"):
        store.add(_chunks("a", "hello", "d1"), [[1.0] * 7])
    assert store.total_chunks == 0
    assert not (tmp_path / "index.faiss").exists()


def test_add_rejects_count_mismatch_and_non_finite(tmp_path: Path):
    store = VectorStore(dim=4, data_dir=tmp_path)
    with pytest.raises(ValueError, match="count"):
        store.add(_chunks("a", "hello", "d1"), [[1.0] * 4, [1.0] * 4])
    with pytest.raises(ValueError, match="finite"):
        store.add(_chunks("a", "hello", "d1"), [[math.nan, 0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="finite"):
        store.add(_chunks("a", "hello", "d1"), [[math.inf, 0.0, 0.0, 0.0]])
    assert store.total_chunks == 0


def test_persisted_dimension_mismatch_is_refused_not_deleted(tmp_path: Path):
    emb = HashingEmbeddings(dim=128)
    store = VectorStore(dim=128, data_dir=tmp_path)
    ch = _chunks("a.txt", "some text", "d1")
    store.add(ch, emb.embed([c.text for c in ch]))

    with pytest.raises(ValueError, match="dimension"):
        VectorStore(dim=64, data_dir=tmp_path)

    assert (tmp_path / "index.faiss").exists() and (tmp_path / "meta.pkl").exists()
    assert VectorStore(dim=128, data_dir=tmp_path).total_chunks == 1


def test_persisted_index_meta_count_mismatch_is_refused(tmp_path: Path):
    emb = HashingEmbeddings(dim=16)
    store = VectorStore(dim=16, data_dir=tmp_path)
    ch = _chunks("a.txt", "some text", "d1")
    store.add(ch, emb.embed([c.text for c in ch]))
    (tmp_path / "meta.pkl").write_bytes(pickle.dumps([]))
    with pytest.raises(ValueError, match="count"):
        VectorStore(dim=16, data_dir=tmp_path)


def test_search_rejects_wrong_query_dimension(tmp_path: Path):
    store = VectorStore(dim=4, data_dir=tmp_path)
    store.add(_chunks("a", "hello", "d1"), [[1.0, 0, 0, 0]])
    with pytest.raises(ValueError, match="dimension"):
        store.search([1.0, 0.0], top_k=1)


def test_filtered_search_beyond_overfetch_window(tmp_path: Path):
    store = VectorStore(dim=4, data_dir=tmp_path)
    for i in range(30):
        store.add(_chunks("decoy", "decoy", f"decoy{i}"), [[1.0, 0, 0, 0]])
    store.add(_chunks("target", "target", "target"), [[0.5, 0.5, 0, 0]])

    hits = store.search([1.0, 0, 0, 0], top_k=1, doc_ids=["target"])
    assert len(hits) == 1 and hits[0][0]["doc_id"] == "target"

    assert store.search([1.0, 0, 0, 0], top_k=3, doc_ids=["nope"]) == []
    assert store.has_any_doc(["nope"]) is False
    assert store.has_any_doc(["target", "nope"]) is True
