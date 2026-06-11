"""A small persistent vector store built on FAISS (inner-product on normalised
vectors = cosine similarity) plus a parallel list of chunk metadata.

Persistence is just two files in DATA_DIR: `index.faiss` and `meta.pkl`. This is
deliberately simple — swap in pgvector/Qdrant later without touching the pipeline.
"""
from __future__ import annotations

import pickle
import threading
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .ingest import Chunk


class VectorStore:
    def __init__(self, dim: int, data_dir: Path):
        import faiss  # lazy import

        self._faiss = faiss
        self.dim = dim
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.data_dir / "index.faiss"
        self._meta_path = self.data_dir / "meta.pkl"
        self._lock = threading.Lock()
        self._index = faiss.IndexFlatIP(dim)
        self._meta: list[dict] = []
        self._load()

    # ---- persistence ----
    def _load(self) -> None:
        if self._index_path.exists() and self._meta_path.exists():
            self._index = self._faiss.read_index(str(self._index_path))
            with open(self._meta_path, "rb") as f:
                self._meta = pickle.load(f)

    def _save(self) -> None:
        self._faiss.write_index(self._index, str(self._index_path))
        with open(self._meta_path, "wb") as f:
            pickle.dump(self._meta, f)

    # ---- helpers ----
    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    # ---- public API ----
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        vecs = self._normalize(np.asarray(embeddings, dtype="float32"))
        with self._lock:
            self._index.add(vecs)
            for ch in chunks:
                self._meta.append(asdict(ch))
            self._save()

    def search(
        self, query_vec: list[float], top_k: int, doc_ids: list[str] | None = None
    ) -> list[tuple[dict, float]]:
        if self._index.ntotal == 0:
            return []
        q = self._normalize(np.asarray([query_vec], dtype="float32"))
        # Over-fetch when filtering so the post-filter still returns top_k.
        fetch = top_k if not doc_ids else min(self._index.ntotal, top_k * 5)
        scores, idxs = self._index.search(q, fetch)
        results: list[tuple[dict, float]] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            meta = self._meta[idx]
            if doc_ids and meta["doc_id"] not in doc_ids:
                continue
            results.append((meta, float(score)))
            if len(results) >= top_k:
                break
        return results

    def documents(self) -> list[dict]:
        """Aggregate chunk metadata into per-document summaries."""
        docs: dict[str, dict] = {}
        for m in self._meta:
            d = docs.setdefault(
                m["doc_id"],
                {"doc_id": m["doc_id"], "source": m["source"], "pages": 0, "chunks": 0},
            )
            d["chunks"] += 1
            if m.get("page"):
                d["pages"] = max(d["pages"], m["page"])
        return list(docs.values())

    @property
    def total_chunks(self) -> int:
        return len(self._meta)
