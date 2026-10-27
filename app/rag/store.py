"""A small persistent vector store built on FAISS (inner-product on normalised
vectors = cosine similarity) plus a parallel list of chunk metadata.

Persistence is just two files in DATA_DIR: `index.faiss` and `meta.pkl`. This is
deliberately simple — swap in pgvector/Qdrant later without touching the pipeline.

Invariants enforced here (never silently repaired): every vector has exactly
`dim` finite components, and `index.ntotal == len(meta)` on disk and in memory.
"""
from __future__ import annotations

import pickle
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .ingest import Chunk


class VectorStore:
    def __init__(self, dim: int, data_dir: Path):
        import faiss  # lazy import

        if dim <= 0:
            raise ValueError(f"Vector dimension must be positive, got {dim}.")
        self._faiss: Any = faiss
        self.dim = dim
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.data_dir / "index.faiss"
        self._meta_path = self.data_dir / "meta.pkl"
        self._lock = threading.RLock()
        self._index: Any = faiss.IndexFlatIP(dim)
        self._meta: list[dict[str, Any]] = []
        self._load()

    # ---- persistence ----
    def _load(self) -> None:
        if not (self._index_path.exists() and self._meta_path.exists()):
            return
        index = self._faiss.read_index(str(self._index_path))
        with open(self._meta_path, "rb") as f:
            meta = pickle.load(f)
        if index.d != self.dim:
            raise ValueError(
                f"Persisted index in {self.data_dir} has dimension {index.d} but the "
                f"configured embedding provider produces {self.dim}. Re-index with a "
                "fresh DATA_DIR (or delete the old index) after changing embedding models."
            )
        if not isinstance(meta, list) or index.ntotal != len(meta):
            raise ValueError(
                f"Persisted index in {self.data_dir} is inconsistent: vector count "
                f"{index.ntotal} != metadata count {len(meta) if isinstance(meta, list) else '?'}. "
                "Refusing to load; re-index with a fresh DATA_DIR."
            )
        self._index = index
        self._meta = meta

    def _save(self) -> None:
        self._faiss.write_index(self._index, str(self._index_path))
        with open(self._meta_path, "wb") as f:
            pickle.dump(self._meta, f)

    # ---- helpers ----
    @staticmethod
    def _normalize(vectors: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def _as_matrix(
        self, embeddings: list[list[float]], expected_rows: int
    ) -> npt.NDArray[np.float32]:
        if len(embeddings) != expected_rows:
            raise ValueError(
                f"Embedding count {len(embeddings)} does not match chunk count {expected_rows}."
            )
        try:
            vecs = np.asarray(embeddings, dtype="float32")
        except (TypeError, ValueError) as e:
            raise ValueError("Embeddings must be a rectangular list of numbers.") from e
        if vecs.ndim != 2 or vecs.shape[1] != self.dim:
            got = vecs.shape[1] if vecs.ndim == 2 else "ragged"
            raise ValueError(f"Embedding dimension {got} does not match store dimension {self.dim}.")
        if not np.isfinite(vecs).all():
            raise ValueError("Embeddings must contain only finite values (no NaN/inf).")
        return self._normalize(vecs)

    def _ids_for_docs(self, doc_ids: list[str]) -> npt.NDArray[np.int64]:
        wanted = set(doc_ids)
        return np.fromiter(
            (i for i, m in enumerate(self._meta) if m["doc_id"] in wanted), dtype="int64"
        )

    # ---- public API ----
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        vecs = self._as_matrix(embeddings, len(chunks))
        with self._lock:
            self._index.add(vecs)
            self._meta.extend(asdict(ch) for ch in chunks)
            self._save()

    def search(
        self, query_vec: list[float], top_k: int, doc_ids: list[str] | None = None
    ) -> list[tuple[dict[str, Any], float]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        q = self._as_matrix([query_vec], 1)
        with self._lock:
            if self._index.ntotal == 0:
                return []
            params = None
            if doc_ids is not None:
                candidate_ids = self._ids_for_docs(doc_ids)
                if candidate_ids.size == 0:
                    return []
                params = self._faiss.SearchParameters(
                    sel=self._faiss.IDSelectorBatch(candidate_ids)
                )
            k = min(top_k, self._index.ntotal)
            scores, idxs = self._index.search(q, k, params=params)
            return [
                (self._meta[idx], float(score))
                for score, idx in zip(scores[0], idxs[0])
                if idx >= 0
            ]

    def has_any_doc(self, doc_ids: list[str]) -> bool:
        with self._lock:
            return self._ids_for_docs(doc_ids).size > 0

    def documents(self) -> list[dict[str, Any]]:
        """Aggregate chunk metadata into per-document summaries."""
        docs: dict[str, dict[str, Any]] = {}
        with self._lock:
            meta = list(self._meta)
        for m in meta:
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
        with self._lock:
            return len(self._meta)
