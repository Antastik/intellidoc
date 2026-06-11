"""Local / offline providers: sentence-transformers for embeddings and a local
Ollama server for generation. Both dependencies are imported lazily."""
from __future__ import annotations

from .base import EmbeddingProvider, LLMProvider


class LocalEmbeddings(EmbeddingProvider):
    name = "local"

    def __init__(self, model: str):
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(model)
        self.model = model
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vecs]


class LocalLLM(LLMProvider):
    name = "local"

    def __init__(self, base_url: str, model: str):
        self._base_url = base_url.rstrip("/")
        self.model = model

    def complete(self, system: str, user: str) -> str:
        import httpx  # lazy import

        resp = httpx.post(
            f"{self._base_url}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "options": {"temperature": 0.1},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
