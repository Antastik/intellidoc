"""Google Gemini providers. SDK imported lazily."""
from __future__ import annotations

from .base import EmbeddingProvider, LLMProvider


class GeminiEmbeddings(EmbeddingProvider):
    name = "gemini"

    def __init__(self, api_key: str, model: str):
        import google.generativeai as genai  # lazy import

        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for the Gemini embedding provider.")
        genai.configure(api_key=api_key)
        self._genai = genai
        self.model = model
        self.dim = 768

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs: list[list[float]] = []
        for t in texts:
            r = self._genai.embed_content(model=self.model, content=t)
            vecs.append(r["embedding"])
        if vecs:
            self.dim = len(vecs[0])
        return vecs


class GeminiLLM(LLMProvider):
    name = "gemini"

    def __init__(self, api_key: str, model: str):
        import google.generativeai as genai  # lazy import

        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for the Gemini LLM provider.")
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model)
        self.model = model

    def complete(self, system: str, user: str) -> str:
        resp = self._model.generate_content(f"{system}\n\n{user}")
        return resp.text or ""
