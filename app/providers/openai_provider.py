"""OpenAI-backed providers. The SDK is imported lazily so the app runs even
when the openai package is not installed (as long as you don't select it)."""
from __future__ import annotations

from .base import EmbeddingProvider, LLMProvider


class OpenAIEmbeddings(EmbeddingProvider):
    name = "openai"

    def __init__(self, api_key: str, model: str):
        from openai import OpenAI  # lazy import

        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for the OpenAI embedding provider.")
        self._client = OpenAI(api_key=api_key)
        self.model = model
        # text-embedding-3-small -> 1536 dims; we discover it from the first call instead
        self.dim = 1536

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self.model, input=texts)
        vecs = [d.embedding for d in resp.data]
        if vecs:
            self.dim = len(vecs[0])
        return vecs


class OpenAILLM(LLMProvider):
    name = "openai"

    def __init__(self, api_key: str, model: str):
        from openai import OpenAI  # lazy import

        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for the OpenAI LLM provider.")
        self._client = OpenAI(api_key=api_key)
        self.model = model

    def complete(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""
