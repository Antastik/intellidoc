"""Google Gemini providers on the `google-genai` SDK (successor to the
deprecated `generativeai` package). The SDK is imported lazily; a pre-built
client can be injected for tests."""
from __future__ import annotations

from typing import Any

from .base import EmbeddingProvider, LLMProvider, ProviderError

# Gemini API limit on `contents` per embed_content request.
_EMBED_BATCH = 100


def _client(api_key: str, injected: Any | None) -> Any:
    if injected is not None:
        return injected
    from google import genai  # lazy import

    return genai.Client(api_key=api_key)


class GeminiEmbeddings(EmbeddingProvider):
    name = "gemini"

    def __init__(self, api_key: str, model: str, dim: int = 768, client: Any | None = None):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for the Gemini embedding provider.")
        if dim <= 0:
            raise ValueError("Gemini embedding dimension must be positive.")
        self._client = _client(api_key, client)
        self.model = model
        self.dim = dim

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), _EMBED_BATCH):
            batch = texts[start : start + _EMBED_BATCH]
            resp = self._client.models.embed_content(
                model=self.model,
                contents=batch,
                config={"task_type": task_type, "output_dimensionality": self.dim},
            )
            embeddings = getattr(resp, "embeddings", None) or []
            if len(embeddings) != len(batch):
                raise ProviderError(
                    f"Gemini returned {len(embeddings)} embeddings for {len(batch)} inputs."
                )
            for e in embeddings:
                values = getattr(e, "values", None)
                if not values or len(values) != self.dim:
                    raise ProviderError(
                        f"Gemini embedding has {len(values or [])} dims, expected {self.dim}."
                    )
                out.append([float(v) for v in values])
        return out

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


class GeminiLLM(LLMProvider):
    name = "gemini"

    def __init__(self, api_key: str, model: str, client: Any | None = None):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for the Gemini LLM provider.")
        self._client = _client(api_key, client)
        self.model = model

    def complete(self, system: str, user: str) -> str:
        resp = self._client.models.generate_content(
            model=self.model,
            contents=user,
            config={"system_instruction": system, "temperature": 0.1},
        )
        text = getattr(resp, "text", None)
        if text and text.strip():
            return text
        raise ProviderError(f"Gemini returned no text ({_why_empty(resp)}).")


def _why_empty(resp: Any) -> str:
    feedback = getattr(resp, "prompt_feedback", None)
    block = getattr(feedback, "block_reason", None)
    if block:
        return f"prompt blocked: {block}"
    candidates = getattr(resp, "candidates", None) or []
    reason = getattr(candidates[0], "finish_reason", None) if candidates else None
    return f"finish_reason={reason}" if reason else "no candidates"
