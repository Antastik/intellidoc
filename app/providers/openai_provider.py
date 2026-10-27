"""OpenAI-backed providers. The SDK is imported lazily so the app runs even
when the openai package is not installed (as long as you don't select it).
A pre-built client can be injected for tests."""
from __future__ import annotations

from typing import Any

from .base import EmbeddingProvider, LLMProvider, ProviderError

# Native output sizes. text-embedding-3-* also accept a smaller `dimensions`.
_KNOWN_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}
_SUPPORTS_DIMENSIONS_PREFIX = "text-embedding-3"
# OpenAI's per-request input limit for the embeddings endpoint.
_EMBED_BATCH = 2048


def _client(api_key: str, injected: Any | None) -> Any:
    if injected is not None:
        return injected
    from openai import OpenAI  # lazy import

    return OpenAI(api_key=api_key)


class OpenAIEmbeddings(EmbeddingProvider):
    name = "openai"

    def __init__(
        self, api_key: str, model: str, dim: int | None = None, client: Any | None = None
    ):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for the OpenAI embedding provider.")
        resolved = dim or _KNOWN_DIMS.get(model)
        if not resolved or resolved <= 0:
            raise ValueError(
                f"Unknown embedding dimension for OpenAI model {model!r}; "
                "set OPENAI_EMBEDDING_DIM explicitly."
            )
        self._client = _client(api_key, client)
        self.model = model
        self.dim = resolved
        self._send_dimensions = model.startswith(_SUPPORTS_DIMENSIONS_PREFIX)

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), _EMBED_BATCH):
            batch = texts[start : start + _EMBED_BATCH]
            kwargs: dict[str, Any] = {"model": self.model, "input": batch}
            if self._send_dimensions:
                kwargs["dimensions"] = self.dim
            resp = self._client.embeddings.create(**kwargs)
            data = getattr(resp, "data", None) or []
            if len(data) != len(batch):
                raise ProviderError(
                    f"OpenAI returned {len(data)} embeddings for {len(batch)} inputs."
                )
            for item in data:
                vec = getattr(item, "embedding", None)
                if not vec or len(vec) != self.dim:
                    raise ProviderError(
                        f"OpenAI embedding has {len(vec or [])} dims, expected {self.dim}."
                    )
                out.append([float(v) for v in vec])
        return out


class OpenAILLM(LLMProvider):
    name = "openai"

    def __init__(self, api_key: str, model: str, client: Any | None = None):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for the OpenAI LLM provider.")
        self._client = _client(api_key, client)
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
        choices = getattr(resp, "choices", None) or []
        content = choices[0].message.content if choices else None
        if content and content.strip():
            return content
        reason = getattr(choices[0], "finish_reason", None) if choices else "no choices"
        raise ProviderError(f"OpenAI returned no text (finish_reason={reason}).")
