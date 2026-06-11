"""Provider interfaces. Every concrete provider implements one or both of these,
so the RAG core never imports a vendor SDK directly."""
from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Turns text into vectors."""

    name: str = "base"
    dim: int = 0

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns one vector per input text."""
        raise NotImplementedError

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class LLMProvider(ABC):
    """Generates an answer given a prompt."""

    name: str = "base"
    model: str = "base"

    @abstractmethod
    def complete(self, system: str, user: str) -> str:
        raise NotImplementedError
