"""Builds the configured providers. This is the only place that knows about
every vendor, keeping the rest of the app vendor-agnostic."""
from __future__ import annotations

from ..config import Settings
from .base import EmbeddingProvider, LLMProvider


def build_embedding_provider(s: Settings) -> EmbeddingProvider:
    provider = s.embedding_provider.lower()
    if provider == "openai":
        from .openai_provider import OpenAIEmbeddings
        return OpenAIEmbeddings(s.openai_api_key, s.openai_embedding_model)
    if provider == "gemini":
        from .gemini_provider import GeminiEmbeddings
        return GeminiEmbeddings(s.gemini_api_key, s.gemini_embedding_model)
    if provider == "local":
        from .local_provider import LocalEmbeddings
        return LocalEmbeddings(s.local_embedding_model)
    if provider == "testing":
        from .testing import HashingEmbeddings
        return HashingEmbeddings()
    raise ValueError(f"Unknown embedding provider: {s.embedding_provider!r}")


def build_llm_provider(s: Settings) -> LLMProvider:
    provider = s.llm_provider.lower()
    if provider == "openai":
        from .openai_provider import OpenAILLM
        return OpenAILLM(s.openai_api_key, s.openai_llm_model)
    if provider == "gemini":
        from .gemini_provider import GeminiLLM
        return GeminiLLM(s.gemini_api_key, s.gemini_llm_model)
    if provider == "local":
        from .local_provider import LocalLLM
        return LocalLLM(s.ollama_base_url, s.local_llm_model)
    if provider == "testing":
        from .testing import EchoLLM
        return EchoLLM()
    raise ValueError(f"Unknown LLM provider: {s.llm_provider!r}")
