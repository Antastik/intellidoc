"""Central configuration, loaded from environment / .env via pydantic-settings."""
from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Provider selection
    llm_provider: str = "openai"        # openai | gemini | local
    embedding_provider: str = "openai"  # openai | gemini | local

    # OpenAI
    openai_api_key: str = ""
    openai_llm_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # Gemini
    gemini_api_key: str = ""
    gemini_llm_model: str = "gemini-1.5-flash"
    gemini_embedding_model: str = "text-embedding-004"

    # Local
    ollama_base_url: str = "http://localhost:11434"
    local_llm_model: str = "llama3.1"
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # RAG
    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 4
    data_dir: str = "./data"

    @property
    def data_path(self) -> Path:
        p = Path(self.data_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


def get_settings() -> Settings:
    return Settings()
