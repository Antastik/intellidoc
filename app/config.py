"""Central configuration, loaded from environment / .env via pydantic-settings."""
from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Provider selection
    llm_provider: str = "openai"        # openai | gemini | local | testing
    embedding_provider: str = "openai"  # openai | gemini | local | testing

    # OpenAI
    openai_api_key: str = ""
    openai_llm_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    # Only needed for models not in the built-in dimension map, or to shrink
    # text-embedding-3-* vectors. Changing it requires re-indexing.
    openai_embedding_dim: PositiveInt | None = None

    # Gemini (google-genai SDK). `gemini-flash-latest` is the documented
    # rolling alias; gemini-embedding-001 supports output_dimensionality.
    gemini_api_key: str = ""
    gemini_llm_model: str = "gemini-flash-latest"
    gemini_embedding_model: str = "gemini-embedding-001"
    gemini_embedding_dim: PositiveInt = 768

    # Local
    ollama_base_url: str = "http://localhost:11434"
    local_llm_model: str = "llama3.1"
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # RAG
    chunk_size: PositiveInt = 900
    chunk_overlap: NonNegativeInt = 150
    top_k: PositiveInt = 4
    data_dir: str = "./data"

    @model_validator(mode="after")
    def _overlap_smaller_than_chunk(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"CHUNK_OVERLAP ({self.chunk_overlap}) must be smaller than "
                f"CHUNK_SIZE ({self.chunk_size})."
            )
        return self

    @property
    def data_path(self) -> Path:
        p = Path(self.data_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


def get_settings() -> Settings:
    return Settings()
