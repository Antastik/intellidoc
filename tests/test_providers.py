"""Provider + settings regression tests. SDK clients are faked; the real SDK
clients are only *constructed* with dummy keys (no network, no real keys)."""
from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.providers.base import ProviderError

from .conftest import IsolatedSettings


def _settings(**kw: Any) -> Settings:
    return IsolatedSettings(**kw)


# ---- Settings ----

def test_gemini_defaults_are_not_retired_models():
    s = _settings()
    assert s.gemini_llm_model == "gemini-flash-latest"
    assert s.gemini_embedding_model == "gemini-embedding-001"
    assert s.gemini_embedding_dim == 768


@pytest.mark.parametrize(
    "kw",
    [
        {"chunk_size": 0},
        {"chunk_size": -5},
        {"chunk_overlap": -1},
        {"chunk_size": 100, "chunk_overlap": 100},
        {"chunk_size": 100, "chunk_overlap": 150},
        {"top_k": 0},
    ],
)
def test_settings_rejects_invalid_rag_values(kw):
    with pytest.raises(ValidationError):
        _settings(**kw)


def test_settings_accepts_valid_rag_values():
    s = _settings(chunk_size=100, chunk_overlap=99, top_k=1)
    assert (s.chunk_size, s.chunk_overlap, s.top_k) == (100, 99, 1)


# ---- Gemini (google.genai) ----

class _FakeGenaiModels:
    def __init__(self, dim: int, text: str | None = "ok"):
        self.dim = dim
        self.text = text
        self.embed_calls: list[dict[str, Any]] = []
        self.generate_calls: list[dict[str, Any]] = []

    def embed_content(self, *, model, contents, config):
        self.embed_calls.append({"model": model, "contents": list(contents), "config": config})
        dim = config["output_dimensionality"]
        return SimpleNamespace(
            embeddings=[SimpleNamespace(values=[0.1] * dim) for _ in contents]
        )

    def generate_content(self, *, model, contents, config):
        self.generate_calls.append({"model": model, "contents": contents, "config": config})
        return SimpleNamespace(
            text=self.text,
            candidates=[SimpleNamespace(finish_reason="SAFETY")],
            prompt_feedback=None,
        )


def _fake_genai(dim: int = 768, text: str | None = "ok"):
    return SimpleNamespace(models=_FakeGenaiModels(dim, text))


def test_gemini_provider_module_does_not_use_retired_sdk():
    import app.providers.gemini_provider as mod

    src = inspect.getsource(mod)
    assert "google.generativeai" not in src
    assert "from google import genai" in src


def test_gemini_embeddings_batched_with_task_types_and_fixed_dim():
    from app.providers.gemini_provider import GeminiEmbeddings

    fake = _fake_genai()
    emb = GeminiEmbeddings("dummy-key", "gemini-embedding-001", dim=768, client=fake)
    assert emb.dim == 768

    texts = [f"chunk {i}" for i in range(250)]
    vecs = emb.embed(texts)
    assert len(vecs) == 250 and all(len(v) == 768 for v in vecs)
    calls = fake.models.embed_calls
    assert [len(c["contents"]) for c in calls] == [100, 100, 50]
    assert all(c["model"] == "gemini-embedding-001" for c in calls)
    assert all(c["config"]["task_type"] == "RETRIEVAL_DOCUMENT" for c in calls)
    assert all(c["config"]["output_dimensionality"] == 768 for c in calls)

    emb.embed_query("question?")
    assert calls[-1]["config"]["task_type"] == "RETRIEVAL_QUERY"
    assert emb.dim == 768


def test_gemini_embeddings_rejects_short_or_missing_vectors():
    from app.providers.gemini_provider import GeminiEmbeddings

    class Short(_FakeGenaiModels):
        def embed_content(self, *, model, contents, config):
            return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1] * 10)])

    emb = GeminiEmbeddings("k", "gemini-embedding-001", dim=768, client=SimpleNamespace(models=Short(768)))
    with pytest.raises(ProviderError):
        emb.embed(["a", "b"])

    class Empty(_FakeGenaiModels):
        def embed_content(self, *, model, contents, config):
            return SimpleNamespace(embeddings=None)

    emb = GeminiEmbeddings("k", "gemini-embedding-001", dim=768, client=SimpleNamespace(models=Empty(768)))
    with pytest.raises(ProviderError):
        emb.embed(["a"])


def test_gemini_llm_uses_system_instruction_and_rejects_empty():
    from app.providers.gemini_provider import GeminiLLM

    fake = _fake_genai(text="answer [1]")
    llm = GeminiLLM("dummy-key", "gemini-flash-latest", client=fake)
    assert llm.complete("SYS", "USER") == "answer [1]"
    call = fake.models.generate_calls[0]
    assert call["model"] == "gemini-flash-latest"
    assert call["contents"] == "USER"
    assert call["config"]["system_instruction"] == "SYS"

    blocked = GeminiLLM("dummy-key", "gemini-flash-latest", client=_fake_genai(text=None))
    with pytest.raises(ProviderError, match="SAFETY"):
        blocked.complete("SYS", "USER")


def test_real_sdk_clients_construct_with_dummy_keys_no_network():
    pytest.importorskip("google.genai")
    pytest.importorskip("openai")
    from app.providers.gemini_provider import GeminiEmbeddings, GeminiLLM
    from app.providers.openai_provider import OpenAIEmbeddings, OpenAILLM

    assert GeminiEmbeddings("dummy", "gemini-embedding-001").dim == 768
    assert GeminiLLM("dummy", "gemini-flash-latest").model == "gemini-flash-latest"
    assert OpenAIEmbeddings("sk-dummy", "text-embedding-3-large").dim == 3072
    assert OpenAILLM("sk-dummy", "gpt-4o-mini").model == "gpt-4o-mini"


def test_missing_keys_raise():
    from app.providers.gemini_provider import GeminiEmbeddings
    from app.providers.openai_provider import OpenAIEmbeddings

    with pytest.raises(ValueError):
        GeminiEmbeddings("", "gemini-embedding-001", client=_fake_genai())
    with pytest.raises(ValueError):
        OpenAIEmbeddings("", "text-embedding-3-small", client=object())


# ---- OpenAI ----

class _FakeOpenAI:
    def __init__(self, dim: int, content: str | None = "ok"):
        self.calls: list[dict[str, Any]] = []
        self._dim = dim
        self._content = content
        outer = self

        class _Emb:
            def create(self, **kw):
                outer.calls.append(kw)
                d = kw.get("dimensions", outer._dim)
                return SimpleNamespace(data=[SimpleNamespace(embedding=[0.2] * d) for _ in kw["input"]])

        class _Completions:
            def create(self, **kw):
                outer.calls.append(kw)
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=outer._content))])

        self.embeddings = _Emb()
        self.chat = SimpleNamespace(completions=_Completions())


def test_openai_embeddings_large_model_dim_known_before_first_call():
    from app.providers.openai_provider import OpenAIEmbeddings

    fake = _FakeOpenAI(3072)
    emb = OpenAIEmbeddings("sk-dummy", "text-embedding-3-large", client=fake)
    assert emb.dim == 3072
    vecs = emb.embed(["a", "b"])
    assert len(vecs) == 2 and len(vecs[0]) == 3072
    assert fake.calls[0]["dimensions"] == 3072
    assert emb.dim == 3072


def test_openai_embeddings_explicit_dim_and_unknown_model():
    from app.providers.openai_provider import OpenAIEmbeddings

    fake = _FakeOpenAI(1536)
    emb = OpenAIEmbeddings("sk-dummy", "text-embedding-3-small", dim=512, client=fake)
    assert emb.dim == 512
    assert len(emb.embed(["x"])[0]) == 512
    assert fake.calls[0]["dimensions"] == 512

    with pytest.raises(ValueError, match="dimension"):
        OpenAIEmbeddings("sk-dummy", "some-future-model", client=fake)
    emb = OpenAIEmbeddings("sk-dummy", "some-future-model", dim=42, client=_FakeOpenAI(42))
    assert emb.dim == 42


def test_openai_embeddings_batches_and_validates_response():
    from app.providers.openai_provider import OpenAIEmbeddings

    fake = _FakeOpenAI(1536)
    emb = OpenAIEmbeddings("sk-dummy", "text-embedding-3-small", client=fake)
    emb.embed([f"t{i}" for i in range(2500)])
    assert [len(c["input"]) for c in fake.calls] == [2048, 452]

    bad = _FakeOpenAI(10)
    emb = OpenAIEmbeddings("sk-dummy", "text-embedding-ada-002", client=bad)
    with pytest.raises(ProviderError):
        emb.embed(["x"])


def test_openai_llm_rejects_empty_content():
    from app.providers.openai_provider import OpenAILLM

    assert OpenAILLM("sk-dummy", "gpt-4o-mini", client=_FakeOpenAI(1, "hi")).complete("s", "u") == "hi"
    with pytest.raises(ProviderError):
        OpenAILLM("sk-dummy", "gpt-4o-mini", client=_FakeOpenAI(1, None)).complete("s", "u")


def test_factory_wires_dims_from_settings():
    from app.providers import build_embedding_provider

    s = _settings(embedding_provider="gemini", gemini_api_key="dummy", gemini_embedding_dim=1536)
    pytest.importorskip("google.genai")
    assert build_embedding_provider(s).dim == 1536
