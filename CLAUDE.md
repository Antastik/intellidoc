# CLAUDE.md — IntelliDoc

Context for Claude Code working in this repo. Read this first.

## What this is
IntelliDoc is a **source-citing RAG service over PDFs**. Upload PDFs, ask
questions, get answers **with citations** (filename + page + similarity score +
snippet). It was rebuilt from an old Streamlit prototype into a properly layered
FastAPI service with a **pluggable provider layer** (OpenAI / Gemini / local).

Owner: Animesh Goswami. Goal: a strong portfolio piece for **AI-leaning software
engineering roles**. Prioritise clean, tested, deployable code over cleverness.

## Architecture (don't break these boundaries)
- `app/config.py` — settings via `pydantic-settings` (reads `.env`).
- `app/models.py` — Pydantic API schemas. This is the public contract.
- `app/providers/` — `base.py` defines `EmbeddingProvider` / `LLMProvider`.
  Concrete impls (`openai_provider.py`, `gemini_provider.py`,
  `local_provider.py`, `testing.py`) **lazy-import** their SDKs. `factory.py` is
  the ONLY place that knows every vendor. Keep the rest vendor-agnostic.
- `app/rag/ingest.py` — PDF → page-aware overlapping chunks (`Chunk` dataclass).
- `app/rag/store.py` — FAISS (inner-product on normalised vecs = cosine) +
  pickled metadata, persisted to `DATA_DIR`. Swappable for pgvector/Qdrant later.
- `app/rag/pipeline.py` — embed → retrieve → grounded prompt → cited answer.
- `app/main.py` — FastAPI composition root (builds providers once at startup).
- `frontend/index.html` — single-file upload + ask UI.
- `eval/` — retrieval eval harness. `tests/` — pytest.

## The `testing` provider matters
`app/providers/testing.py` has `HashingEmbeddings` + `EchoLLM` — deterministic,
no API keys, no downloads. Tests and the eval harness use it. **Keep it working**
so CI and `pytest` run with zero secrets.

## Run / test
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # set LLM_PROVIDER + EMBEDDING_PROVIDER + key

uvicorn app.main:app --reload # http://localhost:8000

pytest                                                  # unit tests (no keys)
EMBEDDING_PROVIDER=testing LLM_PROVIDER=testing python -m eval.run_eval
```
Last verified: **6/6 tests pass; eval retrieval Hit@4 = 100%** (echo LLM, so
answer-recall is 0 by design — use a real provider for meaningful answer scores).

## Conventions
- Python 3.11, type hints, `from __future__ import annotations`.
- New vendors: add a provider class + wire it in `factory.py` only.
- Anything touching retrieval quality should come with an eval number.
- `.env`, `data/`, `__pycache__` are gitignored — never commit them.

## Next tasks (priority order)
1. **Deploy** (Render/Railway free tier) and put the live URL in `README.md`.
2. **Record a GIF** (upload → ask → cited answer) for the top of the README.
3. **Streaming responses** (SSE) so answers render token-by-token in the UI.
4. **Cross-encoder re-ranking** of retrieved chunks (e.g. `bge-reranker`).
5. **Hybrid search** (BM25 + dense) — measure the lift with the eval harness.
6. **Delete-document** endpoint + per-document namespaces.
7. Optional: swap FAISS for pgvector/Qdrant behind the existing `store` interface.

See `HANDOFF.md` for the wider context (resume + other projects).
