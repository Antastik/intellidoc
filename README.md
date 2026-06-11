# IntelliDoc — Source-Citing RAG over your PDFs

Ask natural-language questions about your documents and get answers **with
citations back to the exact page they came from**. Built as a clean FastAPI
service with a pluggable model layer — run it on OpenAI, Google Gemini, or fully
offline with local models, by changing one line in `.env`.

> Rebuilt from an earlier Streamlit prototype into a properly layered service:
> typed API, provider abstraction, persistent vector store, source citations,
> an evaluation harness, and tests.

---

## Why this is more than a demo

- **Source citations, not just answers.** Every response lists the chunks it
  used — filename, page, similarity score, and a snippet — and the model is
  prompted to cite inline as `[1] [2]`. This is the single biggest thing that
  separates a trustworthy RAG system from a hallucination machine.
- **Pluggable providers.** `OpenAI | Gemini | local` for both the LLM and the
  embeddings, behind a small interface (`app/providers/base.py`). Vendor SDKs are
  imported lazily, so you only install what you use.
- **Multi-document + page-aware retrieval.** Upload many PDFs; query across all
  of them or scope to specific documents.
- **Persistent vector store.** FAISS index + metadata are saved to disk, so your
  corpus survives restarts.
- **Evaluation harness.** `eval/run_eval.py` measures retrieval Hit@k and answer
  keyword-recall, so "did I make it better?" is a number, not a vibe.
- **Tested.** `pytest` covers chunking, retrieval, persistence, and the
  empty-store edge case — all runnable with **no API keys** via a deterministic
  `testing` provider.

## Architecture

```
            ┌──────────────┐    upload (PDF)     ┌───────────────┐
  Browser   │  frontend/   │ ───────────────────▶│  FastAPI      │
  (HTML/JS) │  index.html  │    query (JSON)     │  app/main.py  │
            └──────────────┘ ◀─────────────────── └──────┬────────┘
                  answer + citations                     │
                                                         ▼
              ┌──────────────────────────────────────────────────────┐
              │                   RAG pipeline                        │
              │  ingest → chunk → embed → FAISS search → grounded LLM │
              │            app/rag/{ingest,store,pipeline}.py         │
              └───────────────┬───────────────────────┬───────────────┘
                              ▼                       ▼
                   ┌───────────────────┐   ┌────────────────────┐
                   │ EmbeddingProvider │   │   LLMProvider      │
                   │ openai/gemini/    │   │ openai/gemini/     │
                   │ local/testing     │   │ local/testing      │
                   └───────────────────┘   └────────────────────┘
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then set your provider + key

uvicorn app.main:app --reload
# open http://localhost:8000
```

### Choosing a provider

Edit `.env`:

```ini
# OpenAI (needs OPENAI_API_KEY)
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=openai

# Gemini (needs GEMINI_API_KEY)
LLM_PROVIDER=gemini
EMBEDDING_PROVIDER=gemini

# Fully local: Ollama for the LLM + sentence-transformers for embeddings
LLM_PROVIDER=local
EMBEDDING_PROVIDER=local
```

> Tip: you can mix them, e.g. local embeddings (free) + Gemini for generation.

## API

| Method | Path             | Description                                   |
|--------|------------------|-----------------------------------------------|
| GET    | `/api/health`    | Provider info + corpus stats                  |
| GET    | `/api/documents` | List indexed documents                        |
| POST   | `/api/upload`    | Upload one or more PDFs (multipart)           |
| POST   | `/api/query`     | `{ "question": "...", "doc_ids": [...] }`     |

Example:

```bash
curl -F "files=@handbook.pdf" http://localhost:8000/api/upload
curl -X POST http://localhost:8000/api/query \
     -H "Content-Type: application/json" \
     -d '{"question":"How many days of annual leave do I get?"}'
```

## Tests & evaluation

```bash
# Unit tests — no keys needed
pytest

# Retrieval eval with the dependency-free providers
EMBEDDING_PROVIDER=testing LLM_PROVIDER=testing python -m eval.run_eval
```

## Docker

```bash
docker build -t intellidoc .
docker run -p 8000:8000 --env-file .env -v $(pwd)/data:/data intellidoc
```

## Project layout

```
app/
  config.py           # settings (pydantic-settings)
  models.py           # API schemas
  providers/          # openai | gemini | local | testing, behind one interface
  rag/
    ingest.py         # PDF → page-aware overlapping chunks
    store.py          # FAISS + metadata, persistent
    pipeline.py       # retrieve → grounded prompt → cited answer
  main.py             # FastAPI app
frontend/index.html   # minimal upload + ask UI
eval/                 # eval set + harness
tests/                # pytest
```

## Roadmap

- Streaming responses (SSE) in the UI
- Re-ranking retrieved chunks with a cross-encoder
- Hybrid search (BM25 + dense)
- `pgvector`/Qdrant backend swap (the store interface already isolates this)
- Per-user document namespaces + auth
