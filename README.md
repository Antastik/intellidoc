# IntelliDoc — Source-Citing RAG over your PDFs

<!-- LIVE_URL --> <!-- Live demo: https://intellidoc.onrender.com  (added after first deploy) -->

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
- **Tested.** `pytest` covers chunking, retrieval, persistence, the HTTP API
  end-to-end (upload → list → cited query, bad/scanned PDFs, document filters,
  provider failures, concurrent uploads) and the vendor SDK adapters via faked
  clients — all runnable with **no API keys** via a deterministic `testing`
  provider.

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
python3.11 -m venv .venv && source .venv/bin/activate   # Python 3.11 or 3.12
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

# No keys at all (smoke tests / CI): hashing embeddings + an echo stub LLM
LLM_PROVIDER=testing
EMBEDDING_PROVIDER=testing
```

> Tip: you can mix them, e.g. local embeddings (free) + Gemini for generation.

> **`testing` is not a model.** It returns `[echo] <first context line>` instead
> of a real answer and uses a bag-of-words hash as the embedding. It exists so
> `pytest`/the eval harness run with zero secrets — don't judge answer quality
> with it.

Gemini runs on the `google-genai` SDK with the documented rolling alias
`gemini-flash-latest` and `gemini-embedding-001` (truncated to
`GEMINI_EMBEDDING_DIM=768` via `output_dimensionality`; documents and queries use
the `RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY` task types). OpenAI vector sizes
come from a built-in map (`text-embedding-3-small/large`, `ada-002`); set
`OPENAI_EMBEDDING_DIM` for other models or to shrink `text-embedding-3-*`.

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
# Unit + API tests — no keys needed
pytest

# Frontend tests (XSS escaping, error rendering) — Node >= 18, no npm install
node --test tests/test_frontend.cjs

# Retrieval eval with the dependency-free providers
EMBEDDING_PROVIDER=testing LLM_PROVIDER=testing python -m eval.run_eval
```

## Troubleshooting

- **Re-index after changing embedding models.** The FAISS index in `DATA_DIR`
  only makes sense for the model that produced it. Switching models (including
  the migration from `text-embedding-004` to `gemini-embedding-001`) requires a
  fresh `DATA_DIR` **even when the dimension is unchanged** — same size, different
  vector space, garbage similarity scores. If the dimension *does* differ, the
  app refuses to start with a clear error rather than silently loading or
  deleting the old index; point `DATA_DIR` somewhere new or remove
  `index.faiss` + `meta.pkl` yourself.
- **`502 provider_error`.** The upstream model/embedding call failed (bad key,
  retired model name, quota, safety block, empty response). The response body is
  deliberately generic — `{"detail": ..., "error": "provider_error", "stage":
  "embedding" | "query"}` — and the real cause is in the server log.
- **Startup fails with a settings error.** `CHUNK_SIZE` and `TOP_K` must be
  positive and `CHUNK_OVERLAP` must be smaller than `CHUNK_SIZE`.
- **`422` on upload.** The file is not a parseable PDF, or it has no extractable
  text (scanned images need OCR first). Multi-file uploads are all-or-nothing:
  one bad file rejects the batch and nothing is indexed.
- **Dependency pins.** `google-genai` 2.x requires `httpx>=0.28` and
  `pydantic>=2.12`; `openai` 1.51 does not work with `httpx` 0.28, hence the
  bumped pins in both requirements files. Install from a clean venv.

## Docker

```bash
docker build -t intellidoc .
docker run -p 8000:8000 --env-file .env -v $(pwd)/data:/data intellidoc
```

The image uses `requirements-deploy.txt` (core + OpenAI/Gemini, no PyTorch) to
stay small enough for free hosting tiers. The container binds to `$PORT` if the
host sets one, else `8000`.

## Deploy (Render free tier)

This repo ships a [`render.yaml`](./render.yaml) blueprint, so deploying is
near one-click:

1. Push this repo to GitHub (already done if you're reading it there).
2. Get a free **Gemini API key** from
   [aistudio.google.com/apikey](https://aistudio.google.com/apikey).
3. On [render.com](https://render.com): **New → Blueprint**, pick this repo.
   Render reads `render.yaml` and provisions a free Docker web service.
4. When prompted, paste your key into the `GEMINI_API_KEY` env var (it is
   marked `sync: false`, so it never lives in git).
5. Deploy. Your live URL will be `https://<service-name>.onrender.com`.

> Notes: the free tier sleeps after ~15 min idle (first request after sleep is
> slow to wake), and the FAISS store is ephemeral — uploaded PDFs reset on
> redeploy/sleep, which is expected for a public demo. For a persistent corpus,
> attach a Render disk (paid) or swap the store for pgvector/Qdrant.

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
