"""FastAPI application wiring the providers, vector store, and RAG pipeline
together behind a small HTTP API, plus a static frontend."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings, get_settings
from .models import (
    DocumentInfo,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    UploadResponse,
)
from .providers import build_embedding_provider, build_llm_provider
from .rag import RAGPipeline, VectorStore, ingest_pdf_bytes

app = FastAPI(title="IntelliDoc", version="1.0.0",
              description="Pluggable, source-citing RAG over your PDFs.")

# ---- Composition root: build everything once at startup ----
_settings: Settings = get_settings()
_embedder = build_embedding_provider(_settings)
_llm = build_llm_provider(_settings)
_store = VectorStore(dim=_embedder.dim, data_dir=_settings.data_path)
_pipeline = RAGPipeline(_store, _embedder, _llm, top_k=_settings.top_k)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        llm_provider=_llm.name,
        embedding_provider=_embedder.name,
        documents=len(_store.documents()),
        chunks=_store.total_chunks,
    )


@app.get("/api/documents", response_model=list[DocumentInfo])
def list_documents() -> list[DocumentInfo]:
    return [DocumentInfo(**d) for d in _store.documents()]


@app.post("/api/upload", response_model=UploadResponse)
async def upload(files: list[UploadFile] = File(...)) -> UploadResponse:
    out: list[DocumentInfo] = []
    for f in files:
        if not (f.filename or "").lower().endswith(".pdf"):
            raise HTTPException(400, f"Only PDF files are supported: {f.filename}")
        data = await f.read()
        try:
            chunks = ingest_pdf_bytes(
                f.filename, data, _settings.chunk_size, _settings.chunk_overlap
            )
        except Exception as e:  # noqa: BLE001
            raise HTTPException(422, f"Failed to parse {f.filename}: {e}") from e
        if not chunks:
            raise HTTPException(422, f"No extractable text in {f.filename}.")
        embeddings = _embedder.embed([c.text for c in chunks])
        _store.add(chunks, embeddings)
        pages = max((c.page or 0) for c in chunks)
        out.append(
            DocumentInfo(
                doc_id=chunks[0].doc_id,
                source=f.filename,
                pages=pages,
                chunks=len(chunks),
            )
        )
    return UploadResponse(documents=out)


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    return _pipeline.answer(req.question, top_k=req.top_k, doc_ids=req.doc_ids)


# ---- Static frontend ----
_FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
if _FRONTEND.exists():
    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(_FRONTEND / "index.html")

    app.mount("/static", StaticFiles(directory=_FRONTEND), name="static")
