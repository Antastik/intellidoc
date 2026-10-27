"""FastAPI application wiring the providers, vector store, and RAG pipeline
together behind a small HTTP API, plus a static frontend.

`create_app(settings)` is the composition root; the module-level `app` is what
uvicorn serves. Provider calls are confined to `_provider_call`, which turns any
vendor/SDK failure into a structured 502 without echoing exception contents
(which can include request bodies or credentials) to the client."""
from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
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
from .rag import Chunk, RAGPipeline, VectorStore, ingest_pdf_bytes

logger = logging.getLogger("intellidoc")

_T = TypeVar("_T")

_PROVIDER_MESSAGES = {
    "embedding": "The embedding provider request failed. Check the provider "
                 "configuration and server logs.",
    "query": "The model provider request failed while answering. Check the "
             "provider configuration and server logs.",
}


class ProviderFailure(Exception):
    def __init__(self, stage: str):
        super().__init__(stage)
        self.stage = stage


def _provider_call(stage: str, fn: Callable[[], _T]) -> _T:
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 - boundary: any vendor/SDK error
        logger.error("Provider failure during %s: %s", stage, type(e).__name__, exc_info=True)
        raise ProviderFailure(stage) from None


def _parse_upload(f: UploadFile, settings: Settings) -> list[Chunk]:
    name = f.filename or ""
    if not name.lower().endswith(".pdf"):
        raise HTTPException(400, f"Only PDF files are supported: {name}")
    data = f.file.read()
    try:
        chunks = ingest_pdf_bytes(name, data, settings.chunk_size, settings.chunk_overlap)
    except Exception as e:  # noqa: BLE001 - pypdf raises many exception types
        raise HTTPException(422, f"Failed to parse {name}: {type(e).__name__}") from e
    if not chunks:
        raise HTTPException(422, f"No extractable text in {name}.")
    return chunks


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    embedder = build_embedding_provider(settings)
    llm = build_llm_provider(settings)
    store = VectorStore(dim=embedder.dim, data_dir=settings.data_path)
    pipeline = RAGPipeline(store, embedder, llm, top_k=settings.top_k)

    app = FastAPI(title="IntelliDoc", version="1.0.0",
                  description="Pluggable, source-citing RAG over your PDFs.")
    app.state.settings = settings
    app.state.embedder = embedder
    app.state.llm = llm
    app.state.store = store
    app.state.pipeline = pipeline

    @app.exception_handler(ProviderFailure)
    def _provider_failure(_: Request, exc: ProviderFailure) -> JSONResponse:
        return JSONResponse(
            status_code=502,
            content={
                "detail": _PROVIDER_MESSAGES[exc.stage],
                "error": "provider_error",
                "stage": exc.stage,
            },
        )

    @app.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            llm_provider=app.state.llm.name,
            embedding_provider=app.state.embedder.name,
            documents=len(store.documents()),
            chunks=store.total_chunks,
        )

    @app.get("/api/documents", response_model=list[DocumentInfo])
    def list_documents() -> list[DocumentInfo]:
        return [DocumentInfo(**d) for d in store.documents()]

    @app.post("/api/upload", response_model=UploadResponse)
    def upload(files: list[UploadFile] = File(...)) -> UploadResponse:
        parsed = [_parse_upload(f, settings) for f in files]
        all_chunks = [c for chunks in parsed for c in chunks]
        embeddings = _provider_call(
            "embedding", lambda: app.state.embedder.embed([c.text for c in all_chunks])
        )
        store.add(all_chunks, embeddings)
        return UploadResponse(
            documents=[
                DocumentInfo(
                    doc_id=chunks[0].doc_id,
                    source=chunks[0].source,
                    pages=max((c.page or 0) for c in chunks),
                    chunks=len(chunks),
                )
                for chunks in parsed
            ]
        )

    @app.post("/api/query", response_model=QueryResponse)
    def query(req: QueryRequest) -> QueryResponse:
        return _provider_call(
            "query",
            lambda: app.state.pipeline.answer(req.question, top_k=req.top_k, doc_ids=req.doc_ids),
        )

    frontend = Path(__file__).resolve().parent.parent / "frontend"
    if frontend.exists():
        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(frontend / "index.html")

        app.mount("/static", StaticFiles(directory=frontend), name="static")

    return app


_app: FastAPI | None = None


def __getattr__(name: str) -> FastAPI:
    """Build the production `app` on first access (uvicorn `app.main:app`), so
    importing this module for tests has no side effects."""
    global _app
    if name != "app":
        raise AttributeError(name)
    if _app is None:
        _app = create_app()
    return _app
