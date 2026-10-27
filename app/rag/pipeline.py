"""The RAG pipeline: embed the question, retrieve supporting chunks, build a
grounded prompt, generate an answer, and return it *with* citations.

The prompt forces the model to answer only from the provided context and to cite
sources as [1], [2], ... which map back to the returned Citation list.
"""
from __future__ import annotations

from typing import Any

from ..models import Citation, QueryResponse
from ..providers.base import EmbeddingProvider, LLMProvider, ProviderError
from .store import VectorStore

SYSTEM_PROMPT = (
    "You are IntelliDoc, a precise document assistant. Answer the user's question "
    "using ONLY the numbered context passages provided. Cite the passages you use "
    "with bracketed numbers like [1] or [2]. If the answer is not contained in the "
    "context, say you don't have enough information in the documents. Never invent "
    "facts or citations."
)


def _build_context(results: list[tuple[dict[str, Any], float]]) -> str:
    blocks = []
    for i, (meta, _score) in enumerate(results, start=1):
        loc = f"{meta['source']}"
        if meta.get("page"):
            loc += f", p.{meta['page']}"
        blocks.append(f"[{i}] (source: {loc})\n{meta['text']}")
    return "\n\n".join(blocks)


def _snippet(text: str, n: int = 240) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + "…"


class RAGPipeline:
    def __init__(
        self,
        store: VectorStore,
        embedder: EmbeddingProvider,
        llm: LLMProvider,
        top_k: int = 4,
    ):
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.top_k = top_k

    def answer(
        self, question: str, top_k: int | None = None, doc_ids: list[str] | None = None
    ) -> QueryResponse:
        k = top_k or self.top_k
        if self.store.total_chunks == 0:
            return self._no_answer(
                "I don't have any documents indexed yet, so I can't answer that."
            )
        if doc_ids is not None and not self.store.has_any_doc(doc_ids):
            return self._no_answer(
                "None of the selected documents are in the index, so I can't answer that."
            )

        q_vec = self.embedder.embed_query(question)
        results = self.store.search(q_vec, k, doc_ids=doc_ids)
        if not results:
            return self._no_answer("I couldn't find any relevant passages for that question.")

        context = _build_context(results)
        user_prompt = (
            f"Context passages:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer with inline citations like [1]."
        )
        answer = self.llm.complete(SYSTEM_PROMPT, user_prompt)
        if not answer or not answer.strip():
            raise ProviderError(f"LLM provider {self.llm.name!r} returned an empty answer.")

        citations = [
            Citation(
                doc_id=meta["doc_id"],
                source=meta["source"],
                page=meta.get("page"),
                chunk_index=meta["chunk_index"],
                score=round(score, 4),
                snippet=_snippet(meta["text"]),
            )
            for meta, score in results
        ]
        return QueryResponse(answer=answer, citations=citations, model=self.llm.model)

    def _no_answer(self, message: str) -> QueryResponse:
        return QueryResponse(answer=message, citations=[], model=self.llm.model)
