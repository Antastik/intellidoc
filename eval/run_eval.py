"""Retrieval evaluation harness.

Builds an in-memory index from the eval corpus, runs each question, and measures
how often the expected keywords appear in the retrieved context (Hit@k) and in
the final generated answer (Answer-keyword recall).

Run:
    python -m eval.run_eval                 # uses the configured provider
    EMBEDDING_PROVIDER=testing LLM_PROVIDER=testing python -m eval.run_eval

The `testing` providers need no API keys or downloads, so this works anywhere.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# Allow running as `python eval/run_eval.py` too.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.providers import build_embedding_provider, build_llm_provider  # noqa: E402
from app.rag import RAGPipeline, VectorStore  # noqa: E402
from app.rag.ingest import chunk_pages  # noqa: E402

EVAL_FILE = Path(__file__).resolve().parent / "eval_set.json"


def main() -> int:
    spec = json.loads(EVAL_FILE.read_text())
    settings = get_settings()
    embedder = build_embedding_provider(settings)
    llm = build_llm_provider(settings)

    with tempfile.TemporaryDirectory() as tmp:
        store = VectorStore(dim=embedder.dim, data_dir=Path(tmp))
        # Index corpus: one Chunk per corpus entry.
        for i, entry in enumerate(spec["corpus"]):
            chunks = chunk_pages(
                entry["source"], [entry["text"]],
                settings.chunk_size, settings.chunk_overlap, doc_id=f"doc{i}"
            )
            store.add(chunks, embedder.embed([c.text for c in chunks]))

        pipe = RAGPipeline(store, embedder, llm, top_k=settings.top_k)

        hits, ans_hits, total = 0, 0, 0
        print(f"\nProvider: embed={embedder.name}  llm={llm.name}\n" + "-" * 60)
        for q in spec["questions"]:
            total += 1
            resp = pipe.answer(q["question"])
            ctx = " ".join(c.snippet.lower() for c in resp.citations)
            kws = [k.lower() for k in q["expected_keywords"]]
            ctx_hit = all(k in ctx for k in kws)
            ans_hit = any(k in resp.answer.lower() for k in kws)
            hits += ctx_hit
            ans_hits += ans_hit
            mark = "PASS" if ctx_hit else "MISS"
            print(f"[{mark}] {q['question']}")
            print(f"        answer: {resp.answer[:90].strip()}")

        print("-" * 60)
        print(f"Retrieval Hit@{settings.top_k}: {hits}/{total}  ({hits/total:.0%})")
        print(f"Answer keyword recall: {ans_hits}/{total}  ({ans_hits/total:.0%})")
        return 0 if hits == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
