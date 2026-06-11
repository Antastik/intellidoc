"""Deterministic, dependency-free providers used by the test suite and for
local smoke-testing without any API keys or model downloads.

The embedding is a hashing bag-of-words projected into a fixed-dim vector and
L2-normalised. It is *not* semantically smart, but it is stable and fast, which
is exactly what unit tests need.
"""
from __future__ import annotations

import hashlib
import math
import re

from .base import EmbeddingProvider, LLMProvider

_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbeddings(EmbeddingProvider):
    name = "testing"

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.model = f"hashing-{dim}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for tok in _TOKEN.findall(text.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                vec[h % self.dim] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class EchoLLM(LLMProvider):
    """Returns a templated answer that quotes the supplied context, so the RAG
    pipeline can be exercised end-to-end without a real model."""

    name = "testing"
    model = "echo"

    def complete(self, system: str, user: str) -> str:
        first_line = next((ln for ln in user.splitlines() if ln.strip()), "")
        return f"[echo] {first_line[:200]}"
