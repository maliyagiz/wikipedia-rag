"""Local embeddings via Ollama's REST API (nomic-embed-text by default).

We talk to Ollama directly with `requests` — no LangChain, no Ollama Python SDK.
This keeps dependencies minimal and the call-flow visible.

nomic-embed-text v1.5 is a task-prefixed model: it expects the text to start
with `search_document: ` for stored passages and `search_query: ` for user
queries. Without those prefixes the embeddings collapse to a similar region
and retrieval becomes nearly random — which is exactly what we observed on
this homework. We add the prefixes here transparently via the `role` arg.

An optional `EmbedCache` short-circuits requests for chunks we've already
embedded. The cache key is derived from the *formatted* text (prefix + body),
so document-mode and query-mode embeddings never alias.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Optional, Sequence

import requests

from config import EMBED_MODEL, EMBED_WORKERS, OLLAMA_HOST
from src.embed_cache import EmbedCache

Role = Literal["document", "query"]


class OllamaEmbedder:
    def __init__(self, model: str = EMBED_MODEL, host: str = OLLAMA_HOST,
                 cache: Optional[EmbedCache] = None):
        self.model = model
        self.url = f"{host.rstrip('/')}/api/embeddings"
        self.cache = cache

    def _format(self, text: str, role: Role) -> str:
        # Only nomic-embed-text v1.x uses task-prefix tokens. Other models
        # (mxbai-embed-large, bge-*, snowflake-arctic-embed, etc.) take raw
        # text and adding a prefix would actually hurt them.
        if "nomic" not in self.model.lower():
            return text
        if role == "document":
            return f"search_document: {text}"
        if role == "query":
            return f"search_query: {text}"
        return text

    def _fetch(self, formatted: str) -> list[float]:
        r = requests.post(
            self.url,
            json={"model": self.model, "prompt": formatted},
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        if "embedding" not in data:
            raise RuntimeError(f"Unexpected Ollama embedding response: {data}")
        return data["embedding"]

    def embed_one(self, text: str, role: Role = "document") -> list[float]:
        formatted = self._format(text, role)
        if self.cache is not None:
            cached = self.cache.get(formatted)
            if cached is not None:
                return cached
        emb = self._fetch(formatted)
        if self.cache is not None:
            self.cache.put(formatted, emb)
        return emb

    def embed_many(self, texts: Sequence[str], role: Role = "document",
                   workers: int = EMBED_WORKERS) -> list[list[float]]:
        formatted = [self._format(t, role) for t in texts]
        results: list[Optional[list[float]]] = [None] * len(texts)
        misses: list[int] = []
        if self.cache is not None:
            for i, f in enumerate(formatted):
                hit = self.cache.get(f)
                if hit is not None:
                    results[i] = hit
                else:
                    misses.append(i)
        else:
            misses = list(range(len(texts)))

        if misses:
            miss_texts = [formatted[i] for i in misses]
            if workers <= 1 or len(misses) <= 1:
                fetched = [self._fetch(t) for t in miss_texts]
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    fetched = list(pool.map(self._fetch, miss_texts))
            for j, i in enumerate(misses):
                results[i] = fetched[j]
                if self.cache is not None:
                    self.cache.put(formatted[i], fetched[j])

        return results  # type: ignore[return-value]
