"""Query routing + retrieval."""
from __future__ import annotations

from config import TOP_K
from src.classifier import classify
from src.embedder import OllamaEmbedder
from src.vector_store import Hit, VectorStore


class Retriever:
    def __init__(self, store: VectorStore, embedder: OllamaEmbedder):
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str, k: int = TOP_K) -> tuple[list[Hit], str]:
        kind = classify(query)
        emb = self.embedder.embed_one(query, role="query")

        if kind == "both":
            people_hits = self.store.query(emb, k=k, type_filter="person")
            place_hits = self.store.query(emb, k=k, type_filter="place")
            merged = sorted(people_hits + place_hits, key=lambda h: h.distance)
            return merged[: 2 * k], kind

        return self.store.query(emb, k=k, type_filter=kind), kind
