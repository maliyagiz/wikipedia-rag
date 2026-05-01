"""End-to-end RAG pipeline assembly."""
from __future__ import annotations

from dataclasses import dataclass

from src.embedder import OllamaEmbedder
from src.generator import generate, generate_stream
from src.retriever import Retriever
from src.vector_store import Hit, VectorStore


@dataclass
class RagResult:
    answer: str
    hits: list[Hit]
    routing: str   # "person" | "place" | "both"


class RAG:
    def __init__(self):
        self.store = VectorStore()
        self.embedder = OllamaEmbedder()
        self.retriever = Retriever(self.store, self.embedder)

    def ask(self, query: str) -> RagResult:
        hits, routing = self.retriever.retrieve(query)
        answer = generate(query, hits)
        return RagResult(answer=answer, hits=hits, routing=routing)

    def ask_stream(self, query: str):
        hits, routing = self.retriever.retrieve(query)
        return generate_stream(query, hits), hits, routing
