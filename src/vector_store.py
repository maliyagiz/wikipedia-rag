"""Persistent numpy-backed vector store with cosine similarity.

Why not Chroma:
- Chroma 0.5+ ships a Rust HNSW backend that fails to load persisted index
  segments on Windows ("Error loading hnsw index"), particularly when the
  data directory contains non-ASCII path components.
- The homework spec asks us to "use language native functionality rather than
  fully featured libraries that do the core work of the exercise out of the
  box". A few lines of numpy do exactly the core work (cosine similarity over
  ~few-thousand 1024-d vectors with mxbai-embed-large), with full
  transparency.

Design choice (Option B from the spec): a SINGLE store holding both people and
place chunks, with a `type` metadata field on every record. Routing is done at
query-time via a boolean mask before argsort.

Why one store over two:
- Mixed queries ("Compare Einstein and the Eiffel Tower") need a single index
  to run one query over both kinds of chunks.
- Adding a new entity type later (e.g. "events") is a metadata value, not a
  new store.
- Disk overhead is lower with one .npy + one chunks.json.

Storage layout under VECTOR_DIR:
    embeddings.npy   — float32 array of shape (N, dim)
    chunks.json      — list[{text, metadata}] with the same N order

Persistence is atomic: writes go to *.tmp, then replace.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import VECTOR_DIR


@dataclass
class Hit:
    text: str
    metadata: dict
    distance: float


_EPS = 1e-12


class VectorStore:
    EMB_FILE = "embeddings.npy"
    META_FILE = "chunks.json"

    def __init__(self, persist_dir: Path = VECTOR_DIR):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.emb_path = self.persist_dir / self.EMB_FILE
        self.meta_path = self.persist_dir / self.META_FILE
        self._embeddings: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self._normed: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self._chunks: list[dict] = []
        self._load()

    # ---- persistence ----
    def _load(self) -> None:
        if self.emb_path.exists() and self.meta_path.exists():
            try:
                self._embeddings = np.load(self.emb_path).astype(np.float32, copy=False)
                self._chunks = json.loads(self.meta_path.read_text(encoding="utf-8"))
                if len(self._chunks) != self._embeddings.shape[0]:
                    # Mismatched files — treat as empty.
                    self._embeddings = np.zeros((0, 0), dtype=np.float32)
                    self._chunks = []
            except Exception:
                self._embeddings = np.zeros((0, 0), dtype=np.float32)
                self._chunks = []
        self._renormalize()

    def _renormalize(self) -> None:
        if self._embeddings.size == 0:
            self._normed = self._embeddings
            return
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True) + _EPS
        self._normed = (self._embeddings / norms).astype(np.float32, copy=False)

    def _persist(self) -> None:
        # Atomic writes via *.tmp -> replace.
        # NB: np.save(path, ...) helpfully appends ".npy" if missing, so we
        # pass a file handle to bypass that behaviour and keep the exact name.
        tmp_emb = self.emb_path.with_name(self.emb_path.name + ".tmp")
        tmp_meta = self.meta_path.with_name(self.meta_path.name + ".tmp")
        with open(tmp_emb, "wb") as f:
            np.save(f, self._embeddings, allow_pickle=False)
        tmp_meta.write_text(json.dumps(self._chunks, ensure_ascii=False), encoding="utf-8")
        tmp_emb.replace(self.emb_path)
        tmp_meta.replace(self.meta_path)

    # ---- write path ----
    def add(self, texts: list[str], embeddings: list[list[float]],
            metadatas: list[dict]) -> None:
        if not texts:
            return
        new = np.asarray(embeddings, dtype=np.float32)
        if self._embeddings.size == 0:
            self._embeddings = new
        else:
            if new.shape[1] != self._embeddings.shape[1]:
                raise ValueError(
                    f"Embedding dim mismatch: store has {self._embeddings.shape[1]}, "
                    f"new vectors are {new.shape[1]}-d"
                )
            self._embeddings = np.vstack([self._embeddings, new])
        for t, m in zip(texts, metadatas):
            self._chunks.append({"text": t, "metadata": dict(m or {})})
        self._renormalize()
        self._persist()

    def reset(self) -> None:
        self._embeddings = np.zeros((0, 0), dtype=np.float32)
        self._normed = self._embeddings
        self._chunks = []
        for p in (self.emb_path, self.meta_path):
            if p.exists():
                p.unlink()

    def count(self) -> int:
        return len(self._chunks)

    # ---- read path ----
    def query(self, embedding: list[float], k: int = 4,
              type_filter: str | None = None) -> list[Hit]:
        if not self._chunks or self._normed.size == 0:
            return []
        q = np.asarray(embedding, dtype=np.float32)
        q = q / (np.linalg.norm(q) + _EPS)
        sims = self._normed @ q  # cosine similarity, shape (N,)

        if type_filter:
            mask = np.array(
                [c["metadata"].get("type") == type_filter for c in self._chunks],
                dtype=bool,
            )
            if not mask.any():
                return []
            ranking = np.where(mask, sims, -np.inf)
        else:
            ranking = sims

        k = min(k, int((ranking > -np.inf).sum()))
        if k <= 0:
            return []
        # argpartition for top-k, then sort the top-k by similarity desc.
        top = np.argpartition(-ranking, kth=k - 1)[:k]
        top = top[np.argsort(-ranking[top])]

        hits: list[Hit] = []
        for i in top:
            i = int(i)
            sim = float(sims[i])
            hits.append(Hit(
                text=self._chunks[i]["text"],
                metadata=self._chunks[i]["metadata"],
                distance=1.0 - sim,
            ))
        return hits
