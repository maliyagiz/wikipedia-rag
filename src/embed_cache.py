"""On-disk JSON cache: sha256(model + chunk_text) → embedding vector.

Survives across runs. If you re-run ingestion (even after wiping the vector
store), already-embedded chunks are served from the cache and never hit Ollama.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional


class EmbedCache:
    def __init__(self, path: Path, model: str):
        self.path = Path(path)
        self.model = model
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, list[float]] = {}
        if self.path.exists():
            try:
                self._cache = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self._cache = {}
        self._dirty = False

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}::{text}".encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[list[float]]:
        return self._cache.get(self._key(text))

    def put(self, text: str, embedding: list[float]) -> None:
        self._cache[self._key(text)] = embedding
        self._dirty = True

    def flush(self) -> None:
        if not self._dirty:
            return
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._cache), encoding="utf-8")
        tmp.replace(self.path)
        self._dirty = False

    def __len__(self) -> int:
        return len(self._cache)
