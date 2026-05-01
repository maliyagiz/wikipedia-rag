"""End-to-end ingestion: download Wikipedia → chunk → embed → store locally.

Run from project root:
    python -m scripts.run_ingest             # incremental (skip what's done)
    python -m scripts.run_ingest --reset     # wipe vector store first

Resumability:
- Articles already on disk (data/<type>/<slug>.json) are NOT re-downloaded.
- Embeddings are cached on disk (data/embed_cache.json), so re-running is
  ~free even after wiping the vector store: every chunk you've embedded once
  is served from the cache instead of hitting Ollama.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Make project root importable when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DATA_DIR, EMBED_MODEL, PEOPLE, PLACES
from src.chunker import chunk_text
from src.embed_cache import EmbedCache
from src.embedder import OllamaEmbedder
from src.ingest import ingest_entities, load_all
from src.vector_store import VectorStore


def _already_indexed_titles(store: VectorStore) -> set[tuple[str, str]]:
    """Return the set of (title, type) pairs already present in the store."""
    titles: set[tuple[str, str]] = set()
    for c in store._chunks:  # internal but fine — same module ecosystem
        m = c.get("metadata") or {}
        t, k = m.get("title"), m.get("type")
        if t and k:
            titles.add((t, k))
    return titles


def main(reset: bool = False) -> None:
    print(">>> 1/3  Downloading Wikipedia articles (cached files are skipped) ...")
    ingest_entities(PEOPLE, "person")
    ingest_entities(PLACES, "place")

    docs = load_all()
    print(f"\n>>> 2/3  Loaded {len(docs)} articles. Chunking & embedding ...")

    cache = EmbedCache(DATA_DIR / "embed_cache.json", model=EMBED_MODEL)
    print(f"    embedding cache: {len(cache):,} entries on disk")

    embedder = OllamaEmbedder(cache=cache)
    store = VectorStore()
    if reset:
        store.reset()

    already = _already_indexed_titles(store)
    if already:
        print(f"    ({len(already)} articles already in vector store; skipping those)")

    total_chunks = 0
    for di, doc in enumerate(docs, 1):
        key = (doc["title"], doc["type"])
        if key in already:
            print(f"  = [{di}/{len(docs)}] {doc['type']:6s}  {doc['title']:40s}  [in store]")
            continue

        chunks = chunk_text(doc["text"])
        if not chunks:
            continue
        metas = [
            {
                "title": doc["title"],
                "type": doc["type"],
                "url": doc.get("url", ""),
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]
        t0 = time.time()
        # Embed in modest batches so the user sees progress lines.
        BATCH = 20
        embeddings: list[list[float]] = []
        for start in range(0, len(chunks), BATCH):
            piece = chunks[start : start + BATCH]
            embeddings.extend(embedder.embed_many(piece, role="document"))
            done = min(start + BATCH, len(chunks))
            elapsed = time.time() - t0
            print(f"    [{di}/{len(docs)}] {doc['title']:35s}  "
                  f"chunk {done:3d}/{len(chunks):3d}  ({elapsed:5.1f}s)")
        cache.flush()
        store.add(chunks, embeddings, metas)
        total_chunks += len(chunks)
        print(f"  + {doc['type']:6s}  {doc['title']:40s}  {len(chunks):3d} chunks "
              f"in {time.time()-t0:.1f}s")

    cache.flush()
    print(f"\n>>> 3/3  Done. {total_chunks} new chunks added "
          f"(total in store = {store.count()}, cache = {len(cache):,}).")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--reset", action="store_true",
                   help="Wipe the vector store before re-embedding (cache is kept).")
    args = p.parse_args()
    main(reset=args.reset)
