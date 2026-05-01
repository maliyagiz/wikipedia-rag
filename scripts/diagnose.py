"""Quick diagnostic: what's in the store, and where does Einstein rank?

Run:
    python -m scripts.diagnose
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DATA_DIR, EMBED_MODEL
from src.embed_cache import EmbedCache
from src.embedder import OllamaEmbedder
from src.vector_store import VectorStore


def main() -> None:
    store = VectorStore()
    print(f"Total chunks in store: {store.count()}")
    print(f"Embedding shape: {store._embeddings.shape}")

    titles: Counter[str] = Counter()
    for c in store._chunks:
        titles[c["metadata"].get("title", "?")] += 1
    print(f"\nDistinct articles: {len(titles)}")
    for t, n in sorted(titles.items()):
        print(f"  {n:4d}  {t}")

    einstein_chunks = [c for c in store._chunks if "Einstein" in c["metadata"].get("title", "")]
    print(f"\nEinstein chunks in store: {len(einstein_chunks)}")
    if einstein_chunks:
        print("First Einstein chunk preview:")
        print("  " + einstein_chunks[0]["text"][:240].replace("\n", " ") + "...")

    cache = EmbedCache(DATA_DIR / "embed_cache.json", model=EMBED_MODEL)
    embedder = OllamaEmbedder(cache=cache)
    q_text = "Who was Albert Einstein and what is he known for?"
    q_emb = embedder.embed_one(q_text, role="query")
    print(f"\nQuery: {q_text!r}")
    print(f"Query vector len: {len(q_emb)}, first 5 values: {q_emb[:5]}")

    print("\nTop 10 results (no filter):")
    for h in store.query(q_emb, k=10):
        print(f"  dist={h.distance:.3f}  type={h.metadata.get('type')!s:6s}  "
              f"title={h.metadata.get('title')}")

    print("\nTop 10 results (type=person):")
    for h in store.query(q_emb, k=10, type_filter="person"):
        print(f"  dist={h.distance:.3f}  title={h.metadata.get('title')}")

    # ---- self-check: does the stored doc embedding match a fresh prefixed
    # re-embedding of the same text? If sim≈1.0, docs were stored WITH prefix.
    # If sim is much higher to the no-prefix variant, docs are STILL pre-fix.
    import numpy as np
    if einstein_chunks:
        idx = next(i for i, c in enumerate(store._chunks)
                   if "Einstein" in c["metadata"].get("title", ""))
        stored_vec = store._embeddings[idx]
        stored_text = store._chunks[idx]["text"]
        stored_norm = stored_vec / (np.linalg.norm(stored_vec) + 1e-12)

        # Bypass the cache for this comparison so we hit Ollama fresh.
        plain_embedder = OllamaEmbedder(cache=None)
        e_doc = np.asarray(plain_embedder.embed_one(stored_text, role="document"),
                           dtype=np.float32)
        e_raw = np.asarray(plain_embedder._fetch(stored_text), dtype=np.float32)
        e_doc_n = e_doc / (np.linalg.norm(e_doc) + 1e-12)
        e_raw_n = e_raw / (np.linalg.norm(e_raw) + 1e-12)
        sim_doc = float(stored_norm @ e_doc_n)
        sim_raw = float(stored_norm @ e_raw_n)
        print("\nSelf-check on first Einstein chunk:")
        print(f"  cosine(stored, fresh-with-prefix)    = {sim_doc:.4f}")
        print(f"  cosine(stored, fresh-without-prefix) = {sim_raw:.4f}")
        if sim_doc > 0.99:
            print("  -> docs are CORRECTLY stored with prefix.")
        elif sim_raw > 0.99 and sim_doc < 0.95:
            print("  -> docs are stored WITHOUT prefix (re-ingest with --reset).")
        else:
            print("  -> ambiguous; embeddings differ from both variants.")


if __name__ == "__main__":
    main()
