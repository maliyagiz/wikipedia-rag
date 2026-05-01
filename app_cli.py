"""Tiny CLI chat interface for the local Wikipedia RAG.

Commands:
    /context   show the source chunks used for the last answer
    /reset     wipe the vector store (forces a re-ingest)
    /quit      exit
"""
from __future__ import annotations

from src.rag import RAG


HELP = """\
Local Wikipedia RAG — type a question and press enter.
Commands:  /context   /reset   /quit
"""


def main() -> None:
    print(HELP)
    rag = RAG()
    last = None
    while True:
        try:
            q = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q in ("/quit", "/exit"):
            break
        if q == "/reset":
            rag.store.reset()
            print("[vector store cleared — run `python -m scripts.run_ingest` to refill]")
            continue
        if q == "/context":
            if not last:
                print("(no previous answer)")
                continue
            for i, h in enumerate(last.hits, 1):
                title = h.metadata.get("title", "?")
                kind = h.metadata.get("type", "?")
                print(f"\n--- [{i}] {kind}: {title}  (distance={h.distance:.3f}) ---")
                print(h.text[:400] + ("..." if len(h.text) > 400 else ""))
            continue

        last = rag.ask(q)
        print(f"\nbot> [{last.routing}] {last.answer}")


if __name__ == "__main__":
    main()
