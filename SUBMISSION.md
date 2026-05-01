# BLG483E – Project 3 Submission

**Course:** AI-Aided Computer Engineering (BLG483E)
**Project:** Build a Local Wikipedia RAG Assistant
**Student:** Muhammet Ali Yağız
**Student ID:** 820220327
**Term:** Spring 2026

---

## Links

- **GitHub repository:** https://github.com/maliyagiz/wikipedia-rag
- **Demo video (Loom):** https://www.loom.com/share/e2ceabb7179c4aad997df0480eca9c45

---

## What's in this zip

This archive contains the full source code of the project. The two
runtime-generated directories (`data/` for cached Wikipedia JSON and
`vector_store/` for the persisted numpy embeddings) are intentionally
excluded — they are rebuilt on first run via:

```
python -m scripts.run_ingest
```

To run the project locally, follow the step-by-step instructions in
`README.md`. Everything is reproducible from a fresh clone in under
10 minutes (plus the one-off ~30 minute Wikipedia embedding pass).

## Files included

- `README.md` — installation, run, examples, troubleshooting
- `product_prd.md` — product requirements doc
- `recommendation.md` — production deployment recommendations
- `requirements.txt` — Python dependencies
- `config.py` — central configuration
- `app_streamlit.py` — Streamlit chat UI
- `app_cli.py` — CLI chat interface
- `src/` — pipeline modules (ingest, chunker, embedder, vector store, retriever, generator, RAG glue)
- `scripts/run_ingest.py` — end-to-end ingestion script
- `scripts/diagnose.py` — retrieval-quality diagnostic tool
