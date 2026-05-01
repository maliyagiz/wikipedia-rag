# Local Wikipedia RAG Assistant

A fully local, ChatGPT-style assistant that answers questions about famous
**people** and **places** by retrieving Wikipedia content from a local vector
store and generating grounded answers with a local LLM via [Ollama].
Course project for **BLG483E – AI-Aided Computer Engineering**, ITU.

No external LLM API is used. Embeddings, retrieval, and generation all run on
`localhost`.

## Architecture

```
 ┌────────────┐   ┌──────────┐   ┌──────────────────┐   ┌──────────┐   ┌────────┐
 │ Wikipedia  │──▶│  Chunk   │──▶│  Embed           │──▶│  numpy   │──▶│  RAG   │
 │  REST API  │   │ (custom) │   │ mxbai-embed-large│   │ store    │   │ Pipe-  │
 └────────────┘   └──────────┘   │ via Ollama       │   │ + cache  │   │  line  │
                                 └──────────────────┘   └──────────┘   └───┬────┘
                                                                           │
                                              ┌────────────────────────────┴──┐
                                              │  Query → classify             │
                                              │  (person / place / both)      │
                                              │  → numpy cosine + type filter │
                                              │  → top-k chunks               │
                                              │  → llama3.2:3b (Ollama)       │
                                              │  → grounded answer            │
                                              └────────────┬──────────────────┘
                                                           │
                                              CLI  (app_cli.py)
                                              UI   (app_streamlit.py)
```

## Project layout

```
wikipedia-rag/
├── app_cli.py             # Minimal CLI chat
├── app_streamlit.py       # Streamlit chat UI (streaming, sources panel)
├── config.py              # Models, paths, chunking config, entity lists
├── requirements.txt
├── README.md
├── product_prd.md         # Product requirements doc (for AI to rebuild it)
├── recommendation.md      # Production deployment recommendations
├── scripts/
│   └── run_ingest.py      # End-to-end ingestion pipeline
├── src/
│   ├── ingest.py          # Wikipedia REST API → JSON files
│   ├── chunker.py         # Hand-written paragraph-aware sliding window
│   ├── embedder.py        # Ollama /api/embeddings client (with optional cache)
│   ├── embed_cache.py     # On-disk JSON cache: sha256(model+text) → vector
│   ├── vector_store.py    # numpy-backed cosine store (Option B: type metadata)
│   ├── classifier.py      # Rule-based person/place/both router
│   ├── retriever.py       # classify → embed → numpy top-k with type filter
│   ├── generator.py       # Ollama /api/generate with strict grounding prompt
│   └── rag.py             # Glue: end-to-end RAG pipeline
├── data/                  # Persisted Wikipedia JSON + embed_cache.json
└── vector_store/          # embeddings.npy + chunks.json (created on ingest)
```

## Prerequisites

- **Python 3.10+**
- **[Ollama](https://ollama.com/)** installed and running locally
- ~5 GB free disk for models + data

## 1. Install dependencies

```bash
git clone <your-fork-url> wikipedia-rag
cd wikipedia-rag
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## 2. Run the local model

Install and start Ollama, then pull the LLM and the embedding model:

```bash
# In a separate terminal — keep it running:
ollama serve

# In your project terminal:
ollama pull llama3.2:3b          # default LLM (configurable in config.py)
ollama pull mxbai-embed-large     # embedding model
```

Verify Ollama is reachable: <http://localhost:11434>.

You can swap to `phi3` or `mistral` by editing `LLM_MODEL` in [`config.py`](config.py).

## 3. Ingest Wikipedia data

This downloads the 22 people + 22 places listed in `config.py`, chunks them,
embeds every chunk locally, and persists the embeddings to disk as a numpy
array. Every embedding is also cached to `data/embed_cache.json` so future
re-ingests (after a chunker change, model swap, or accidental wipe) reuse
the cached vectors and skip Ollama entirely.

```bash
python -m scripts.run_ingest             # incremental: skip what's already done
python -m scripts.run_ingest --reset     # wipe vector store and re-embed
```

You should see ~3000 chunks total across the 44 articles. The first ingest
takes ~30 minutes on CPU (most of which is the Ollama embedding pass).
Subsequent re-ingests are nearly instant because every chunk hits the disk
embed cache.

## 4. Start the application

**Streamlit UI (recommended):**

```bash
streamlit run app_streamlit.py
```

Open the URL Streamlit prints (defaults to <http://localhost:8501>).

**CLI:**

```bash
python app_cli.py
```

CLI commands: `/context` (show last sources), `/reset` (wipe vector store),
`/quit` (exit).

## 5. Example queries

People:

- *Who was Albert Einstein and what is he known for?*
- *What did Marie Curie discover?*
- *Why is Nikola Tesla famous?*
- *Compare Lionel Messi and Cristiano Ronaldo.*
- *What is Frida Kahlo known for?*

Places:

- *Where is the Eiffel Tower located?*
- *Why is the Great Wall of China important?*
- *What is Machu Picchu?*
- *What was the Colosseum used for?*
- *Where is Mount Everest?*

Mixed:

- *Which famous place is located in Turkey?*
- *Which person is associated with electricity?*
- *Compare Albert Einstein and Nikola Tesla.*
- *Compare the Eiffel Tower and the Statue of Liberty.*

Failure cases (should return *"I don't know"*):

- *Who is the president of Mars?*
- *Tell me about a random unknown person John Doe.*

## How it works (short tour)

1. **Ingest** (`src/ingest.py`) — calls the MediaWiki API directly with
   `prop=extracts&explaintext=1` and saves each article as JSON under
   `data/{person|place}/`.
2. **Chunk** (`src/chunker.py`) — paragraph-aware sliding window:
   `CHUNK_SIZE=1200`, `CHUNK_OVERLAP=150`. Designed for very long documents.
3. **Embed & store** (`src/embedder.py` + `src/vector_store.py`) — embeddings
   come from Ollama's `mxbai-embed-large` (1024-d). We persist them as a
   single numpy array (`embeddings.npy`) plus a metadata sidecar
   (`chunks.json`) — Option B in the spec, with a `type` field per chunk
   filtered at query-time. (We started with Chroma but its Rust HNSW backend
   on Windows had path-encoding/HNSW-segment bugs; numpy gives full
   transparency for ~3000 vectors.)
4. **Retrieve** (`src/classifier.py` + `src/retriever.py`) — a rule-based
   classifier inspects the query for entity name matches and intent words;
   it returns `person`, `place`, or `both`. The retriever embeds the query,
   then runs cosine similarity over the numpy array, applying a boolean
   type-mask before argpartition (and merging both masks when needed).
5. **Generate** (`src/generator.py`) — Ollama's `/api/generate` is called with
   a strict grounding prompt (must answer from context only or say
   *"I don't know"*).
6. **UI** (`app_streamlit.py`) — token-streamed responses, expandable
   "Sources" panel showing each retrieved chunk with title/type/distance,
   and chat-history memory for the session.

## Resetting

- *Wipe vector store, keep raw Wikipedia JSON:* `/reset` in the CLI, or the
  "Reset vector store" button in the sidebar.
- *Wipe everything:* delete `data/` and `vector_store/`, then re-run ingest.

## Demo video

**Demo:** <https://www.loom.com/share/e2ceabb7179c4aad997df0480eca9c45>

## Troubleshooting

- `Connection refused` to `localhost:11434` → start `ollama serve`.
- Empty answers → make sure `python -m scripts.run_ingest` finished
  successfully and the sidebar shows a non-zero "Indexed chunks" count.
- Slow first answer → `llama3.2:3b` is being loaded into RAM by Ollama; the
  second answer will be much faster.
