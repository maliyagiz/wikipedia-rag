# Product Requirements Document — Local Wikipedia RAG Assistant

> Audience: an AI coding agent (or a fresh engineer) tasked with rebuilding
> this project from scratch. This PRD specifies *what* must exist, *why*, and
> *how it should behave* — not the exact line-by-line implementation.

## 1. Problem statement

University students and curious users want a private, offline, ChatGPT-style
assistant that can answer factual questions about famous **people** and
**places** without sending data to any third-party LLM provider. Existing
hosted assistants are accurate but require cloud APIs; existing open-source
RAG kits (LangChain / LlamaIndex) hide the moving parts behind heavy
abstractions.

The product is a **simplified, transparent Retrieval-Augmented-Generation
system** that:

- Runs entirely on `localhost`.
- Uses **only local models** (LLM + embeddings).
- Sources its knowledge from Wikipedia.
- Exposes its retrieval mechanism so a learner can read the code top-to-bottom.

## 2. Goals & non-goals

### Goals
- Answer free-form questions about ≥20 people and ≥20 places, plus mixed
  comparisons.
- Cite (or expose) the chunks used for an answer.
- Refuse to answer (literal `"I don't know"`) when the indexed context does
  not support a confident answer.
- Be runnable end-to-end from the README in under 10 minutes.
- Use language-native primitives wherever practical (no LangChain wrappers
  around the core retrieval/embedding/generation logic).

### Non-goals
- Not a general web search agent.
- Not a multi-turn reasoning agent (single-turn QA with chat history is enough).
- No fine-tuning. No reranker training.
- Not optimized for production scale.

## 3. Users & top user stories

| # | As a … | I want to … | So that … |
|---|--------|-------------|-----------|
| 1 | curious user | ask "Who is X?" and get a concise factual answer | I learn about a person quickly |
| 2 | curious user | ask "Where is X?" / "What is X?" about a famous place | I learn about a landmark |
| 3 | curious user | ask a comparison ("Compare A and B") across people or places | I see a structured side-by-side |
| 4 | careful user | view the source chunks behind an answer | I can verify accuracy |
| 5 | privacy-conscious user | run the system fully offline | nothing leaves my laptop |
| 6 | developer / student | read the code without LangChain magic | I understand RAG mechanics |
| 7 | developer | wipe the vector store and re-ingest | I can iterate on chunking/embedding |

## 4. Functional requirements

### 4.1 Ingestion
- Pull Wikipedia articles for a configurable list of titles (default: 22
  people + 22 places, including the spec's required minimum set).
- Persist raw articles as JSON to disk so re-embedding is possible without
  re-downloading.
- Be polite to Wikipedia (User-Agent header, small inter-request sleep).

### 4.2 Chunking
- Split articles into chunks of bounded size with overlap.
- Default: ~1200 chars with 150 char overlap.
- Must handle articles tens of thousands of characters long (paragraph-aware
  sliding window with hard fallback for monster paragraphs).

### 4.3 Embedding & storage
- All embeddings produced locally via Ollama (`mxbai-embed-large` by default;
  `nomic-embed-text` is supported as an alternative) — **no external API**.
- Single numpy-backed store (Option B from spec): `embeddings.npy` of shape
  (N, dim) plus a parallel `chunks.json` carrying
  `{title, type ∈ {person, place}, url, chunk_index}` per row.
- Cosine similarity via numpy dot product on a pre-normalized matrix.
- Persists across runs in `vector_store/`. An on-disk JSON embed cache
  (`data/embed_cache.json`) keyed by `sha256(model + formatted_text)` makes
  re-ingest a no-op for chunks already seen.

### 4.4 Query routing & retrieval
- Classify each query as `person`, `place`, or `both` using a rule-based
  classifier (entity name matches > keyword hints > default `both`).
- Retrieve top-k chunks via numpy cosine + boolean type-mask. For `both`,
  fetch top-k from each type and merge by distance.

### 4.5 Generation
- Use a local LLM via Ollama (`llama3.2:3b` default; configurable to `phi3` /
  `mistral`).
- Prompt the model with a strict grounding instruction:
  - Answer from context only.
  - Reply *exactly* `I don't know` when context is insufficient.
  - Be concise; produce side-by-side structure for comparison questions.
- Post-process the LLM output to strip a tag-along trailing
  "I don't know" that small models sometimes append after a real answer
  (`clean_answer()` in `src/generator.py`). Pure refusals are preserved
  unchanged.

### 4.6 Chat interface
Two surfaces, both meeting the spec's "ask / answer / view context / reset"
requirements:

- **CLI** (`app_cli.py`): `/context`, `/reset`, `/quit`.
- **Streamlit** (`app_streamlit.py`):
  - Streamed token-by-token answer.
  - Expandable "Sources" panel per assistant message.
  - Sidebar buttons: "Clear chat" (in-memory) and "Reset vector store".
  - Toggle to hide/show sources globally.

### 4.7 Reset / re-ingest
- `python -m scripts.run_ingest` is incremental: articles already on disk
  are not re-downloaded, articles already in the vector store are not
  re-embedded. Pass `--reset` to wipe the vector store first.
- UI button "Reset vector store" drops the in-memory and on-disk store
  without touching the Wikipedia JSON cache or the embed cache.

## 5. Non-functional requirements

| Concern | Target |
|---------|--------|
| Privacy | 100% local. No outbound calls except Wikipedia at ingest time. |
| Setup time | < 10 min on a fresh laptop with internet (Ollama + 2 models + ingest). |
| Latency | Streaming first token < 3 s on M1/M2 or modern x86. Full answer < 15 s. |
| Footprint | < 5 GB on disk including models. < 4 GB RAM during inference. |
| Code clarity | Every file < ~150 lines, no hidden frameworks for core RAG flow. |

## 6. Out-of-scope (explicit non-features)
- Multi-user auth.
- Server-side persistence beyond the local numpy/JSON files.
- Image / multimodal inputs.
- Fine-tuned reranker.
- Conversational memory across sessions (current chat history is per
  Streamlit session only).

## 7. Success criteria
- All 14 example queries from the spec produce reasonable answers.
- The two failure-case queries return *"I don't know"* (or an honest refusal).
- Instructor can `git clone … && pip install -r requirements.txt && ollama pull … && python -m scripts.run_ingest && streamlit run app_streamlit.py`
  and have a working chat in one go.

## 8. Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Ollama not running | README has a clear `ollama serve` step; UI surfaces connection errors. |
| Wikipedia article rename / redirect | We pass `redirects=1`; classifier still uses the *configured* title for routing hints. |
| Small LLM hallucinates | Strict grounding prompt + low temperature (0.2) + "I don't know" fallback. |
| Chunk too coarse → retrieval misses | Configurable `CHUNK_SIZE` / `CHUNK_OVERLAP`; one-line change + re-ingest. |
| Multi-entity comparison loses one side | Router emits `both` and we fetch top-k *per type* before merging. |

## 9. Future extensions (optional)
- Pluggable reranker (BM25 hybrid, or a local cross-encoder).
- Cache `(query → answer)` keyed by hashed retrieved-chunk-IDs.
- Side-by-side panel comparing two local models on the same question.
- Latency dashboard (per-stage timing).
- Citation footnotes inline in the answer text.
