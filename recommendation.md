# Production Deployment Recommendations

This document describes how the local prototype would have to evolve to be
deployed as a production service. The current implementation is intentionally
single-process, single-user, and runs on a laptop. For real users, the
following changes are recommended.

## 1. Architecture shift: from monolith to services

The prototype lives in one Python process. In production, split into:

| Component | Why split it out |
|-----------|------------------|
| **Ingestion worker** | Crawling/embedding is bursty; should not block serving. Run on a schedule (Airflow / Prefect / GitHub Actions). |
| **Embedding service** | GPU-bound; one shared instance amortizes cold-start across many ingestion jobs. |
| **Vector DB** | Move from the local numpy file to a managed/clustered store (Qdrant, Weaviate, Pinecone, or pgvector on managed Postgres). Persistent disk + backups + proper HNSW indexing instead of brute-force cosine. |
| **LLM serving** | Replace single `ollama serve` with `vLLM` / `TGI` / `Ollama in container` behind a load balancer; supports batching, paged-KV cache, and concurrent users. |
| **API gateway** | A FastAPI service that owns the RAG pipeline, exposes `/ask`, `/sources/{id}`, `/healthz`. |
| **UI** | Streamlit is fine for demos. For real traffic, use Next.js / SvelteKit talking to the API gateway. |

## 2. Model & retrieval upgrades

- **LLM:** `llama3.2:3b` is excellent for a laptop, weak for production
  reasoning. Move to `Llama-3.1-8B-Instruct` or `Mistral-7B-Instruct` on a
  GPU with vLLM; consider `Llama-3.1-70B` if budget allows.
- **Embeddings:** `mxbai-embed-large` is already strong; further upgrade to `bge-large-en-v1.5` or
  `E5-large-v2` for stronger retrieval. Store dimension explicitly so
  re-embedding migrations are tractable.
- **Reranker:** Add a local cross-encoder reranker (`bge-reranker-base`)
  between vector search and the LLM. Top-50 candidates → rerank → top-5 to
  prompt. Big quality jump for comparison queries.
- **Hybrid search:** Combine BM25 (Elasticsearch / OpenSearch / Postgres FTS)
  with vector search; merge with Reciprocal Rank Fusion. Vector-only fails
  on rare proper nouns (e.g. exact monument names).

## 3. Data pipeline & freshness

- Treat Wikipedia ingestion as a **scheduled job** (e.g. weekly).
- Use Wikipedia's `revid` field to detect article updates and skip unchanged
  pages.
- Store raw articles in object storage (S3 / GCS), not in the repo.
- Track schema with **migrations** (Alembic for relational metadata,
  versioned vector-store snapshots for embeddings; record the embedding
  model + dim alongside each snapshot so model swaps are explicit).
- Build a **test set** of canonical Q/A pairs. Run it after every embedding
  or LLM swap. Block deploys on regressions.

## 4. Observability

- **Metrics:** per-stage latency (classify, embed, retrieve, generate),
  tokens-in / tokens-out, retrieval recall@k against the eval set, error
  rates per route. Prometheus + Grafana.
- **Logging:** structured logs (JSON) with a `trace_id` per request linking
  retrieval and generation. ELK / Loki.
- **Tracing:** OpenTelemetry spans for embed → vector search → generate.
- **Quality dashboards:** sample of answers labelled by humans periodically;
  track "I don't know" rate and unsupported-claim rate.

## 5. Security & privacy

- Never log raw user queries in plain text by default; redact PII.
- Rate-limit per IP / per API key.
- Strict CORS and CSRF on the UI.
- LLM prompt-injection defense: keep system prompt server-side, refuse to
  honour `ignore previous instructions`-style content in retrieved docs by
  using delimiter fencing and an instruction-following audit.
- If multi-tenant: per-tenant collections (or a `tenant_id` filter) and
  signed URLs for any retrieved context exposed to the client.

## 6. Cost & performance

- **GPU sharing:** one A10/L4 can serve a 7-8B model for ~50 concurrent
  light users with vLLM. Batch generation requests.
- **Caching layers:**
  1. Embedding cache keyed by `sha256(text)` (Redis / Memcached).
  2. Answer cache keyed by `(model, prompt_hash, retrieved_chunk_ids)`.
  3. HTTP cache headers on idempotent endpoints.
- **Cold-start mitigation:** keep the LLM warm with a low-frequency keepalive
  request; keep the numpy embedding matrix mmap'd in memory across requests.
- **Quantization:** `q4_K_M` GGUF or AWQ INT4 for self-hosted models cuts
  GPU memory ~3× with marginal quality loss.

## 7. Reliability

- Health checks for: Ollama/vLLM up, vector DB reachable, embed service up,
  recent ingest succeeded.
- **Graceful degradation:** if the LLM is down, return retrieved chunks +
  "service degraded" banner instead of erroring.
- **Backups:** snapshot the vector DB nightly; export Wikipedia raw cache to
  cold storage.
- Multi-region: stateless API + replicated vector DB; sticky to nearest
  region for latency.

## 8. CI / CD

- Lint + type-check (`ruff`, `mypy`) on PRs.
- Unit tests for chunker (boundary cases, huge paragraphs) and classifier
  (each spec example gets the correct route).
- An **eval job** that ingests a fixed mini-corpus, asks the spec's example
  questions, and checks that key facts appear in the answer (substring
  assertions or LLM-as-judge against a frozen reference model).
- Container image scanning (Trivy). SBOM generation.

## 9. Compliance / content

- Respect Wikipedia's CC-BY-SA: surface attribution + article URL alongside
  every answer.
- Add an "Open Wikipedia article" link in the sources panel.
- Decide on a content-safety layer (e.g. `Llama Guard`) if exposing to the
  public.

## 10. Phased rollout

| Phase | Scope | Exit criteria |
|-------|-------|---------------|
| 0 (today) | Local prototype, single user | Spec questions answer reasonably. |
| 1 | Containerized monolith + Postgres + pgvector + reranker | 95% of eval set answered correctly; p95 latency < 4 s. |
| 2 | Split services (LLM, embed, API), GPU LLM, hybrid search | 100 concurrent users, p95 < 2 s. |
| 3 | Multi-tenant, observability, scheduled ingest of full topic catalog | SLO 99.5% uptime; weekly eval signoff. |
