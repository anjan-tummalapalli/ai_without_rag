# AI CLI Gateway — v0.3.0

> Enterprise-grade multi-provider AI CLI gateway with Security &
> Identity Management, Tool Calling, ReAct Agents, and an enhanced
> Developer Experience — now with production-ready Advanced RAG:
> robust chunking, embedding orchestration, and vector DB querying.

---

## What's New in v0.3.0

This release extends the multi-provider routing and resilience base
with production-grade RAG features focused on:

- Reliable semantic chunking (token-aware, overlap, adaptive sizing,
  OCR-aware)
- Scalable embedding pipelines (batching, async, provider + local
  models)
- Pluggable indexer interface with hybrid search (vector +
  metadata), filters, and upsert/delete
- CLI and Python APIs to ingest, re-embed, index, query, and monitor
  vector stores
- Tests, metrics, and docs for integration with FAISS, Qdrant, Chroma,
  pgvector

---

## Project Structure (RAG-focused files)

```
src/ai_cli/rag/
├── __init__.py
├── pipeline.py           # RAGPipeline: retrieval, augmentation,
                           # grounding, caching
├── ingest.py             # CLI/programmatic ingestion: parse -> chunk
                           # -> embed -> index
├── chunking.py           # SemanticChunker: token/char chunkers,
                           # overlap, adaptive sizing, OCR hooks
├── embeddings.py         # EmbedderRegistry: provider adapters,
                           # sentence-transformers adapter, batching
└── indexers/
  ├── base.py             # Indexer interface: upsert(), search(),
                           # delete(), info()
  ├── faiss_indexer.py    # Local FAISS indexer with disk snapshots
                           # + memory map
  ├── qdrant_indexer.py   # Qdrant client indexer with retry/CB and
                           # hybrid filtering
  ├── chroma_indexer.py   # Chroma client/local indexer
  └── pgvector_indexer.py # pgvector (Postgres) indexer with SQL +
                           # vector ops
```

Files changed/added for Advanced RAG:
- src/ai_cli/rag/{pipeline.py, ingest.py, chunking.py, embeddings.py}
- src/ai_cli/rag/indexers/{faiss_indexer.py, qdrant_indexer.py,
  chroma_indexer.py, pgvector_indexer.py}
- CLI: src/ai_cli/cli.py (new flags/subcommands for RAG)
- Config profiles updated: src/ai_cli/config/profiles.py (rag options)
- Docs: docs/rag/ (usage, deployment, qdrant-compose, contributor
  guide)

---

## Core RAG Concepts — Implementation Notes

- Chunking
  - SemanticChunker supports token-aware chunking (uses tokenizers
    when available), character fallback, and overlap parameters.
  - Adaptive chunk sizing: tune per-document type (code, long text,
    PDF pages) and enforce max token limits for model context.
  - Source metadata preserved (path, page, byte-range, text-hash) to
    support precise citations and provenance.

- Embeddings
  - EmbedderRegistry exposes adapters:
    - sentence-transformers (local models, batch, GPU/CPU)
    - provider-based (OpenAI/Anthropic) with rate-limited batched
      requests
  - Supports configurable batch sizes, parallel workers, and
    retry/backoff.
  - Re-embedding workflow available to migrate embeddings between
    models.

- Indexers & Querying
  - Indexer interface: upsert(documents), search(
    query_embedding, k, filter, hybrid_weight), delete(ids), info()
  - Qdrant indexer implements vector search with cosine/inner
    product, JSON metadata filtering, hybrid scoring, pagination and
    cursor support.
  - FAISS indexer supports on-disk indices and periodic snapshot/
    restore.
  - pgvector indexer exposes SQL-friendly hybrid queries for
    join/filter scenarios.

- Pipeline Behavior
  - RAGPipeline orchestrates: retrieve candidates -> dedupe -> augment
    prompt with trimmed context & citations -> LLM call
  - Cache layer for embeddings and retrieval results to reduce cost
    and latency
  - RBAC and audit pipeline wrap ingest/index operations

---

## Quickstart (Updated for Advanced RAG)

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/ai-cli.git
cd ai-cli

poetry install

# Optional RAG extras
pip install "faiss-cpu"                # local FAISS
pip install qdrant-client              # Qdrant remote
pip install chromadb                   # Chroma
pip install "psycopg2-binary"          # pgvector (Postgres)
pip install sentence-transformers      # local embeddings models
```

Or use extras:
```bash
poetry add .[rag]
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill provider API keys plus vector DB config
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
DATABASE_URL=postgresql://user:pass@localhost:5432/pgdb
CHROMA_SETTINGS='{"path":"/data/chroma"}'
EMBED_PROVIDER=openai
EMBED_MODEL=text-embedding-3-small
```

### 3. CLI Examples

```bash
# Single prompt
ai-cli -q "Explain Kubernetes operators"

# Index a docs folder to Qdrant using sentence-transformers
ai-cli rag index docs/ --db qdrant --namespace docs-2026 \
  --embed-model all-MiniLM-L6-v2 --chunk-size 1000 \
  --chunk-overlap 200

# Query RAG pipeline (auto retrieval + augmentation)
ai-cli --rag --rag-db qdrant --rag-namespace docs-2026 \
  -q "Summarize our security model" --rag-k 5

# Re-embed an index with a new model
ai-cli rag reembed --db qdrant --namespace docs-2026 \
  --from-model all-MiniLM-L6-v2 --to-model multi-qa-MiniLM
```

Usage instructions (common workflows):
- Index a folder to a vector DB:
  ai-cli rag index <path> --db <db> --namespace <name> --embed-model <m>
- Query a namespace:
  ai-cli rag query --db <db> --namespace <name> --k <N> --filter '<JSON>' \
    --hybrid-weight 0.5 "QUERY"
- Re-embed an existing namespace:
  ai-cli rag reembed --db <db> --namespace <name> --from-model X --to-model Y
- Check index status:
  ai-cli rag status --db <db> --namespace <name>

---

## CLI RAG Commands & Flags (Highlights)

- ai-cli rag index <path> [--db DB] [--namespace NAME]
  [--embed-model MODEL] [--chunk-size N] [--chunk-overlap M]
  [--embed-batch-size B]
- ai-cli rag query --db DB --namespace NAME --k N
  --filter '{"key":"value"}' [--hybrid-weight 0.5] "QUERY"
- ai-cli rag status --db DB --namespace NAME
- ai-cli rag reembed --db DB --namespace NAME --from-model X
  --to-model Y

Common flags:
- --db qdrant|faiss|chroma|pgvector
- --embed-model <model>
- --embed-batch-size (default: 256)
- --chunk-size (default: 1000 chars)
- --chunk-overlap (default: 200 chars)
- --hybrid-weight (0..1) (vector vs metadata weighting)
- --filter JSON (metadata filter)

---

## Python API (Examples)

```python
from ai_cli.rag.ingest import ingest_documents, chunk_documents, \
  embed_documents
from ai_cli.rag.pipeline import RAGPipeline
from ai_cli.rag.indexers.registry import IndexerRegistry
from ai_cli.rag.embeddings import EmbedderRegistry

# Ingest (parse -> chunk -> embed -> index)
ingest_documents(path="docs/", db="qdrant", namespace="project-arch",
  embed_model="all-MiniLM-L6-v2", chunk_size=1000, chunk_overlap=200)

# Programmatic chunking + embedding
chunks = chunk_documents("docs/manual.md", chunk_size=1200,
  overlap=250)
embeds = embed_documents(chunks, model="all-MiniLM-L6-v2",
  batch_size=128)

# Direct search with indexer
idx = IndexerRegistry.get("qdrant")
results = idx.search(query="rotate secrets in production", k=6,
  embed_model="all-MiniLM-L6-v2", hybrid_weight=0.7,
  filter={"path": {"$eq": "docs/manual.md"}})

# RAG pipeline ask
pipeline = RAGPipeline(provider="openai",
  embed_model="all-MiniLM-L6-v2", indexer="qdrant",
  namespace="project-arch")
answer = pipeline.ask("How do we rotate secrets in production?", k=6)
print(answer.text)
```

Key APIs:
- ingest_documents(path, db, namespace, embed_model, chunk_size,
  chunk_overlap, metadata_fn)
- chunk_documents(file, chunk_size, overlap) -> list[Chunk]
- embed_documents(chunks, model, batch_size) -> list[Embeddings]
- Indexer.upsert/docs/search/delete/info
- RAGPipeline.ask(query, k, filter, hybrid_weight)

---

## Testing & CI

New tests:
- rag/test_ingest.py (parsing, chunking, metadata)
- rag/test_embeddings.py (batching, provider fallback)
- rag/test_indexers.py (upsert/search/delete with mocks)
- rag/test_pipeline.py (end-to-end with mocked providers)

Run:
```bash
pytest tests/test_rag.py -v
pytest tests/test_indexers.py -k qdrant -v
```

---

## Observability & Monitoring

New metrics:
- ai_rag_documents_indexed_total
- ai_rag_index_latency_seconds
- ai_rag_retrieval_requests_total
- ai_rag_retrieval_latency_seconds
- ai_rag_embedding_requests_total
- ai_rag_hybrid_query_hits_total

Add Prometheus scrape configs for indexer services when deployed.

---

## Security & Governance

- All ingest/index operations go through RBAC & AuditLogger
- Sensitive fields can be redacted during chunking (PromptSanitiser
  hooks)
- Per-identity budget and rate limits apply to embedding and
  retrieval operations

---

## Deployment Notes

- Qdrant: prefer stateful deployment (docker-compose/helm); include
  snapshot/backup strategy for collections
- FAISS: use mmap snapshots for large indices; maintain periodic
  snapshot cron
- pgvector: use Postgres migrations for schema and index creation
- Chroma: consider server mode for multi-instance setups

See docs/rag/ for sample docker-compose, helm charts, and scaling
guides.

---

## Roadmap (RAG)

- ✅ Semantic chunking, embeddings, FAISS, Qdrant, Chroma, pgvector
  support
- ✅ CLI ingestion and indexing, hybrid querying
- ✅ RAGPipeline with prompt augmentation and caching
- ⬜ Vector DB auto-scaling and multi-region replication
- ⬜ Streaming retrieval + incremental re-ranking
- ⬜ Multilingual embedding pipelines and online re-training

---

## License

MIT License — see [LICENSE](LICENSE) for details.

