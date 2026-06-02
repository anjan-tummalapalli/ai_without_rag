# Development Guidelines for AI CLI

## Introduction
This document outlines guidelines for contributing to the AI CLI project with a focus on Advanced RAG (Retrieval-Augmented Generation): document chunking, embedding, and vector-database querying. It covers environment setup, new dependencies, recommended defaults for chunking & embeddings, vector DB options, CLI workflows, testing and maintenance.

## Setting Up Your Development Environment

1. **Clone the Repository**
   ```
   git clone https://github.com/yourusername/ai_cli.git
   cd ai_cli
   ```

2. **Create a Virtual Environment**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**
   Add RAG-specific packages and run:
   ```
   pip install -r requirements.txt
   pip install sentence-transformers faiss-cpu python-dotenv
   # Optionally for cloud/vector services:
   pip install pinecone-client pymilvus openai
   ```
   Update `requirements.txt` accordingly.

4. **Set Up Environment Variables**
   Create a `.env` file in the root directory (based on `.env.example`) and set these variables as needed:
   - OPENAI_API_KEY=...
   - EMBEDDING_MODEL=all-MiniLM-L6-v2 (or your preferred model)
   - VECTOR_DB=faiss|pinecone|milvus
   - PINECONE_API_KEY=...
   - PINECONE_ENV=...
   - MILVUS_HOST=...
   - MILVUS_PORT=...
   - CHUNK_SIZE=1000
   - CHUNK_OVERLAP=200

   Ensure `.env` is not committed.

## Advanced RAG: Chunking

- Purpose: split long documents into semantically-coherent, overlapping chunks that fit embedding/token limits.
- Defaults:
  - chunk_size: 1000 characters (or ~512 tokens)
  - overlap: 200 characters
- Strategy:
  - Prefer sentence/paragraph boundaries when possible.
  - Keep overlap to allow context carry-over (100–300 chars).
- Example chunker (high-level):
  ```
  def chunk_text(text, chunk_size=1000, overlap=200):
      # split by paragraphs/sentences, accumulate until chunk_size reached,
      # then backtrack by overlap for next chunk.
      return list_of_chunks
  ```
- Tips:
  - For code/docs, use logical delimiters (headers, code fences).
  - For very large corpora, process files in streaming/batch mode to avoid memory spikes.

## Advanced RAG: Embeddings

- Models:
  - Local: sentence-transformers (e.g., all-MiniLM-L6-v2) — fast, cheap.
  - Cloud: OpenAI or other API-based embedding providers — higher quality for some tasks.
- Embedding pipeline:
  1. Load chunk.
  2. Normalize text (strip, normalize whitespace).
  3. Compute embedding vector.
  4. Store vector + metadata (source, chunk_id, text, tokens).
- Example (conceptual):
  ```
  from sentence_transformers import SentenceTransformer
  model = SentenceTransformer("all-MiniLM-L6-v2")
  vectors = model.encode(chunks, show_progress_bar=True)
  ```
- Env var `EMBEDDING_MODEL` controls model selection in the CLI.

## Advanced RAG: Vector DB Querying

- Supported backends:
  - faiss (local, file-backed index) — good for local dev and CI.
  - pinecone (managed) — production ready, scalable.
  - milvus (self-hosted) — scalable open-source alternative.
- Indexing:
  - Store: vector, id, metadata (source/path, chunk_index, text, created_at)
  - Persist indexes to disk (faiss) or to remote service credentials via env.
- Retrieval:
  - Use cosine similarity or inner-product depending on vector normalization.
  - Return top-k results with similarity scores.
- Example pseudo-workflow:
  ```
  # build index
  vectors = embed(chunks)
  index.upsert(vectors, metadata)

  # query
  q_vec = embed([query])[0]
  hits = index.search(q_vec, top_k=5)
  ```
- Connection config:
  - For FAISS: file path for persisted index (e.g., data/faiss_index.pkl)
  - For Pinecone: set PINECONE_API_KEY and PINECONE_ENV
  - For Milvus: MILVUS_HOST / PORT and collection name

## CLI Commands (new / updated)

- Build or refresh vector index:
  ```
  ai_cli build-index --source docs/ --db faiss \
    --chunk-size 1000 --overlap 200 --embed-model all-MiniLM-L6-v2
  ```
- Query the index:
  ```
  ai_cli query --db faiss --k 5 "How do I set up environment variables?"
  ```
- Delete or reindex:
  ```
  ai_cli index-delete --db faiss
  ai_cli rebuild-index --source docs/
  ```
- All commands honor `.env` and CLI overrides. Add `--dry-run` to inspect chunking/embedding without writing.

## Testing

- Add tests covering:
  - chunking behavior (edge cases, overlap)
  - embedding pipeline integration (mock models if needed)
  - vector DB adapters (use faiss in-memory for CI)
  - end-to-end RAG query flow (mock external APIs)
- Example test command:
  ```
  poetry run pytest tests/test_rag.py -v
  ```
- Continuous Integration:
  - Use small faiss-backed fixtures for CI to avoid external dependencies.
  - Mock network calls for Pinecone/OpenAI in unit tests.

## Documentation Updates

- Update README and docs/ with:
  - New env vars and configuration examples.
  - Vector DB setup guides (faiss local, Pinecone quickstart, Milvus notes).
  - Examples of chunking parameters and their trade-offs.
  - CLI usage examples for build-index and query flows.
- Add `docs/architecture/rag.md` describing:
  - data flow (ingest -> chunk -> embed -> index -> query -> re-rank -> generate)
  - metadata schema stored with vectors
  - performance characteristics and scaling tips

## Coding Standards (RAG-specific)

- Ensure adapters for each vector DB share a common interface (upsert, search, delete, persist).
- Keep embedding logic abstracted so swapping models/providers is a config change.
- Add type hints and docstrings to new modules; provide small integration examples in docs.

## Contributing

1. Branching:
   ```
   git checkout -b feature/advanced-rag
   ```
2. Commit messages:
   - Use clear messages, e.g., "feat(rag): add faiss adapter and chunker"
3. Tests & CI:
   - Include tests for new behavior and ensure they run in CI.
4. PR Description:
   - Describe reindexing/migration steps (if any), env changes, and CLI examples.

## Best Practices

- Reindex when changing chunking or embedding model.
- Version vector schema (metadata keys + embedding dim) in index metadata for migrations.
- Keep chunks and raw sources linked in metadata to allow provenance and reassembly.
- Small focused commits and code reviewers should validate both accuracy and cost implications of embedding/model choices.

## Conclusion
This update adds clear, testable workflows and tooling to support Advanced RAG features: chunking, embeddings, and vector DB querying. Follow the instructions above to set up dev environments, run the new CLI commands, and add tests for any new functionality.
