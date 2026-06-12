# Development Guidelines for AI CLI

## Introduction

Guidelines for contributing to the AI CLI project, including environment
setup, coding standards, and RAG module development.

## Setting Up Your Development Environment

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/ai_cli.git
   cd ai_cli
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   poetry install
   poetry install --with rag   # FAISS, numpy, langchain, etc.
   poetry install --with dev   # pytest, ruff, mypy
   ```

4. **Environment variables**

   Create a `.env` file (never commit it):

   ```
   OPENAI_API_KEY=...
   EMBEDDING_MODEL=all-MiniLM-L6-v2
   RAG_CHUNK_SIZE=500
   RAG_CHUNK_OVERLAP=50
   RAG_TOP_K=5
   ```

## Project Layout

```
src/ai_cli/
├── cli.py              # CLI entrypoint
├── core/               # Public API, resilience, services
├── providers/          # LLM provider implementations + registry
├── rag/                # Chunking, embeddings, vector store, retrieval
├── config/             # RAG configuration
├── plugins/            # Plugin hooks
├── telemetry/          # Metrics and monitoring
└── utils/              # Validation, secrets
tests/                  # pytest test suite
docs/                   # User and API documentation
```

## Adding a Provider

1. Create `src/ai_cli/providers/my_provider.py` implementing `chat()` or `send()`
2. Register in `src/ai_cli/providers/loader.py`
3. Add tests in `tests/test_providers.py`
4. Document the required env variable in README and docs/USAGE.md

Provider registration is lazy — `bootstrap.init_providers()` loads all
providers once per process.

## RAG Development

### Chunking

- **Simple:** `rag/chunker.py` — sliding-window `SemanticChunker`
- **Advanced:** `rag/pipeline.py` — token-aware sentence chunker with spans
- **Models:** `rag/models.py` — `Document`, `Chunk`, `RetrievalResult`

Defaults are in `config/rag_config.py`. Override via `RAGConfig.from_env()`.

### Embeddings

`EmbeddingGenerator` wraps sentence-transformers with batching and
normalization. For tests, mock `SentenceTransformer` (see
`tests/test_enhanced.py`).

### Vector Store

`VectorStore` uses FAISS IndexFlatL2. Requires `faiss-cpu` and `numpy`.
Use in-memory fixtures in CI to avoid disk I/O.

### CLI RAG vs Production RAG

| Layer | Module | Use case |
|-------|--------|----------|
| CLI in-memory | `rag/in_memory.py` | Quick prototyping, no extra deps |
| Production | `embeddings.py` + `vector_store.py` + `retriever.py` | Semantic search with FAISS |

## Testing

```bash
pytest tests/ -v
pytest tests/test_enhanced.py -v        # RAG smoke tests
pytest tests/test_provider_contract.py  # Provider interface
pytest tests/test_imports.py            # Module import checks
```

Guidelines:

- Mock external API calls and heavy models in unit tests
- Use the `echo` provider for integration tests without API keys
- Mark slow/integration tests with `@pytest.mark.integration`

## Coding Standards

- Python 3.10+ with type hints
- Line length: 80 (black/ruff)
- Match existing naming and module structure
- Keep changes focused — avoid unrelated refactors in the same PR
- Add docstrings for public APIs; comments only for non-obvious logic

## CLI Development

The CLI lives in `src/ai_cli/cli.py`. Key helpers:

- `_build_ask_kwargs()` — adapts to `ask()` signature via introspection
- `_invoke_with_retries()` — retry transient errors with exponential backoff
- `run_interactive()` — REPL with `/index`, `/search`, provider switching

Entry point: `ai-cli = ai_cli.cli:main` (see `pyproject.toml`).

## Documentation

When adding features, update:

- `README.md` — overview and quickstart
- `docs/USAGE.md` — CLI examples
- `docs/API.md` — Python API reference
- `docs/DEVELOPMENT.md` — this file, if workflow changes

## Contributing

1. Branch: `git checkout -b feature/my-feature`
2. Write tests for new behavior
3. Run `pytest` and fix lint issues (`ruff check`, `black`)
4. Open a PR with a clear description and test plan

## License

MIT — see [LICENSE](../LICENSE).
