# AI CLI Gateway — v0.3.0

> Multi-provider AI CLI gateway with pluggable LLM providers, resilience
> helpers, and a modular RAG stack (chunking, embeddings, FAISS vector
> store, and in-memory retrieval for quick prototyping).

---

## What's New in v0.3.0

- **Provider registry** — lazy-loaded providers with deterministic registration (`auto`, `openai`, `gemini`, `cohere`, `perplexity`, `xai`, `zai`, `echo`)
- **CLI RAG flags** — `--rag`, `--rag-docs`, `--rag-chunk-size`, `--rag-top-k` for in-memory retrieval during prompts
- **RAG modules** — `SemanticChunker`, `EmbeddingGenerator`, FAISS `VectorStore`, and `Retriever`
- **Resilience** — retry/backoff engine and response validation

---

## Project Structure

```
src/ai_cli/
├── cli.py                  # CLI entrypoint (ai-cli)
├── core/
│   ├── api.py              # ask() public API
│   ├── resilience.py       # RetryEngine, circuit breaker
│   └── service/
│       ├── ask_service.py  # Chat provider wrapper
│       └── embed_service.py
├── providers/              # LLM provider implementations
│   ├── registry.py         # Provider registration & factory
│   ├── bootstrap.py        # Lazy provider initialization
│   └── *_provider.py
├── rag/
│   ├── chunker.py          # Sliding-window SemanticChunker
│   ├── pipeline.py         # Token-aware sentence chunker
│   ├── embeddings.py       # EmbeddingGenerator (sentence-transformers)
│   ├── vector_store.py     # FAISS-backed VectorStore
│   ├── retriever.py        # Query → embed → search
│   ├── in_memory.py        # Lightweight CLI RAG pipeline
│   └── models.py           # Document, Chunk, RetrievalResult
└── config/rag_config.py    # RAG defaults and RAGConfig dataclass
```

---

## Quickstart

### 1. Install

```bash
git clone https://github.com/yourusername/ai-cli.git
cd ai-cli
poetry install

# Optional RAG extras (FAISS, numpy, document parsers)
poetry install --with rag
```

### 2. Configure environment

```bash
cp .env.example .env
# Set provider API keys, e.g.:
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
```

### 3. CLI examples

```bash
# Single prompt
ai-cli -q "Explain Kubernetes operators"

# Specify provider and model
ai-cli --provider openai --model gpt-4o-mini -q "Hello"

# Interactive chat
ai-cli --interactive --provider openai

# RAG: index docs and query with context
ai-cli --rag --rag-docs docs/manual.md README.md \
  -q "Summarize the security model" --rag-top-k 5

# Interactive RAG (use /index and /search inside the session)
ai-cli --interactive --rag --provider echo
```

### CLI flags (highlights)

| Flag | Description |
|------|-------------|
| `-p`, `--provider` | Provider name (default: `auto`) |
| `-q`, `--prompt` | Prompt text (or pipe via stdin) |
| `-m`, `--model` | Model override |
| `-i`, `--interactive` | Interactive REPL |
| `--rag` | Enable in-memory RAG context injection |
| `--rag-docs` | File paths or raw text to index |
| `--rag-chunk-size` | Chunk size in characters (default: 500) |
| `--rag-chunk-overlap` | Overlap between chunks (default: 50) |
| `--rag-top-k` | Top-k chunks to retrieve (default: 5) |
| `--timeout` | Request timeout in seconds |
| `--stream` | Stream responses when supported |
| `--debug` | Enable debug logging |

---

## Python API

### Ask a provider

```python
from ai_cli.core.api import ask

response = ask(
    prompt="What is RAG?",
    provider="openai",
    model="gpt-4o-mini",
)
print(response)
```

### Chunk documents

```python
from ai_cli.rag import chunk_text, SemanticChunker

chunks = chunk_text("Long document text...", source="manual.md")
# Or use the chunker class directly:
chunker = SemanticChunker(chunk_size=500, overlap=50)
chunks = chunker.chunk_text(text, source="manual.md")
```

### Embeddings + FAISS vector store

```python
from ai_cli.rag import EmbeddingGenerator, VectorStore

embedder = EmbeddingGenerator(model="all-MiniLM-L6-v2", batch_size=32)
vectors = embedder.embed_batch([c.text for c in chunks])

store = VectorStore()
store.create_index(dimension=len(vectors[0]))
store.add_embeddings(vectors, chunks)
store.save()
```

### Semantic retrieval

```python
from ai_cli.rag import Retriever

retriever = Retriever(store=store, embedder=embedder, top_k=5)
context = retriever.build_context("How do we rotate secrets?")
answer = ask(prompt=f"Context:\n{context}\n\nQuestion: How do we rotate secrets?")
```

---

## Supported Providers

| Provider | Env variable | Notes |
|----------|--------------|-------|
| `auto` | — | Falls back through available providers |
| `openai` | `OPENAI_API_KEY` | Chat + embeddings |
| `gemini` | `GEMINI_API_KEY` | Google Gemini |
| `cohere` | `COHERE_API_KEY` | Cohere chat |
| `perplexity` | `PERPLEXITY_API_KEY` | Search-augmented |
| `xai` | `XAI_API_KEY` | xAI Grok |
| `zai` | `ZAI_API_KEY` | Z.AI |
| `echo` | — | Local test provider (no API key) |

---

## Testing

```bash
pytest tests/ -v
pytest tests/test_enhanced.py -v   # RAG chunking + embeddings
pytest tests/test_providers.py -v  # Provider contracts
```

---

## Documentation

- [docs/USAGE.md](docs/USAGE.md) — CLI workflows and examples
- [docs/API.md](docs/API.md) — Python API reference
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) — Contributing guide

---

## License

MIT License — see [LICENSE](LICENSE) for details.
