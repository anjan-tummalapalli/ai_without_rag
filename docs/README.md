# AI CLI Chat Tool

## Overview

The AI CLI Chat Tool is a command-line interface for interacting with
multiple AI providers (OpenAI, Gemini, Cohere, Perplexity, xAI, Z.AI, and
more). It includes a modular RAG stack for document chunking, embedding
generation, FAISS vector storage, and retrieval-augmented prompting.

## Features

- Multi-provider routing with automatic fallback (`auto` provider)
- Interactive REPL with provider/model switching
- In-memory RAG for quick document indexing and context injection
- Production RAG modules: chunking, sentence-transformers embeddings,
  FAISS vector store, semantic retriever
- Retry/backoff resilience and response validation

## Installation

```bash
git clone https://github.com/yourusername/ai_cli.git
cd ai_cli
pip install -e .
# Or with Poetry:
poetry install

# Optional RAG dependencies (FAISS, numpy, etc.)
poetry install --with rag
```

Set API keys in a `.env` file (see `.env.example`):

```
OPENAI_API_KEY=...
GEMINI_API_KEY=...
COHERE_API_KEY=...
```

## Usage

```bash
# Single prompt
python -m ai_cli -q "What is the capital of France?"

# With provider and model
python -m ai_cli --provider openai --model gpt-4o-mini -q "Hello"

# Interactive session
python -m ai_cli --interactive --provider openai

# RAG-enabled query
python -m ai_cli --rag --rag-docs docs/ \
  -q "Summarize the deployment process" --rag-top-k 5
```

## RAG Workflow (Python)

```python
from ai_cli.rag import chunk_text, EmbeddingGenerator, VectorStore, Retriever
from ai_cli.core.api import ask

# 1. Chunk
chunks = chunk_text(open("docs/manual.md").read(), source="manual.md")

# 2. Embed
embedder = EmbeddingGenerator(model="all-MiniLM-L6-v2")
vectors = embedder.embed_batch([c.text for c in chunks])

# 3. Index
store = VectorStore()
store.create_index(dimension=len(vectors[0]))
store.add_embeddings(vectors, chunks)

# 4. Retrieve + generate
retriever = Retriever(store, embedder)
context = retriever.build_context("How does authentication work?")
print(ask(prompt=f"Context:\n{context}\n\nQuestion: How does authentication work?"))
```

## Documentation

- [USAGE.md](USAGE.md) — Detailed CLI usage
- [API.md](API.md) — Python API reference
- [DEVELOPMENT.md](DEVELOPMENT.md) — Contributing guidelines

## License

MIT License — see [LICENSE](../LICENSE).
