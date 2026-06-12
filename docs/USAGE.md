# Usage Instructions for AI CLI

## Overview

The AI CLI supports single-shot prompts, interactive chat, and optional
in-memory RAG (Retrieval-Augmented Generation). For production semantic
search, use the Python RAG modules with FAISS and sentence-transformers.

## Installation

```bash
git clone <repository-url>
cd ai_cli
pip install -e .

# RAG extras (FAISS, numpy, sentence-transformers)
pip install faiss-cpu numpy sentence-transformers
```

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | OpenAI chat and embeddings |
| `GEMINI_API_KEY` | Google Gemini |
| `COHERE_API_KEY` | Cohere |
| `PERPLEXITY_API_KEY` | Perplexity |
| `XAI_API_KEY` | xAI Grok |
| `ZAI_API_KEY` | Z.AI |
| `RAG_CHUNK_SIZE` | Override default chunk size (500) |
| `RAG_CHUNK_OVERLAP` | Override default overlap (50) |
| `RAG_TOP_K` | Override default top-k (5) |

## CLI Usage

Base invocation:

```bash
python -m ai_cli --help
ai-cli --help
```

### Single-shot prompt

```bash
ai-cli --provider openai --prompt "What is the capital of France?"
```

Pipe a prompt via stdin:

```bash
echo "Summarize this" | ai-cli --provider echo
```

### Interactive chat

```bash
ai-cli --interactive --provider openai
```

Interactive commands:

| Command | Action |
|---------|--------|
| `/switch <provider>` | Change provider |
| `/model <model>` | Set model override |
| `/profile <name>` | Set profile |
| `/stream` | Toggle streaming |
| `/index <file\|text>` | Index into RAG store |
| `/search <query>` | Show retrieved context |
| `/exit`, `/quit` | Exit session |

### In-memory RAG

Index documents and query with retrieved context:

```bash
# Index files and ask with RAG context
ai-cli --rag \
  --rag-docs docs/manual.md docs/faq.md \
  --rag-chunk-size 800 \
  --rag-chunk-overlap 100 \
  --rag-top-k 5 \
  -q "How do I configure authentication?"
```

Flags:

- `--rag` — inject retrieved context into the prompt
- `--rag-docs` — file paths or raw text strings to index before querying
- `--rag-chunk-size` — characters per chunk (default: 500)
- `--rag-chunk-overlap` — overlap between chunks (default: 50)
- `--rag-top-k` — number of chunks to retrieve (default: 5)

> **Note:** CLI RAG uses deterministic hash-based embeddings for
> lightweight prototyping. For semantic search, use the Python API with
> `EmbeddingGenerator` and `VectorStore` (see below).

## Python RAG Pipeline

### Chunking

```python
from ai_cli.rag import chunk_text, SemanticChunker

chunks = chunk_text("Long text...", source="doc.md", chunk_size=800, overlap=200)
```

### Embeddings

```python
from ai_cli.rag import EmbeddingGenerator

embedder = EmbeddingGenerator(model="all-MiniLM-L6-v2", batch_size=32)
vectors = embedder.embed_batch(["text one", "text two"])
```

### FAISS vector store

```python
from ai_cli.rag import VectorStore

store = VectorStore()
store.create_index(dimension=384)
store.add_embeddings(vectors, chunks)
store.save()

# Later
store.load()
results = store.search(query_embedding, top_k=5)
```

### Retriever

```python
from ai_cli.rag import Retriever

retriever = Retriever(store=store, embedder=embedder, top_k=5)
context = retriever.build_context("Your question here")
```

## Configuration

RAG defaults live in `src/ai_cli/config/rag_config.py`. Load from
environment:

```python
from ai_cli.config.rag_config import RAGConfig

cfg = RAGConfig.from_env()
cfg.ensure_dirs()
```

## Best Practices

- Tune chunk size to match your embedding model (500–1000 chars is a good start)
- Use overlap (50–200 chars) to preserve context across chunk boundaries
- Persist FAISS indexes with `VectorStore.save()` / `load()`
- Store source metadata on chunks for citation and filtering
- Use the `echo` provider for local testing without API keys

## Debugging

```bash
ai-cli --debug -q "test prompt"
```

## Security

- Do not commit `.env` files with API keys
- Be mindful of PII in documents indexed for RAG
- Use local embedding models when data cannot leave your network
