# API Documentation for AI CLI

## Overview

This document describes the public Python APIs for provider interaction
and the RAG (Retrieval-Augmented Generation) stack.

---

## Core API

### `ask(prompt, provider="auto", model=None, api_key=None, timeout=None, **kwargs)`

Send a prompt to an AI provider and return the response.

**Module:** `ai_cli.core.api`

```python
from ai_cli.core.api import ask

response = ask(
    prompt="Explain RAG in one paragraph",
    provider="openai",
    model="gpt-4o-mini",
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `prompt` | `str` | User message |
| `provider` | `str` | Provider key (default: `"auto"`) |
| `model` | `str \| None` | Model override |
| `api_key` | `str \| None` | API key (falls back to `{PROVIDER}_API_KEY` env var) |
| `timeout` | `float \| None` | Request timeout in seconds |

**Returns:** Provider response (typically `str`; may be async iterable when streaming).

---

## Service Layer

### `ask_service.ask(prompt, provider="auto", ...)`

Lower-level wrapper using `get_chat_provider` and the provider's `chat()` method.

**Module:** `ai_cli.core.service.ask_service`

### `embed_service.embed(texts, provider, model=None, api_key=None)`

Generate embeddings via a registered embedding provider.

**Module:** `ai_cli.core.service.embed_service`

---

## Provider Registry

**Module:** `ai_cli.providers.registry`

| Function | Description |
|----------|-------------|
| `build_provider(name, **kwargs)` | Instantiate any registered provider |
| `get_chat_provider(name, **kwargs)` | Instantiate a chat provider |
| `get_embedding_provider(name, **kwargs)` | Instantiate an embedding provider |
| `list_providers()` | Sorted list of registered provider names |
| `register_provider(name, cls)` | Register a provider class |

Registered providers: `auto`, `echo`, `openai`, `gemini`, `cohere`,
`perplexity`, `xai`, `zai`.

---

## RAG Models

**Module:** `ai_cli.rag.models`

### `Document`

```python
@dataclass
class Document:
    content: str
    source: str
    metadata: dict[str, Any]
```

Methods: `split_into_chunks(chunk_size, chunk_overlap, preserve_whole_words)`.

### `Chunk`

```python
@dataclass
class Chunk:
    id: str
    text: str
    source: str
    chunk_index: int
    metadata: dict[str, Any]
```

### `RetrievalResult`

```python
@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
```

---

## Chunking

### `chunk_text(text, source="unknown", chunk_size=500, overlap=50) -> list[Chunk]`

Sliding-window chunker returning `Chunk` objects.

**Module:** `ai_cli.rag.chunker`

### `SemanticChunker`

Configurable chunker with optional custom tokenizer.

```python
from ai_cli.rag import SemanticChunker

chunker = SemanticChunker(chunk_size=500, overlap=50)
chunks = chunker.chunk_text(text, source="file.md")
```

### Token-aware chunker (pipeline)

**Module:** `ai_cli.rag.pipeline`

Sentence-aware chunker with token limits and character span metadata.
Uses a local `Chunk` dataclass with `start`/`end` indices.

---

## Embeddings

### `EmbeddingGenerator`

Wraps sentence-transformers with batching and L2 normalization.

**Module:** `ai_cli.rag.embeddings`

```python
from ai_cli.rag import EmbeddingGenerator

gen = EmbeddingGenerator(model="all-MiniLM-L6-v2", batch_size=32, normalize=True)
vector = gen.embed_text("hello world")
vectors = gen.embed_batch(["text one", "text two"])
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `all-MiniLM-L6-v2` | Model name or pre-instantiated model |
| `batch_size` | `32` | Encoding batch size |
| `normalize` | `True` | L2-normalize output vectors |
| `device` | `None` | Optional device (`cpu`, `cuda`, `mps`) |

Alias: `EmbeddingsProvider = EmbeddingGenerator`

---

## Vector Store

### `VectorStore`

FAISS-backed vector store with persist/load support.

**Module:** `ai_cli.rag.vector_store`

| Method | Description |
|--------|-------------|
| `create_index(dimension)` | Create a new FAISS flat L2 index |
| `add_embeddings(embeddings, chunks)` | Append vectors and chunks |
| `upsert(embeddings, chunks)` | Replace by chunk id or append new |
| `delete(ids)` | Remove chunks by id |
| `search(query_embedding, top_k, filter_fn)` | Similarity search |
| `save()` | Persist index, metadata, and embeddings |
| `load()` | Load from disk |

Alias: `InMemoryVectorStore = VectorStore`

---

## Retriever

### `Retriever`

Embeds queries and searches a `VectorStore`.

**Module:** `ai_cli.rag.retriever`

```python
from ai_cli.rag import Retriever, VectorStore, EmbeddingGenerator

retriever = Retriever(store=store, embedder=embedder, top_k=5)
results = retriever.retrieve("How does auth work?")
context = retriever.build_context("How does auth work?", separator="\n\n")
```

| Method | Returns |
|--------|---------|
| `retrieve(query, top_k, filter_fn)` | `list[RetrievalResult]` |
| `build_context(query, top_k, separator)` | Concatenated chunk text |

---

## In-Memory RAG Pipeline

### `InMemoryRAGPipeline` (alias: `RAGPipeline`)

Lightweight hash-based RAG for CLI prototyping.

**Module:** `ai_cli.rag.in_memory`

```python
from ai_cli.rag import InMemoryRAGPipeline

pipeline = InMemoryRAGPipeline(embed_dim=128)
pipeline.upsert_documents(["doc text..."], chunk_size=500, overlap=50)
context = pipeline.retrieve_context("your question", top_k=5)
```

| Method | Description |
|--------|-------------|
| `chunk_text(text, chunk_size, overlap)` | Split text into chunks |
| `embed_texts(texts)` | Deterministic hash embeddings |
| `upsert_documents(doc_texts, doc_ids, chunk_size, overlap)` | Index documents |
| `retrieve_context(query, top_k)` | Top-k context string |

---

## Configuration

**Module:** `ai_cli.config.rag_config`

| Constant / Class | Description |
|------------------|-------------|
| `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K` | Default RAG parameters |
| `EMBEDDING_MODEL` | Default embedding model name |
| `FAISS_INDEX_PATH`, `METADATA_PATH` | Persist paths |
| `RAGConfig` | Dataclass with paths, chunking, embedding, and vector store settings |
| `RAGConfig.from_env(prefix="RAG_")` | Load from environment variables |

---

## Error Handling

- Unknown providers raise `ValueError` from the registry
- Missing optional dependencies (FAISS, numpy) raise `RuntimeError` with install hints
- Provider errors propagate to the CLI, which retries transient failures (timeout, connection)

---

## Example: End-to-End RAG

```python
from ai_cli.core.api import ask
from ai_cli.rag import chunk_text, EmbeddingGenerator, VectorStore, Retriever

text = open("docs/manual.md").read()
chunks = chunk_text(text, source="manual.md", chunk_size=800, overlap=100)

embedder = EmbeddingGenerator(model="all-MiniLM-L6-v2")
vectors = embedder.embed_batch([c.text for c in chunks])

store = VectorStore()
store.create_index(dimension=len(vectors[0]))
store.add_embeddings(vectors, chunks)
store.save()

retriever = Retriever(store, embedder, top_k=5)
query = "How do we rotate secrets in production?"
context = retriever.build_context(query)

answer = ask(
    prompt=f"Use this context to answer.\n\nContext:\n{context}\n\nQuestion: {query}",
    provider="openai",
)
print(answer)
```
