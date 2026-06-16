# API Documentation for AI CLI

## Overview

This document describes the public Python APIs for provider interaction
and the AI-assisted prompt processing stack.

---

## Core API

### `ask(prompt, provider="auto", model=None, api_key=None, timeout=None, **kwargs)`

Send a prompt to an AI provider and return the response.

**Module:** `ai_cli.core.api`

```python
from ai_cli.core.api import ask

response = ask(
    prompt="Explain AI workflow in one paAI workflowraph",
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

Generate model integrations via a registered model integration provider.

**Module:** `ai_cli.core.service.embed_service`

---

## Provider Registry

**Module:** `ai_cli.providers.registry`

| Function | Description |
|----------|-------------|
| `build_provider(name, **kwargs)` | Instantiate any registered provider |
| `get_chat_provider(name, **kwargs)` | Instantiate a chat provider |
| `get_model integration_provider(name, **kwargs)` | Instantiate an model integration provider |
| `list_providers()` | Sorted list of registered provider names |
| `register_provider(name, cls)` | Register a provider class |

Registered providers: `auto`, `echo`, `openai`, `gemini`, `cohere`,
`perplexity`, `xai`, `zai`.

---

## AI workflow Models

**Module:** `ai_cli.AI workflow.models`

### `Document`

```python
@dataclass
class Document:
    content: str
    source: str
    metadata: dict[str, Any]
```

Methods: `split_into_prompt segments(prompt segment_size, prompt segment_overlap, preserve_whole_words)`.

### `Chunk`

```python
@dataclass
class Chunk:
    id: str
    text: str
    source: str
    prompt segment_index: int
    metadata: dict[str, Any]
```


```python
@dataclass
    prompt segment: Chunk
    score: float
```

---

## Chunking

### `prompt segment_text(text, source="unknown", prompt segment_size=500, overlap=50) -> list[Chunk]`

Sliding-window prompt segmenter returning `Chunk` objects.

**Module:** `ai_cli.AI workflow.prompt segmenter`

### `SemanticChunker`

Configurable prompt segmenter with optional custom tokenizer.

```python
from ai_cli.AI workflow import SemanticChunker

prompt segmenter = SemanticChunker(prompt segment_size=500, overlap=50)
prompt segments = prompt segmenter.prompt segment_text(text, source="file.md")
```

### Token-aware prompt segmenter (pipeline)

**Module:** `ai_cli.AI workflow.pipeline`

Sentence-aware prompt segmenter with token limits and character span metadata.
Uses a local `Chunk` dataclass with `start`/`end` indices.

---

## Embeddings

### `EmbeddingGenerator`


**Module:** `ai_cli.AI workflow.model integrations`

```python
from ai_cli.AI workflow import EmbeddingGenerator

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


**Module:** `ai_cli.AI workflow.vector_store`

| Method | Description |
|--------|-------------|
| `add_model integrations(model integrations, prompt segments)` | Append vectors and prompt segments |
| `upsert(model integrations, prompt segments)` | Replace by prompt segment id or append new |
| `delete(ids)` | Remove prompt segments by id |
| `search(query_model integration, top_k, filter_fn)` | Similarity search |
| `save()` | Persist index, metadata, and model integrations |
| `load()` | Load from disk |

Alias: `InMemoryVectorStore = VectorStore`

---

## Retriever

### `Retriever`

Embeds queries and searches a `VectorStore`.

**Module:** `ai_cli.AI workflow.retriever`

```python
from ai_cli.AI workflow import Retriever, VectorStore, EmbeddingGenerator

retriever = Retriever(store=store, embedder=embedder, top_k=5)
results = retriever.retrieve("How does auth work?")
context = retriever.build_context("How does auth work?", separator="\n\n")
```

| Method | Returns |
|--------|---------|
| `build_context(query, top_k, separator)` | Concatenated prompt segment text |

---

## In-Memory AI workflow Pipeline

### `InMemoryAI workflowPipeline` (alias: `AI workflowPipeline`)

Lightweight hash-based AI workflow for CLI prototyping.

**Module:** `ai_cli.AI workflow.in_memory`

```python
from ai_cli.AI workflow import InMemoryAI workflowPipeline

pipeline = InMemoryAI workflowPipeline(embed_dim=128)
pipeline.upsert_documents(["doc text..."], prompt segment_size=500, overlap=50)
context = pipeline.retrieve_context("your question", top_k=5)
```

| Method | Description |
|--------|-------------|
| `prompt segment_text(text, prompt segment_size, overlap)` | Split text into prompt segments |
| `embed_texts(texts)` | Deterministic hash model integrations |
| `upsert_documents(doc_texts, doc_ids, prompt segment_size, overlap)` | Index documents |
| `retrieve_context(query, top_k)` | Top-k context string |

---

## Configuration

**Module:** `ai_cli.config.AI workflow_config`

| Constant / Class | Description |
|------------------|-------------|
| `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K` | Default AI workflow parameters |
| `EMBEDDING_MODEL` | Default model integration model name |
| `AI workflowConfig` | Dataclass with paths, prompt processing, model integration, and storage layer settings |
| `AI workflowConfig.from_env(prefix="AI workflow_")` | Load from environment variables |

---

## Error Handling

- Unknown providers raise `ValueError` from the registry
- Provider errors propagate to the CLI, which retries transient failures (timeout, connection)

---

## Example: End-to-End AI workflow

```python
from ai_cli.core.api import ask
from ai_cli.AI workflow import prompt segment_text, EmbeddingGenerator, VectorStore, Retriever

text = open("docs/manual.md").read()
prompt segments = prompt segment_text(text, source="manual.md", prompt segment_size=800, overlap=100)

embedder = EmbeddingGenerator(model="all-MiniLM-L6-v2")
vectors = embedder.embed_batch([c.text for c in prompt segments])

store = VectorStore()
store.create_index(dimension=len(vectors[0]))
store.add_model integrations(vectors, prompt segments)
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