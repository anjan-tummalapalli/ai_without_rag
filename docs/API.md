# API Documentation for AI CLI

## Overview

The AI CLI is a command-line interface tool that allows users to interact with various AI providers. This document outlines the available functions, their parameters, and return values. New advanced RAG (Retrieval-Augmented Generation) features have been added: chunking, embedding, vector database (vector-DB) operations, and end-to-end RAG query helpers.

## Available Functions

### 1. ask(provider: str, prompt: str, model: Optional[str] = None) -> str
- Description: Dispatches a prompt to the specified AI provider and returns the AI's response.
- Parameters:
  - provider (str): The key of the AI provider to use (e.g., "openai", "gemini").
  - prompt (str): The user's question or instruction.
  - model (Optional[str]): The model identifier to use. If not specified, the provider's default model is used.
- Returns: The AI's response as a string. If an error occurs, it returns a string starting with "[ERROR]".

### 2. run_interactive(provider: str, model: Optional[str] = None) -> None
- Description: Starts an interactive REPL-style chat session with the specified AI provider.
- Parameters:
  - provider (str): The initial provider key (must be one of the keys in PROVIDERS).
  - model (Optional[str]): The initial model override. If not specified, the provider's default model is used.
- Returns: None. The function runs until the user exits.

### 3. print_banner() -> None
- Description: Prints the ASCII-art application banner to stdout, displaying the tool name and supported providers.
- Returns: None.

### 4. c(text: str, color: str) -> str
- Description: Wraps the provided text in ANSI color codes for terminal output.
- Parameters:
  - text (str): The string to colorize.
  - color (str): The color name (e.g., "cyan", "green").
- Returns: The original text wrapped in ANSI escape sequences.

### 5. PromptCorrector (Class & Instance)
- Description: An intelligent, heuristic-based prompt correction utility that pre-processes prompts to resolve typos, balance brackets/quotes, normalize whitespace, and strip control/NUL characters before sending them to LLM providers.
- Instance: prompt_corrector (pre-configured convenience instance).
- Methods:
  - correct(prompt: str) -> str: Sanitizes and corrects the input prompt, returning the clean string.
  - __init__(typo_map=None, collapse_spaces=True, fix_punctuation=True, balance_brackets=True, clean_control_chars=True): Customizes the enabled sanitization rules.

## Advanced RAG Features

The following functions and classes enable document ingestion, embedding, vector storage, similarity search, and RAG-based answer generation.

### chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]
- Description: Splits long text into overlapping chunks for embedding/indexing.
- Parameters:
  - text (str): The document text to chunk.
  - chunk_size (int): Maximum size in characters for each chunk.
  - overlap (int): Number of overlapping characters between consecutive chunks.
- Returns: A list of text chunks.

### embed_texts(provider: str, texts: List[str], model: Optional[str] = None) -> List[List[float]]
- Description: Produces embeddings for a list of texts using the specified provider.
- Parameters:
  - provider (str): Embedding provider key.
  - texts (List[str]): List of strings to embed.
  - model (Optional[str]): Embedding model identifier (defaults per provider).
- Returns: List of embedding vectors (float lists). On error returns an exception or raises based on configuration.

### VectorStore (Class)
- Description: Abstraction over a vector database for upsert/query/persist operations.
- Typical implementers: Pinecone, Weaviate, FAISS, Milvus, local disk-based indexes.
- Methods:
  - upsert(namespace: str, ids: List[str], vectors: List[List[float]], metadata: Optional[List[Dict]] = None) -> None
  - query(namespace: str, query_vector: List[float], top_k: int = 5, score_threshold: Optional[float] = None) -> List[Dict]
    - Returns a list of results: {"id": str, "score": float, "metadata": dict, "text": str}
  - delete(namespace: str, ids: List[str]) -> None
  - persist() -> None
  - load() -> None

### upsert_vectors(namespace: str, ids: List[str], vectors: List[List[float]], metadata: Optional[List[Dict]] = None) -> None
- Description: Convenience wrapper that upserts vectors into the configured VectorStore.
- Parameters:
  - namespace (str): Logical collection/namespace.
  - ids (List[str]): Unique ids for each vector.
  - vectors (List[List[float]]): Embedding vectors.
  - metadata (Optional[List[Dict]]): Optional per-item metadata (e.g., source, chunk_index).

### query_vectors(namespace: str, query_vector: List[float], top_k: int = 5, score_threshold: Optional[float] = None) -> List[Dict]
- Description: Convenience wrapper that performs a similarity search in the configured VectorStore.
- Returns: List of results as described in VectorStore.query.

### build_vector_index(texts: List[str], ids: Optional[List[str]] = None, namespace: str = "default", provider: str = "openai", model: Optional[str] = None, chunk_size: int = 1000, overlap: int = 200) -> List[str]
- Description: End-to-end ingestion helper: chunk the texts, create embeddings, and upsert them into the vector store.
- Parameters:
  - texts (List[str]): Documents to index.
  - ids (Optional[List[str]]): Optional base ids; if omitted, ids are autogenerated.
  - namespace (str): Vector store namespace.
  - provider/model: Embedding provider/model.
  - chunk_size/overlap: Chunking strategy.
- Returns: The list of upserted ids.

### retrieve_rag_answer(provider: str, query: str, namespace: str = "default", top_k: int = 5, model: Optional[str] = None, rerank: bool = True) -> str
- Description: Runs a full RAG retrieval pipeline:
  1. Corrects the query via PromptCorrector (optional).
  2. Embeds the query.
  3. Queries the VectorStore for top_k similar document chunks.
  4. Optionally reranks or filters results.
  5. Builds a context prompt that includes retrieved chunks and sends it to the generator model.
- Parameters:
  - provider (str): The provider used for generation (and optionally for embeddings).
  - query (str): The user's natural-language question.
  - namespace (str): Vector store namespace to search.
  - top_k (int): Number of retrieval results to include.
  - model (Optional[str]): Generation model to use.
  - rerank (bool): Whether to run an optional reranking step before generation.
- Returns: The generated answer string. If retrieval fails or no documents found, returns an informative "[ERROR]" or a fallback generation.

## Pipeline Notes and Best Practices

- Chunking: Use chunk_size tuned to the embedding model's context-to-embedding fidelity; typical values: 500–2000 chars with 100–300 overlap.
- Embeddings: Normalize or batch texts before requesting embeddings to improve throughput.
- Vector DB: Choose a backend that supports metadata filtering if you need scoped searches (e.g., by source or date).
- Security: Never persist sensitive PII to unencrypted vector stores without encryption policies.
- PromptCorrector: Apply before generating embeddings and before final generation to reduce injection and formatting issues.
- Error Handling: All functions return errors with a string that begins with "[ERROR]" or raise exceptions depending on the CLI configuration. VectorStore implementations should map backend errors to consistent error outputs.

## Example Usage

```python
# 1. Ingest documents
texts = ["Long doc text ...", "Another document ..."]
ids = build_vector_index(texts, namespace="knowledge_base", provider="openai")

# 2. Ask a question with RAG
answer = retrieve_rag_answer("openai", "How does feature X work?", namespace="knowledge_base", top_k=5)
print(answer)

# 3. Low-level operations
chunks = chunk_text(long_text, chunk_size=1000, overlap=200)
embs = embed_texts("openai", chunks)
upsert_vectors("kb", ids=["doc1#0", "doc1#1"], vectors=embs, metadata=[{"source":"doc1"}]*len(embs))
results = query_vectors("kb", query_vector=embed_texts("openai", ["How to use X?"])[0], top_k=3)
```

## Error Handling

All functions are designed to handle errors gracefully. If an error occurs during execution, the functions will return or surface a string that begins with "[ERROR]", providing information about the issue, or raise a documented exception depending on context and CLI flags.

## Conclusion

This API documentation now includes comprehensive RAG capabilities — chunking, embedding, vector-DB management, and a high-level retrieve_rag_answer helper for building retrieval-augmented generation workflows. For deployment and backend configuration details, see the `USAGE.md` and provider-specific configuration sections.