# Usage Instructions for AI CLI

## Overview

The AI CLI is a command-line tool that supports both simple prompts and an advanced Retrieval-Augmented Generation (RAG) pipeline: chunking, embedding, building/querying vector databases, and RAG-style query & interactive sessions. Use it to index local documents, build vector indexes, and answer queries with retrieved context.

## Installation

Clone the repository and install required dependencies. For RAG workflows you will likely need additional packages (embedding models, vector DB clients):

```bash
git clone <repository-url>
cd ai_cli
pip install -r requirements.txt

# Optional vector/embedding backends (examples)
pip install sentence-transformers faiss-cpu chromadb
```

Also check `.env.example` for required environment variables (see below).

## Environment Variables

Set keys for providers and DB locations:

- OPENAI_API_KEY — OpenAI API key (embeddings & LLM)
- COHERE_API_KEY — Cohere key (optional)
- GOOGLE_API_KEY / GOOGLE_APPLICATION_CREDENTIALS — for Google-based providers
- CHROMA_DB_DIR — local Chroma DB directory (if using Chroma)
- FAISS_INDEX_DIR — path to store FAISS indexes

Export in shell or use a .env file.

## Concepts

- Chunking: split large documents into smaller passages. Configurable chunk size and overlap.
- Embeddings: convert chunks to vector embeddings; choose provider and model.
- Vector DB / Index: store vectors for fast nearest-neighbor search (FAISS, Chroma, Milvus, etc.).
- Retrieval: nearest-neighbor search returns top-k chunks to provide context.
- RAG: combine retrieved context with an LLM prompt to generate informed answers.

## CLI Usage

Base invocation:

```bash
python -m ai_cli <command> [options]
python -m ai_cli --help
```

### Single-shot Prompt

```bash
python -m ai_cli --provider openai --prompt "What is the capital of France?"
```

Example output:

```
The capital of France is Paris.
```

### Interactive Chat

```bash
python -m ai_cli --interactive --provider openai
```

### Advanced RAG Commands

1. Chunk documents

Split a file (or directory) into chunks.

```bash
python -m ai_cli chunk \
   --input docs/article.md \
   --output chunks.jsonl \
   --chunk-size 800 \
   --overlap 200 \
   --method sentence
```

Options:
- --chunk-size: target tokens/characters per chunk
- --overlap: overlap between chunks
- --method: sentence | paragraph | fixed

You can also pipe content:

```bash
cat big_doc.txt | python -m ai_cli chunk --output - --chunk-size 1000 --overlap 200 > chunks.jsonl
```

2. Create embeddings

Create embeddings for chunks using a chosen provider/model.

```bash
python -m ai_cli embed \
   --chunks chunks.jsonl \
   --emb-provider openai \
   --emb-model text-embedding-3-small \
   --output embeddings.parquet
```

Options:
- --emb-provider: openai | sentence-transformers | cohere | ...
- --emb-model: model name supported by the provider

3. Build / index vectors

Create a vector index (FAISS/Chroma/etc.) from embeddings or chunks.

```bash
python -m ai_cli index \
   --emb-file embeddings.parquet \
   --vector-db faiss \
   --index-path ./indexes/articles.faiss \
   --metadata-file chunks.jsonl
```

Or build directly from chunks (embeddings generated as part of indexing):

```bash
python -m ai_cli index \
   --chunks chunks.jsonl \
   --vector-db chroma \
   --emb-provider openai \
   --emb-model text-embedding-3-small \
   --chroma-dir ./chroma_db
```

Supported DBs:
- faiss, chroma, milvus (pluggable backends)

4. Query an index (RAG)

Retrieve relevant chunks then call an LLM with context:

```bash
python -m ai_cli query \
   --index ./indexes/articles.faiss \
   --query "Explain the main result of the paper" \
   --k 5 \
   --provider openai \
   --model gpt-4o \
   --retrieval-mode hybrid \
   --rerank
```

Options:
- --k: number of nearest neighbors
- --retrieval-mode: similarity | hybrid (similarity + sparse)
- --rerank: use an LLM to re-rank retrieved passages
- --score-threshold: minimum similarity to accept

Example expected output (truncated):

```
Context (top 3):
1) [chunk-id:...] ...summary of paragraph...
2) [chunk-id:...] ...another relevant paragraph...
Answer:
The main result is that ...
```

5. Interactive RAG session

Start a chat session that uses the index for retrieval on every turn.

```bash
python -m ai_cli --rag --index ./indexes/articles.faiss \
   --emb-provider openai --emb-model text-embedding-3-small \
   --provider openai --model gpt-4o --interactive
```

6. Utilities

- List models

```bash
python -m ai_cli --provider openai --list-models
```

- Export / Inspect index

```bash
python -m ai_cli index --inspect --index ./indexes/articles.faiss
```

- Delete / rebuild index

```bash
python -m ai_cli index --index-path ./indexes/articles.faiss --rebuild --chunks chunks.jsonl --emb-provider openai
```

## Typical Workflows

1. One-off QA over a document
- chunk -> embed -> index -> query

2. Periodic indexing pipeline
- Watch a directory for new files -> chunk -> embed -> upsert to vector DB

3. Streaming or interactive RAG
- Use streaming LLM responses while retrieving updated context for follow-ups

## Best Practices

- Tune chunk size and overlap to match your retrieval granularity.
- Normalize text before chunking (strip boilerplate, remove images).
- Cache embeddings to avoid repeated provider calls.
- Secure API keys and limit access to stored indexes if they contain sensitive data.
- Add metadata (source, filename, position) to chunks for traceability.

## Help and Debugging

For detailed per-command options and examples:

```bash
python -m ai_cli <command> --help
```

Enable verbose logging:

```bash
python -m ai_cli <command> --verbose
```

## Security & Privacy

- Be mindful of PII in documents you index. Redact or avoid uploading sensitive content to third-party providers.
- Use on-prem or closed-source embedding models if required by your data policies.

## Example Full Pipeline

```bash
# 1. Chunk
python -m ai_cli chunk --input docs/ --output chunks.jsonl --chunk-size 800 --overlap 200

# 2. Embed
python -m ai_cli embed --chunks chunks.jsonl --emb-provider openai --emb-model text-embedding-3-small --output embeddings.parquet

# 3. Index
python -m ai_cli index --emb-file embeddings.parquet --vector-db faiss --index-path ./indexes/articles.faiss --metadata-file chunks.jsonl

# 4. Query
python -m ai_cli query --index ./indexes/articles.faiss --query "Summarize the policy recommendations" --k 5 --provider openai --model gpt-4o
```

For further details consult the command help. Feedback and contributions welcome.