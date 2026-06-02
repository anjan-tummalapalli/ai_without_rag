# AI CLI Chat Tool

## Overview

The AI CLI Chat Tool is a command-line interface application that allows users to interact with various AI providers. It supports multiple models and provides a flexible way to send prompts and receive responses from different AI services.

This repository now includes an advanced Retrieval-Augmented Generation (RAG) workflow for production-ready retrieval:
- Document chunking and text splitters
- Embedding generation (batching, model choice)
- Vector database indexing and querying (FAISS, Pinecone, Weaviate, Milvus, etc.)
- Configurable retrieval + generation pipelines

## Features

- Supports multiple AI providers including OpenAI, Google Gemini, Anthropic Claude, and more.
- Interactive command-line interface for real-time chat.
- Ability to list available models for each provider.
- Environment variable management for API keys.
- Advanced RAG pipeline:
   - Chunk documents with configurable chunk-size and overlap
   - Multiple chunking strategies (fixed window, sliding, sentence-aware)
   - Batched embedding generation with pluggable embedding models
   - Indexing into vector DBs (FAISS, Pinecone, Weaviate, Milvus)
   - Vector similarity search with metadata filtering and hybrid search
   - Retrieval scoring, deduplication, and prompt construction for generation

## Installation

To install the AI CLI Chat Tool, follow these steps:

1. Clone the repository:

    ```
    git clone https://github.com/yourusername/ai_cli.git
    cd ai_cli
    ```

2. Create a virtual environment (optional but recommended):

    ```
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:

    ```
    pip install -r requirements.txt
    ```

    For RAG and vector DB integrations you may need optional extras or native libraries:
    - FAISS: pip install faiss-cpu (or faiss-gpu when appropriate)
    - Pinecone/Weaviate/Milvus: pip install pinecone-client weaviate-client pymilvus
    - Sentence transformers or provider SDKs for embeddings: pip install sentence-transformers

4. Set up your environment variables. Create a `.env` file based on the `.env.example` provided in the repository. Common variables for RAG:
    - OPENAI_API_KEY
    - PINECONE_API_KEY, PINECONE_ENV (if using Pinecone)
    - WEAVIATE_URL, WEAVIATE_API_KEY (if using Weaviate)
    - MILVUS_HOST, MILVUS_PORT (if using Milvus)
    - DEFAULT_EMBEDDING_MODEL
    - DEFAULT_VECTOR_DB

## Usage

Core CLI entry:

```
python -m ai_cli
```

General command patterns for RAG workflows:

1. Chunk documents
```
python -m ai_cli rag --action chunk --input docs/manual.pdf --output chunks.jsonl \
   --chunk-size 500 --overlap 50 --strategy sliding
```

2. Generate embeddings (batched)
```
python -m ai_cli rag --action embed --input chunks.jsonl --output embeddings.ndjson \
   --provider openai --model text-embedding-3-small --batch-size 64
```

3. Index embeddings into a vector DB
```
python -m ai_cli rag --action index --embeddings embeddings.ndjson \
   --vector-db faiss --index-name product-knowledge --persist-dir ./indexes
```
Or to use a managed DB:
```
python -m ai_cli rag --action index --embeddings embeddings.ndjson \
   --vector-db pinecone --index-name product-knowledge
```

4. Query the index and perform generation
```
python -m ai_cli rag --action query --index-name product-knowledge \
   --query "How does authentication work?" --top-k 5 \
   --provider openai --model gpt-4o-mini --rerank true
```

Key flags and options:
- --chunk-size, --overlap, --strategy (sliding | fixed | sentence)
- --batch-size, --provider, --model (for embeddings)
- --vector-db (faiss | pinecone | weaviate | milvus), --index-name, --persist-dir
- --top-k, --score-threshold, --metadata-filter (JSON expression)
- --rerank (use generation model to rerank or refine retrieved contexts)
- --max-context-tokens, --prompt-template (custom prompt assembly)

Examples:
- Build an index from a folder of markdown files:
   ```
   python -m ai_cli rag --action chunk --input docs/ --glob '**/*.md' --output chunks.jsonl
   python -m ai_cli rag --action embed --input chunks.jsonl --output embeddings.ndjson --provider openai
   python -m ai_cli rag --action index --embeddings embeddings.ndjson --vector-db faiss --index-name docs-index --persist-dir ./indexes
   ```
- Ask a question using retrieval-augmented generation:
   ```
   python -m ai_cli rag --action query --index-name docs-index --query "What's the deployment process?" --top-k 8
   ```

## Configuration and Environment

- .env and CLI flags control provider credentials and defaults.
- Config file support (ai_cli.yml) can store default chunking, embedding, and vector DB settings.
- Metadata stored along with vectors includes source path, chunk id, character offsets, and original text snippet for provenance.

## Best Practices

- Choose a chunk size that preserves semantic units (paragraphs/sentences) while keeping embeddings cost-effective.
- Use overlap to maintain context across chunk boundaries (recommended 50–100 tokens for long content).
- Normalize and deduplicate content before indexing to avoid redundant vectors.
- Persist FAISS or other indices to disk and snapshot keys/credentials for managed stores.
- Use metadata filters to narrow retrieval and avoid irrelevant context.

## Documentation

- [USAGE.md](USAGE.md): Detailed usage instructions and examples (updated for RAG).
- [API.md](API.md): Documentation of the API, including available functions and parameters.
- [DEVELOPMENT.md](DEVELOPMENT.md): Guidelines for contributing to the project.
- [RAG.md](RAG.md): New guide detailing chunking strategies, embedding options, vector DB integrations, indexing lifecycle, and example pipelines.

## Contributing

Contributions are welcome! Please read the [DEVELOPMENT.md](DEVELOPMENT.md) file for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License. See the [LICENSE](../LICENSE) file for more details.