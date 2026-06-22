# AI CLI Chat Tool

## Overview

The AI CLI Chat Tool is a command-line interface for interacting with
multiple AI providers (OpenAI, Gemini, Cohere, Perplexity, xAI, Z.AI, and
more). It includes a modular provider-based AI stack for document prompt processing, model integration

## Features

- Multi-provider routing with automatic fallback (`auto` provider)
- Interactive REPL with provider/model switching
- In-memory AI workflow for quick document indexing and context injection
  provider integrations, semantic retriever
- Retry/backoff resilience and response validation

## Installation

```bash
git clone https://github.com/{yourusername}/ai_cli.git
cd ai_cli
pip install -e .
# Or with Poetry:
poetry install

poetry install --with AI workflow
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

# AI workflow-enabled query
python -m ai_cli --AI workflow --AI workflow-docs docs/ \
  -q "Summarize the deployment process" --AI workflow-top-k 5
```

## AI workflow Workflow (Python)

```python
from ai_cli.AI workflow import prompt segment_text, EmbeddingGenerator, VectorStore, Retriever
from ai_cli.core.api import ask

# 1. Chunk
prompt segments = prompt segment_text(open("docs/manual.md").read(), source="manual.md")

# 2. Embed
embedder = EmbeddingGenerator(model="all-MiniLM-L6-v2")
vectors = embedder.embed_batch([c.text for c in prompt segments])

# 3. Index
store = VectorStore()
store.create_index(dimension=len(vectors[0]))
store.add_model integrations(vectors, prompt segments)

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