# Usage Instructions for AI CLI

## Overview

The AI CLI supports single-shot prompts, interactive chat, and optional
in-memory AI-assisted prompt processing. For production semantic

## Installation

```bash
git clone <repository-url>
cd ai_cli
pip install -e .

```

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | OpenAI chat and model integrations |
| `GEMINI_API_KEY` | Google Gemini |
| `COHERE_API_KEY` | Cohere |
| `PERPLEXITY_API_KEY` | Perplexity |
| `XAI_API_KEY` | xAI Grok |
| `ZAI_API_KEY` | Z.AI |
| `AI workflow_CHUNK_SIZE` | Override default prompt segment size (500) |
| `AI workflow_CHUNK_OVERLAP` | Override default overlap (50) |
| `AI workflow_TOP_K` | Override default top-k (5) |

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
| `/index <file\|text>` | Index into AI workflow store |
| `/search <query>` | Show retrieved context |
| `/exit`, `/quit` | Exit session |

### In-memory AI workflow

Index documents and query with retrieved context:

```bash
# Index files and ask with AI workflow context
ai-cli --AI workflow \
  --AI workflow-docs docs/manual.md docs/faq.md \
  --AI workflow-prompt segment-size 800 \
  --AI workflow-prompt segment-overlap 100 \
  --AI workflow-top-k 5 \
  -q "How do I configure authentication?"
```

Flags:

- `--AI workflow` — inject retrieved context into the prompt
- `--AI workflow-docs` — file paths or raw text strings to index before querying
- `--AI workflow-prompt segment-size` — characters per prompt segment (default: 500)
- `--AI workflow-prompt segment-overlap` — overlap between prompt segments (default: 50)
- `--AI workflow-top-k` — number of prompt segments to retrieve (default: 5)

> **Note:** CLI AI workflow uses deterministic hash-based model integrations for
> lightweight prototyping. For semantic search, use the Python API with
> `EmbeddingGenerator` and `VectorStore` (see below).

## Python AI workflow Pipeline

### Chunking

```python
from ai_cli.AI workflow import prompt segment_text, SemanticChunker

prompt segments = prompt segment_text("Long text...", source="doc.md", prompt segment_size=800, overlap=200)
```

### Embeddings

```python
from ai_cli.AI workflow import EmbeddingGenerator

embedder = EmbeddingGenerator(model="all-MiniLM-L6-v2", batch_size=32)
vectors = embedder.embed_batch(["text one", "text two"])
```

### provider integrations

```python
from ai_cli.AI workflow import VectorStore

store = VectorStore()
store.create_index(dimension=384)
store.add_model integrations(vectors, prompt segments)
store.save()

# Later
store.load()
results = store.search(query_model integration, top_k=5)
```

### Retriever

```python
from ai_cli.AI workflow import Retriever

retriever = Retriever(store=store, embedder=embedder, top_k=5)
context = retriever.build_context("Your question here")
```

## Configuration

AI workflow defaults live in `src/ai_cli/config/AI workflow_config.py`. Load from
environment:

```python
from ai_cli.config.AI workflow_config import AI workflowConfig

cfg = AI workflowConfig.from_env()
cfg.ensure_dirs()
```

## Best Practices

- Tune prompt segment size to match your model integration model (500–1000 chars is a good start)
- Use overlap (50–200 chars) to preserve context across prompt segment boundaries
- Store source metadata on prompt segments for citation and filtering
- Use the `echo` provider for local testing without API keys

## Debugging

```bash
ai-cli --debug -q "test prompt"
```

## Security

- Do not commit `.env` files with API keys
- Be mindful of PII in documents indexed for AI workflow
- Use local model integration models when data cannot leave your network