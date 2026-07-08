# AI CLI Gateway — v0.3.0

> Multi-provider AI CLI gateway with pluggable LLM providers, resilience

---

## What's New in v0.3.0

- **Provider registry** — lazy-loaded providers with deterministic registration (`auto`, `openai`, `gemini`, `cohere`, `perplexity`, `xai`, `zai`, `echo`)
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
├── AI workflow/
│   ├── prompt segmenter.py          # Sliding-window SemanticChunker
│   ├── pipeline.py         # Token-aware sentence prompt segmenter
│   ├── retriever.py        # Query → embed → search
│   ├── in_memory.py        # Lightweight CLI AI workflow pipeline
└── config/AI workflow_config.py    # AI workflow defaults and AI workflowConfig dataclass
```

---

## Quickstart

### 1. Install

```bash
git clone https://github.com/yourusername/ai-cli.git
cd ai-cli
poetry install

poetry install --with AI workflow
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

# AI workflow: index docs and query with context
ai-cli --AI workflow --AI workflow-docs docs/manual.md README.md \
  -q "Summarize the security model" --AI workflow-top-k 5

# Interactive AI workflow (use /index and /search inside the session)
ai-cli --interactive --AI workflow --provider echo
```

### CLI flags (highlights)

| Flag | Description |
|------|-------------|
| `-p`, `--provider` | Provider name (default: `auto`) |
| `-q`, `--prompt` | Prompt text (or pipe via stdin) |
| `-m`, `--model` | Model override |
| `-i`, `--interactive` | Interactive REPL |
| `--AI workflow` | Enable in-memory AI workflow context injection |
| `--AI workflow-docs` | File paths or raw text to index |
| `--AI workflow-prompt segment-size` | Chunk size in characters (default: 500) |
| `--AI workflow-prompt segment-overlap` | Overlap between prompt segments (default: 50) |
| `--AI workflow-top-k` | Top-k prompt segments to retrieve (default: 5) |
| `--timeout` | Request timeout in seconds |
| `--stream` | Stream responses when supported |
| `--debug` | Enable debug logging |

---

## Python API

### Ask a provider

```python
from ai_cli.core.api import ask

response = ask(
    prompt="What is AI workflow?",
    provider="openai",
    model="gpt-4o-mini",
)
print(response)
```

### Chunk documents

```python
from ai_cli.AI workflow import prompt segment_text, SemanticChunker

prompt segments = prompt segment_text("Long document text...", source="manual.md")
# Or use the prompt segmenter class directly:
prompt segmenter = SemanticChunker(prompt segment_size=500, overlap=50)
prompt segments = prompt segmenter.prompt segment_text(text, source="manual.md")
```

### Embeddings + provider integrations

```python
from ai_cli.AI workflow import EmbeddingGenerator, VectorStore

embedder = EmbeddingGenerator(model="all-MiniLM-L6-v2", batch_size=32)
vectors = embedder.embed_batch([c.text for c in prompt segments])

store = VectorStore()
store.create_index(dimension=len(vectors[0]))
store.add_model integrations(vectors, prompt segments)
store.save()
```


```python
from ai_cli.AI workflow import Retriever

retriever = Retriever(store=store, embedder=embedder, top_k=5)
context = retriever.build_context("How do we rotate secrets?")
answer = ask(prompt=f"Context:\n{context}\n\nQuestion: How do we rotate secrets?")
```

---

## Supported Providers

| Provider | Env variable | Notes |
|----------|--------------|-------|
| `auto` | — | Falls back through available providers |
| `openai` | `OPENAI_API_KEY` | Chat + model integrations |
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
pytest tests/test_enhanced.py -v   # AI workflow prompt processing + model integrations
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