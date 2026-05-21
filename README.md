# AI CLI Gateway

Enterprise-grade multi-provider AI CLI gateway for intelligent AI operations, observability, routing, reliability engineering, and infrastructure automation.

---

# Overview

AI CLI Gateway is a production-oriented command-line platform designed to interact with modern Large Language Model (LLM) providers through a unified interface.

The project is built for:

- DevOps engineers
- Site Reliability Engineers (SREs)
- Platform engineers
- AI infrastructure teams
- Enterprise automation platforms
- AI operations (AIOps)

It supports:
- multi-provider orchestration,
- intelligent routing,
- observability,
- reliability engineering,
- failover,
- hallucination detection,
- token/cost tracking,
- enterprise monitoring.

---

# Supported Providers

| Provider | Status |
|---|---|
| OpenAI | Supported |
| Anthropic Claude | Supported |
| Google Gemini | Supported |
| xAI Grok | Supported |
| Cohere | Supported |
| DeepSeek | Supported |
| Groq | Supported |
| Mistral AI | Supported |
| Together AI | Supported |
| GitHub Models | Planned |
| OpenRouter | Planned |

---

# Enterprise Features

## AI Gateway Capabilities

- Multi-provider LLM orchestration
- Unified AI abstraction layer
- Provider failover
- Cost-aware routing
- Latency-aware routing
- Intelligent retry handling
- Multi-model execution

---

## Reliability Engineering

- Retry intelligence
- Exponential backoff
- Hallucination scoring
- Response validation
- Health-aware provider selection
- Timeout management
- Failure correlation

---

## Observability

- OpenTelemetry tracing
- Structured logging
- Prometheus metrics
- Token tracking
- Request latency monitoring
- Error correlation
- Trace IDs

---

## Security

- Prompt sanitization
- Environment variable isolation
- Secure API key handling
- Input validation
- Response validation

---

# Architecture

```text
                    ┌─────────────────────┐
                    │    CLI / Scripts    │
                    └─────────┬───────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │     AI Gateway Core     │
                 └─────────┬───────────────┘
                           │
       ┌───────────────────┼───────────────────┐
       ▼                   ▼                   ▼

┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ Routing      │   │ Reliability  │   │ Observability│
│ Engine       │   │ Engine       │   │ Layer        │
└──────┬───────┘   └──────┬───────┘   └──────┬───────┘
       │                  │                  │
       ▼                  ▼                  ▼

 Cost Routing      Retry Logic         Metrics
 Latency Routing   Validation          Tracing
 Failover          Hallucination       Logging
```

---

# Installation

## Requirements

- Python 3.10+
- Poetry recommended

---

## Clone Repository

```bash
git clone https://github.com/yourusername/ai-cli.git

cd ai-cli
```

---

## Install Dependencies

### Using Poetry (Recommended)

```bash
poetry install
```

### Using pip

```bash
pip install .
```

---

# Environment Variables

Configure provider credentials using environment variables.

## Required Variables

| Provider | Environment Variable |
|---|---|
| OpenAI | `OPENAI_API_KEY` |
| Gemini | `GEMINI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| xAI Grok | `GROK_API_KEY` |
| Cohere | `COHERE_API_KEY` |
| Groq | `GROQ_API_KEY` |
| DeepSeek | `DEEPSEEK_API_KEY` |
| Together AI | `TOGETHER_API_KEY` |
| Mistral AI | `MISTRAL_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |

---

## Example `.env`

```bash
OPENAI_API_KEY=xxxxx
ANTHROPIC_API_KEY=xxxxx
GEMINI_API_KEY=xxxxx
GROK_API_KEY=xxxxx
COHERE_API_KEY=xxxxx
```

---

# Usage

# Single Prompt

```bash
ai-cli \
  --provider openai \
  --prompt "Explain Kubernetes operators"
```

---

# Model Override

```bash
ai-cli \
  --provider anthropic \
  --model claude-sonnet-4 \
  --prompt "Explain Terraform state locking"
```

---

# Pipe Support

```bash
echo "Explain SRE principles" | ai-cli -p grok
```

---

# Debug Mode

```bash
ai-cli \
  --provider openai \
  --prompt "Explain Prometheus" \
  --debug
```

---

# Interactive Mode

```bash
ai-cli --interactive --provider gemini
```

---

# List Available Models

```bash
ai-cli --list-models
```

---

# Example Output

```text
Kubernetes Operators extend Kubernetes functionality
using custom resources and controllers...
```

---

# Observability

# Metrics

The platform exposes enterprise metrics including:

```text
ai_provider_requests_total
ai_provider_errors_total
ai_provider_tokens_total
ai_provider_request_latency_seconds
```

---

# OpenTelemetry

Supports:
- distributed tracing,
- request spans,
- provider correlation,
- latency tracing.

---

# Reliability Features

# Hallucination Detection

Detects:
- suspicious responses,
- invalid structures,
- placeholder outputs,
- low-confidence content.

---

# Retry Intelligence

Supports:
- exponential backoff,
- transient failure retries,
- provider recovery logic,
- timeout handling.

---

# Intelligent Routing

Supports:
- lowest-cost provider selection,
- lowest-latency provider selection,
- provider failover,
- health-aware routing.

---

# Development

# Run Tests

```bash
pytest
```

---

# Run With Coverage

```bash
pytest --cov=src/ai_cli
```

---

# Code Formatting

```bash
black .
```

---

# Linting

```bash
ruff check .
```

---

# Type Checking

```bash
mypy src/
```

---

# Recommended Project Structure

```text
ai-cli/
├── src/
│   └── ai_cli/
│       ├── providers/
│       ├── routing/
│       ├── observability/
│       ├── reliability/
│       ├── governance/
│       ├── metrics/
│       ├── cli.py
│       └── ai_chat.py
│
├── tests/
├── docs/
├── examples/
├── scripts/
├── Dockerfile
├── pyproject.toml
└── README.md
```

---

# Enterprise Roadmap

## Planned Features

- Async provider execution
- Redis caching
- Kubernetes operator
- AI workload autoscaling
- RBAC integration
- Policy enforcement engine
- RAG orchestration
- Multi-region failover
- AI governance framework
- Intelligent workload scheduling

---

# CI/CD Integration

Example:

```bash
ai-cli \
  --provider openai \
  --prompt "Validate Terraform plan"
```

Ideal for:
- Jenkins
- GitHub Actions
- GitLab CI
- Argo Workflows
- Kubernetes Jobs

---

# Contributing

Contributions are welcome.

Please review:

```text
docs/DEVELOPMENT.md
```

before submitting pull requests.

---

# License

Licensed under the MIT License.

See:

```text
LICENSE
```

for details.

---

# Vision

AI CLI Gateway aims to evolve into a complete:

- Enterprise AI Gateway
- AI Infrastructure Platform
- Intelligent Operations Control Plane
- AI Reliability Engineering Framework
- Multi-Provider AI Orchestration Layer