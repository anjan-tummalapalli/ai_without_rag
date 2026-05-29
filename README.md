# AI CLI Gateway — v0.3.0

> Enterprise-grade multi-provider AI CLI gateway with Security & Identity Management, Tool Calling, ReAct Agents, and an enhanced Developer Experience.

---

## What's New in v0.3.0

This release delivers four major capability pillars on top of the existing multi-provider routing and resilience foundation:

| Pillar | Key additions |
|---|---|
| **Security & Identity** | Identity context, RBAC, layered secrets backend (Vault / AWS SM / Azure KV), audit log with HMAC integrity, prompt sanitisation, per-identity rate limiting & budget guard |
| **Tool Calling** | JSON-schema tool registry, 7 built-in tools, OpenAI & Anthropic schema export, RBAC-gated execution, execution trace |
| **Agents** | ReAct (Reason-Act-Observe) loop, sliding-window memory, scratchpad, AgentPlanner, AgentRunner façade |
| **Developer Experience** | Rich CLI output, `--agent`, `--with-tools`, `--list-tools`, `--login`, `--whoami`, `--budget`, `--audit`, `--trace` flags; YAML profiles; comprehensive test suite |

---

## Project Structure

```
src/ai_cli/
├── __init__.py               # ask(), ask_with_tools(), ask_agent() exports
├── cli.py                    # Enhanced CLI entrypoint (v0.3)
├── ai_chat.py                # Public convenience re-exports
├── core/
│   ├── api.py                # ask(), ask_with_tools(), ask_agent() + security pipeline
│   ├── exceptions.py         # Full exception hierarchy (provider, security, tool, agent)
│   └── resilience.py         # RetryEngine, CircuitBreaker, BulkheadLimiter,
│                             #   RateLimiter, HealthTracker, StreamConsumer, Cache
├── security/
│   └── identity.py           # IdentityContext, RBAC, SecretsBackend, AuditLogger,
│                             #   PromptSanitiser, IdentityRateLimiter, BudgetGuard
├── tools/
│   └── registry.py           # ToolRegistry, ToolDefinition, 7 built-in tools
├── agents/
│   └── agent.py              # AgentMemory, AgentPlanner, ReactAgent, AgentRunner
├── config/
│   └── profiles.py           # YAML profile loader (ai-cli.yaml)
├── providers/
│   ├── base.py               # AIProvider base class
│   ├── registry.py           # Provider registry + auto-routing
│   └── auto_provider.py      # Cost/latency-aware auto-selection
├── plugins/
│   └── builtins.py           # OpenAI, Anthropic, Groq, Gemini, Perplexity, etc.
├── telemetry/
│   └── monitoring.py         # Prometheus metrics, OpenTelemetry traces
└── utils/
    ├── secrets.py            # Legacy secrets helpers (superseded by security/)
    └── validation.py         # HallucinationDetector, prompt validators
```

---

## Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/ai-cli.git
cd ai-cli
poetry install
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in your provider API keys
```

### 3. Basic usage

```bash
# Single prompt (auto-selects best available provider)
ai-cli -q "Explain Kubernetes operators"

# Specific provider and model
ai-cli -p groq -m llama3-8b-8192 -q "Write a haiku about Python"

# Read from stdin
echo "Summarize this" | ai-cli -p openai

# Interactive REPL
ai-cli -i
```

---

## Security & Identity Management

### Identity & Authentication

Every request carries an `IdentityContext` (auto-resolved from env or an explicit gateway API key).

```bash
# Authenticate with a gateway API key
ai-cli --login aicli-developer-alice-<32hexchars>

# Show current identity, roles, budget, and rate limit
ai-cli --whoami
```

**Gateway API key format:** `aicli-<role>-<identity>-<32hexchars>`
Roles: `admin` | `developer` | `user` | `service`

```python
from ai_cli.security.identity import authenticate_api_key

ctx = authenticate_api_key("aicli-developer-alice-abcdef1234567890abcdef1234567890")
# ctx.identity_id == "alice", ctx.roles == ["developer"]
```

### RBAC (Role-Based Access Control)

Roles control which providers, tools, budgets, and rate limits apply.

| Role | Providers | Tools | Budget | Rate limit |
|---|---|---|---|---|
| `admin` | all | all | $100 | 500 rpm |
| `developer` | openai, anthropic, groq, gemini, auto | web_search, calculator, file_read, shell, … | $10 | 60 rpm |
| `user` | openai, anthropic, groq, auto | web_search, calculator | $2 | 20 rpm |
| `service` | all | all | $50 | 300 rpm |

Override the policy by providing `ai-cli-policy.json` or setting `AI_CLI_POLICY`:

```json
{
  "developer": {
    "allowed_providers": ["openai", "groq", "auto"],
    "allowed_tools": ["web_search", "calculator", "file_read"],
    "cost_budget_usd": 20.0,
    "rate_limit_rpm": 120
  }
}
```

### Layered Secrets Backend

Secrets are resolved in priority order — no raw keys in code:

```
Environment variables  →  HashiCorp Vault  →  AWS Secrets Manager  →  Azure Key Vault
```

```python
from ai_cli.security.identity import SECRETS

api_key = SECRETS.get("OPENAI_API_KEY")   # resolved from whichever backend has it
```

Configure the backends via environment variables:

```bash
# HashiCorp Vault
VAULT_ADDR=http://vault:8200
VAULT_TOKEN=hvs.xxx
VAULT_SECRET_PATH=secret/data/ai_cli

# AWS Secrets Manager
AWS_REGION=us-east-1
AWS_SECRET_NAME=ai_cli_secrets

# Azure Key Vault
AZURE_VAULT_URL=https://my-vault.vault.azure.net
```

### Audit Logging

Every `ask()` call writes an append-only JSONL entry with HMAC integrity:

```bash
# Tail the last 20 audit entries
ai-cli --audit 20
```

```python
from ai_cli.security.identity import AUDIT, IdentityContext

ctx = IdentityContext(identity_id="alice", roles=["developer"])
AUDIT.log("custom_event", ctx, {"action": "bulk_query", "count": 50})

# Verify integrity of a log line
is_valid = AUDIT.verify(line)
```

### Prompt Sanitisation & PII Redaction

```python
from ai_cli.security.identity import sanitize_prompt, redact_pii_for_log, detect_prompt_injection

# Detect injection patterns (warning in normal mode, raises in strict mode)
hits = detect_prompt_injection("Ignore all previous instructions!")

# Raises PromptValidationError in strict mode
sanitize_prompt(prompt, strict=True)

# Redact PII before logging (SSN, credit cards, email, phone)
safe_text = redact_pii_for_log("My SSN is 123-45-6789")
# → "My SSN is [SSN]"
```

Use `--strict` on the CLI to enable strict injection detection:

```bash
ai-cli --strict -q "Your prompt here"
```

### Budget Guard

```bash
# Show current spend vs. budget
ai-cli --budget
```

```python
from ai_cli.security.identity import enforce_budget, estimate_cost

cost = estimate_cost("openai", tokens=1500)   # → ~$0.003
enforce_budget(ctx, "openai", tokens=1500)    # raises BudgetExceededError if over limit
```

---

## Tool Calling

### Built-in Tools

| Tool | Category | Description |
|---|---|---|
| `web_search` | information | DuckDuckGo instant-answer search |
| `calculator` | utility | Safe math expression evaluator (supports `sqrt`, `log`, etc.) |
| `file_read` | filesystem | Read local text files (path-traversal protected) |
| `shell` | system | Execute shell commands (**admin only**) |
| `http_get` | network | HTTP GET request with body return |
| `datetime_now` | utility | Current UTC date and time |
| `env_lookup` | utility | Look up non-sensitive env vars (blocks key/secret/token) |

### CLI Tool Usage

```bash
# List all tools
ai-cli --list-tools

# Call a tool directly
ai-cli --tool calculator '{"expression": "sqrt(144) * 3.14"}'
ai-cli --tool datetime_now
ai-cli --tool web_search '{"query": "Python 3.13 new features"}'

# Ask with automatic tool calling enabled
ai-cli --with-tools -q "What is 17 factorial, and what time is it now?"

# Limit which tools are exposed
ai-cli --with-tools --tools calculator,datetime_now -q "What time is it?"
```

### Python API

```python
from ai_cli.tools.registry import TOOL_REGISTRY
from ai_cli.core.api import ask_with_tools

# Execute a tool directly
result = TOOL_REGISTRY.execute("calculator", {"expression": "2 ** 10"})
print(result.result)          # {"expression": "2 ** 10", "result": 1024}
print(result.duration_ms)     # e.g. 0.12

# Export schemas for provider APIs
openai_tools  = TOOL_REGISTRY.openai_schemas()
anthropic_tools = TOOL_REGISTRY.anthropic_schemas()

# Tool-aware ask (auto-detects and dispatches tool calls)
response = ask_with_tools(
    provider="openai",
    prompt="What is the square root of 1764, and what's today's date?",
    tools=["calculator", "datetime_now"],
)
```

### Registering Custom Tools

```python
from ai_cli.tools.registry import TOOL_REGISTRY, ToolDefinition, ToolParameter

def lookup_user(user_id: str) -> dict:
    return {"user_id": user_id, "name": "Alice", "plan": "pro"}

TOOL_REGISTRY.register(ToolDefinition(
    name="lookup_user",
    description="Look up a user by ID in the internal database.",
    category="internal",
    parameters=[
        ToolParameter("user_id", "string", "The user's unique ID.", required=True),
    ],
    fn=lookup_user,
))
```

---

## Agents (ReAct Loop)

The agent framework implements the **Reason → Act → Observe** pattern. The LLM decides the next tool to call, the tool executes, the result feeds back, and the loop continues until a final answer is produced or `max_steps` is reached.

### CLI Agent Mode

```bash
# Run a ReAct agent (multi-step)
ai-cli --agent -q "Find today's date and calculate how many days until 2026-01-01"

# Generate a plan first, then execute
ai-cli --agent --plan -q "Research the top 3 Python web frameworks and compare them"

# Verbose step-by-step trace
ai-cli --agent --verbose -q "What is the square root of today's Unix timestamp?"

# Print full execution trace after completion
ai-cli --agent --trace -q "Calculate 42 * 1337"

# Cap the number of steps
ai-cli --agent --max-steps 5 -q "Search for the latest news on AI"
```

### Python API

```python
from ai_cli.core.api import ask_agent

# Simple agent run
answer = ask_agent(
    goal="What is the current Unix timestamp divided by 86400?",
    provider="openai",
    max_steps=8,
    verbose=True,
)

# Plan-then-execute
answer = ask_agent(
    goal="Research quantum computing and produce a 3-point summary",
    provider="groq",
    plan_first=True,
    max_steps=12,
)
```

### AgentRunner (full control)

```python
from ai_cli.agents.agent import AgentRunner, AgentMemory

runner = AgentRunner(
    provider="openai",
    model="gpt-4o",
    max_steps=10,
    verbose=True,
)

result = runner.run("Find today's date and tell me the day of the week")
trace  = runner.get_trace()   # full step-by-step execution trace
```

### Agent Memory & Scratchpad

```python
from ai_cli.agents.agent import AgentMemory, Message

mem = AgentMemory(max_messages=20)

mem.add(Message(role="user", content="Remember: my name is Alice"))
mem.remember("username", "alice")          # persistent scratchpad
mem.remember("preferences", {"lang": "Python"})

name = mem.recall("username")              # → "alice"
summary = mem.summary()                    # last N messages + scratchpad
```

### Interactive REPL Agent Commands

Inside `ai-cli -i`:

```
/agent Find the current Bitcoin price and calculate 500 USD worth
/tool  calculator {"expression": "500 / 45000"}
/tools
/switch groq
/whoami
```

---

## Configuration Profiles

Place `ai-cli.yaml` in your project root (or set `AI_CLI_CONFIG`):

```yaml
profiles:

  default:
    provider: auto
    timeout: 60
    max_steps: 10
    strict_sanitize: false
    cost_budget_usd: 10.0
    tools:
      enabled: true
      allowed: []           # empty = all tools

  production:
    provider: openai
    model: gpt-4o
    timeout: 30
    strict_sanitize: true
    cost_budget_usd: 50.0
    tools:
      enabled: true
      allowed: [web_search, calculator]

  developer:
    provider: groq
    model: llama3-8b-8192
    timeout: 30
    cost_budget_usd: 5.0
    tools:
      enabled: true
      allowed: [web_search, calculator, file_read, datetime_now, http_get]

  agent:
    provider: openai
    model: gpt-4o
    timeout: 120
    max_steps: 15
    verbose: true
    tools:
      enabled: true
      allowed: []
```

```python
from ai_cli.config.profiles import load_profile

cfg = load_profile("production")
print(cfg.provider, cfg.cost_budget_usd)
```
---

# Advanced RAG Support
Features:
- Semantic chunking
- Embedding generation
- FAISS vector database
- Similarity search
- Persistent vector indexes
- Local document retrieval
- PDF/TXT/Markdown ingestion


# Installation

pip install -r requirements.txt

# Example Usage
ai-chat --rag --rag-docs docs/*.pdf "Explain the architecture"

# Index documents
ai-chat index docs/

# Architecture
Documents
   ↓
Chunking
   ↓
Embeddings
   ↓
FAISS Index
   ↓
Similarity Retrieval
   ↓
Prompt Augmentation
   ↓
LLM
---

# Advanced RAG Support

Features:

- Semantic chunking
- Embedding generation
- FAISS vector database
- Persistent vector indexes
- PDF/TXT/Markdown ingestion
- Semantic retrieval
- Retrieval-augmented prompting

## Installation

```bash
pip install -r requirements.txt

Example Usage

ai-chat --rag --rag-docs docs/architecture.pdf "Explain the architecture"

Supported File Types

PDF

TXT

Markdown

RAG Pipeline

Documents
↓
Chunking
↓
Embeddings
↓
FAISS Index
↓
Similarity Search
↓
Prompt Augmentation
↓
LLM Response

---

## Resilience & Reliability

### What's been enhanced in v0.3

- `CircuitBreaker` — now has three states: CLOSED → OPEN → HALF_OPEN recovery probe
- `BulkheadLimiter` — new: caps concurrent requests per provider to prevent cascade failures
- `HealthTracker` — new: tracks success rate and average latency per provider
- `RetryEngine` — configurable `max_delay` cap and custom exception filtering
- `RateLimiter` — new `wait_and_acquire()` method for blocking callers

```python
from ai_cli.core.resilience import (
    CircuitBreaker, BulkheadLimiter, HealthTracker, RetryEngine
)

# Circuit breaker wrapping any callable
cb = CircuitBreaker(threshold=5, timeout=30, name="openai")
protected_fn = cb.wrap(my_provider_call)

# Bulkhead: max 5 concurrent requests to this provider
bh = BulkheadLimiter(max_concurrent=5, name="openai")
isolated_fn = bh.wrap(my_provider_call)

# Health tracking
from ai_cli.core.resilience import HEALTH
HEALTH.record("openai", success=True, latency_ms=320.0)
health = HEALTH.get("openai")
print(health.success_rate, health.avg_latency_ms)
healthy = HEALTH.healthy_providers()
```

---

## Exception Hierarchy

```
AIProviderError
├── PromptValidationError
├── ProviderConfigurationError
├── ProviderRequestError
├── ResponseValidationError
│
├── AuthenticationError          # identity / API key invalid
├── AuthorizationError           # RBAC denied
├── SecretResolutionError        # secret not found in any backend
├── RateLimitExceededError       # per-identity rate limit hit
├── BudgetExceededError          # cost cap reached
├── AuditError                   # audit log write failed (fail-secure)
│
├── ToolNotFoundError            # tool name not in registry
├── ToolExecutionError           # tool raised an exception
├── ToolInputValidationError     # missing/invalid tool arguments
├── ToolOutputValidationError    # tool result failed schema check
│
├── AgentMaxStepsError           # agent exceeded max_steps
├── AgentPlanningError           # planner produced no steps
└── AgentMemoryError             # memory operation failed
```

---

## CLI Reference

```
usage: ai-cli [-h] [-p PROVIDER] [-q PROMPT] [-m MODEL] [-i]
              [--timeout TIMEOUT] [--debug] [--version]
              [-a] [--plan] [--max-steps N] [--trace] [--verbose]
              [--with-tools] [--tools T1,T2] [--tool NAME [JSON]]
              [--list-tools] [--list-providers]
              [--login API_KEY] [--whoami] [--budget] [--audit N]
              [--strict]

Flags:
  -p, --provider       AI provider (default: auto)
  -q, --prompt         Prompt / goal text
  -m, --model          Model override
  -i, --interactive    Start interactive REPL
  --timeout            Request timeout in seconds (default: 60)
  --debug              Enable debug logging
  --version            Show version

Agent:
  -a, --agent          Run ReAct agent loop
  --plan               Generate a plan before running agent
  --max-steps N        Maximum agent steps (default: 10)
  --trace              Print tool execution trace after run
  --verbose            Show step-by-step agent output

Tool Calling:
  --with-tools         Enable automatic tool dispatch in ask mode
  --tools T1,T2        Comma-separated tools to expose
  --tool NAME [JSON]   Call a single tool directly
  --list-tools         List all registered tools
  --list-providers     List all registered providers

Security:
  --login API_KEY      Authenticate with a gateway API key
  --whoami             Show identity, roles, and budget
  --budget             Show detailed budget usage
  --audit N            Tail last N audit log entries
  --strict             Strict prompt injection detection (raises on hit)
```

### Usage Examples

```bash
# Plain ask
ai-cli -q "Explain Kubernetes operators"
ai-cli -p groq -m llama3-8b-8192 -q "Write a haiku about Python"
echo "Summarize this text" | ai-cli -p openai

# Tool calling
ai-cli --tool calculator '{"expression": "sqrt(144) * 3.14"}'
ai-cli --tool datetime_now
ai-cli --with-tools -q "What is 17! and what day is today?"
ai-cli --list-tools

# Agent mode
ai-cli --agent -q "Find today's date, then calculate days left in 2025"
ai-cli --agent --plan --verbose -q "Research the top Python web frameworks"
ai-cli --agent --max-steps 5 --trace -q "What is the cube root of 1000000?"

# Security
ai-cli --login aicli-developer-alice-abcdef1234567890abcdef1234567890
ai-cli --whoami
ai-cli --budget
ai-cli --audit 20
ai-cli --strict -q "Your prompt here"

# Interactive REPL
ai-cli -i
# then inside REPL:
# /agent Find the current time and calculate 86400 - (seconds since midnight)
# /tool calculator {"expression": "2**32"}
# /switch anthropic
# /whoami
```

---

## Developer Experience

### Interactive REPL Commands

| Command | Description |
|---|---|
| `/switch <provider>` | Hot-swap the AI provider mid-session |
| `/agent <goal>` | Run a ReAct agent inline |
| `/tool <name> [json]` | Call a tool and see the result |
| `/tools` | List available tools |
| `/whoami` | Show identity and roles |
| `/clear` | Clear the terminal |
| `/exit` | Quit |

### Rich Terminal Output

If `rich` is installed (`pip install rich`), the CLI automatically renders:
- Markdown-formatted AI responses
- Colour-coded errors and success messages
- Formatted tables for `--list-tools` and `--list-providers`
- Panels for tool results, agent answers, and identity info

Without `rich`, plain-text output is used — no hard dependency.

### Auto-formatting & Linting

```bash
black src/
ruff check src/ --fix
mypy src/
```

---

## Testing

```bash
# Run the full enhanced test suite
pytest tests/test_enhanced.py -v

# Run with coverage
pytest tests/test_enhanced.py --cov=src/ai_cli --cov-report=term-missing

# Run a specific group
pytest tests/test_enhanced.py -v -k "Security"
pytest tests/test_enhanced.py -v -k "Tool"
pytest tests/test_enhanced.py -v -k "Agent"
pytest tests/test_enhanced.py -v -k "Resilience"
```

The test suite covers:

- Identity context, set/get, RBAC authorization
- Secrets backend (env resolution, cache invalidation, missing key error)
- Audit logger (write, verify, tamper detection)
- Prompt sanitisation, PII redaction, injection detection
- Per-identity rate limiting and budget enforcement
- Tool registry (register, get, list, schema export, RBAC gating)
- All 7 built-in tools including edge cases and security checks
- Tool execution tracing and ToolResult properties
- AgentMemory (add, sliding window, scratchpad, serialisation)
- AgentPlanner step parsing
- ReactAgent (immediate answer, tool-call-then-answer, prose fallback, max-steps)
- CircuitBreaker state transitions (CLOSED → OPEN → HALF_OPEN → CLOSED)
- RetryEngine, BulkheadLimiter, HealthTracker
- ProfileConfig defaults and YAML loading
- CLI argument parser flags

---

## Observability & Monitoring

- OpenTelemetry native spans and traces (existing)
- Prometheus metrics (existing):
  - `ai_provider_requests_total`
  - `ai_provider_errors_total`
  - `ai_provider_tokens_total`
  - `ai_provider_request_latency_seconds`
  - `ai_provider_cost_estimated_total`
- `HealthTracker` now provides programmatic success-rate and latency data per provider
- Audit log (`audit.jsonl`) is HMAC-signed for tamper detection and SIEM export

```yaml
# Prometheus scrape config
scrape_configs:
  - job_name: ai_cli
    static_configs:
      - targets: ["ai-cli:9100"]
```

Grafana dashboard templates: `docs/dashboards/`

---

## Environment Variables Reference

```bash
# Provider API keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AIza...
GROQ_API_KEY=gsk_...
PERPLEXITY_API_KEY=pplx-...
DEEPSEEK_API_KEY=sk-...
TOGETHER_API_KEY=...
COHERE_API_KEY=...
OPENROUTER_API_KEY=...
FIREWORKS_API_KEY=...
XAI_API_KEY=...
MISTRAL_API_KEY=...

# Gateway security
AI_CLI_GATEWAY_SECRET=change-me-in-production
AI_CLI_IDENTITY=your_username
AI_CLI_ROLES=developer
AI_CLI_AUDIT_LOG=audit.jsonl
AI_CLI_AUDIT_SECRET=audit-secret-change-me
AI_CLI_POLICY=ai-cli-policy.json

# Secrets backends (all optional)
VAULT_ADDR=http://vault:8200
VAULT_TOKEN=hvs.xxx
VAULT_SECRET_PATH=secret/data/ai_cli
AWS_REGION=us-east-1
AWS_SECRET_NAME=ai_cli_secrets
AZURE_VAULT_URL=https://my-vault.vault.azure.net

# Infrastructure (optional)
REDIS_URL=redis://localhost:6379/0
AI_CLI_CONFIG=ai-cli.yaml
```

---

## Deployment

### Local / Docker

```bash
# Local
poetry install
ai-cli -q "Hello"

# Docker
docker build -t ai-cli:latest .
docker run --env-file .env ai-cli:latest -q "Hello"

# With agent mode
docker run --env-file .env ai-cli:latest --agent -q "Find today's date"
```

### Kubernetes

```bash
helm install ai-cli charts/ai-cli \
  --set secrets.openaiApiKey=$OPENAI_API_KEY \
  --set gateway.secret=$AI_CLI_GATEWAY_SECRET
```

---

## Roadmap

Items moved from "planned" to "done" in v0.3 are marked ✅.

- ✅ Layered secrets backend (Vault, AWS SM, Azure KV)
- ✅ RBAC with per-role provider and tool access control
- ✅ Append-only audit log with HMAC integrity
- ✅ Prompt injection detection and PII redaction
- ✅ Per-identity rate limiting and budget enforcement
- ✅ Tool calling registry with 7 built-in tools
- ✅ ReAct agent loop with memory, scratchpad, and planner
- ✅ YAML configuration profiles
- ✅ CircuitBreaker half-open recovery, BulkheadLimiter, HealthTracker
- ✅ Comprehensive test suite (50+ tests across all new modules)
- ⬜ AWS Bedrock / GCP Vertex AI / Azure OpenAI native IAM auth
- ⬜ Multi-region failover with Cloudflare Workers / Lambda@Edge
- ⬜ Advanced RAG: chunking, embedding, vector-DB querying
- ⬜ Streaming responses (SSE / gRPC)
- ⬜ Shell autocompletion (bash / zsh / fish)
- ⬜ REST + gRPC gateway server mode

---

## Contributing

Follow `docs/DEVELOPMENT.md`. Use GitHub flow, sign commits, include tests and a changelog entry. Run `black`, `ruff`, and `mypy` before opening a PR.

---
Recommended future upgrades

- Hybrid BM25 + vector search
- Cross-encoder reranking
- Async indexing
- GPU embeddings
- Parent-child chunking
- Metadata filtering
- Qdrant integration
- ChromaDB integration
- pgvector support
- Streaming retrieval

---

## License

MIT License — see [LICENSE](LICENSE) for details.