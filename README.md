# AI CLI Gateway — Enhanced README

Enterprise-grade multi-provider AI CLI gateway for intelligent AI operations, observability, routing, reliability engineering, governance, and infrastructure automation.

---

## Quickstart

1. Clone:

```bash
git clone https://github.com/yourusername/ai-cli.git
cd ai-cli
```

2. Install (Poetry recommended; editable install and pipx also supported):

```bash
# Poetry
poetry install

# Editable pip install (local development)
pip install -e .

# Optional: install into isolated runtime with pipx
pipx install .
```

3. Create `.env` or use your secret backend (see Security & Secrets). Required keys are referenced by env names in profiles (e.g., OPENAI_API_KEY).

4. Run a single prompt (Auto-Fallback enabled by default; use --no-fallback to disable):

```bash
PYTHONPATH=src python3 -m ai_cli.cli --prompt "Explain Kubernetes operators"
# with explicit profile and streaming output
PYTHONPATH=src python3 -m ai_cli.cli --profile default --stream --prompt "Explain Kubernetes operators"
```

5. Start Interactive Chat Mode (streaming, history and hot-switch supported):

```bash
PYTHONPATH=src python3 -m ai_cli.cli -i
# In REPL: /switch <provider>  /history  /clear
```

---

## What’s New / Enhancements Overview

Recent updates include:

- Async-first provider adapters and SDK improvements
- Streaming-first CLI and SDK support (SSE/gRPC)
- Provider plugin system with hot-reload and async entrypoints
- Declarative YAML configuration with per-profile secret references
- Secret-backend integrations (AWS Secrets Manager, HashiCorp Vault)
- Opt-in OpenTelemetry auto-instrumentation and improved Prometheus metrics
- Circuit breakers, bulkheads, adaptive retry policies, and request-based cost estimation
- Helm chart, operator improvements, and GitOps-friendly manifests
- Local web UI for quick debugging (http://localhost:8080) and Prometheus exporters
- Expanded test matrix and contract tests for provider adapters

---

## Configuration

Supports environment variables, CLI flags, and declarative YAML profiles.

Example profile (ai-config.yaml):

```yaml
profiles:
  default:
    providers:
      - name: openai
        type: openai
        api_key_env: OPENAI_API_KEY
        region: us-east-1
        timeout: 30s
        max_tokens: 2048
    routing:
      strategy: cost_latency_balance
      cost_weight: 0.6
      latency_weight: 0.4
    reliability:
      retries: 3
      backoff: exponential
      circuit_breaker:
        failure_threshold: 0.05
        window_seconds: 60
```

Profile usage:

```bash
PYTHONPATH=src python3 -m ai_cli.cli --profile default --prompt "Summarize incident report"
```

Notes:
- Secrets can be referenced via env names or a secret backend path (see Security & Secrets).
- Use --format json for structured output convenient for automation.

---

## Provider Plugin API

- Discoverable plugins (pip-installable) with standardized async adapter interface
- Hot-reloadable provider registry and feature flags per-provider
- Example adapter entrypoint (async):

```python
class ProviderAdapter(BaseProvider):
    async def send(self, request): ...
    async def health(self): ...
```

Backward-compatible sync wrappers are provided for simple adapters.

---

## Security & Secrets

- Secret backends supported: AWS Secrets Manager, GCP Secret Manager, HashiCorp Vault, Azure Key Vault
- Configure backend via env: SECRET_BACKEND (e.g., aws_secrets, vault) and backend-specific config in ai-config.yaml
- API key isolation per profile and per-team; secrets injected at runtime — no raw keys in logs
- Prompt sanitization, input validation, response redaction policies, and configurable PII scrubbing
- Audit logs (append-only) exportable to SIEM; SSO/OIDC + SCIM provisioning + RBAC supported

Example: use secret reference instead of env var

```yaml
providers:
  - name: openai
    type: openai
    api_key_secret: "secrets/ai/openai/api_key"
```

---

## Observability & Monitoring

- OpenTelemetry native spans and traces; opt-in via AI_CLI_OTEL_ENABLED=true and OTel config
- Prometheus metrics with example scrape config:

```yaml
scrape_configs:
  - job_name: ai_cli
    static_configs:
      - targets: ["ai-cli:9100"]
```

- Key metrics:
  - ai_provider_requests_total
  - ai_provider_errors_total
  - ai_provider_tokens_total
  - ai_provider_request_latency_seconds
  - ai_provider_cost_estimated_total
- Local web UI provides live traces and metrics; Grafana dashboard JSON templates available in /docs/dashboards

---

## Reliability & Resilience

- Circuit breaker + bulkhead isolation per provider with configurable thresholds
- Adaptive retry policies (jitter, exponential backoff), request-level deadlines, and cancellation
- Health-aware provider selection, automated failover, and canary rollout primitives
- Request validation pipelines, hallucination scoring hooks, and optional content moderation adapters

---

## Cost & Governance

- Token and cost tracking by request, model, and team (billing meters exportable)
- Budget alerts, quotas, and auto-throttling when budgets are exceeded
- Policy engine (policy-as-code) to enforce model usage, content rules, and data residency
- Configurable data retention and request audit policies per profile

---

## Deployment Options

- Single-binary CLI (local)
- Docker image:

```bash
docker build -t ai-cli:latest .
docker run --env-file .env ai-cli:latest --profile default --prompt "Hello"
```

- Kubernetes:
  - Helm chart (charts/ai-cli) with values for secret backends and observability
  - Optional operator for autoscaling and multi-region failover
- Cloud-native: OCI images + GitOps (ArgoCD) examples in /deploy

---

## Developer Experience

- Interactive Chat REPL: streaming, history, provider hot-swap (`/switch <provider>`)
- Shell completion (bash/zsh/fish) and CLI man page
- Local devcontainer, mock provider, and local web UI for inspection
- Auto-formatting and lint hooks: black, ruff, mypy; pre-commit included

---

## SDKs & API

- CLI, REST and gRPC endpoints (configurable)
- Python SDK: importable package (ai_cli.gateway) for embedding the gateway into applications
- Streaming responses supported via gRPC streams and SSE; clients can opt into structured JSON streaming

Example Python usage:

```python
from ai_cli.gateway import Client
client = Client.from_profile("default")
resp = await client.complete("Explain durable queues")
```

---

## Testing & CI

- Unit, integration, and contract tests for provider adapters
- Example GitHub Actions workflows with matrix builds and cache layers in .github/workflows
- Local test harness spins up mock providers, Prometheus, and runs contract checks

---

## Troubleshooting

- Debug logging and diagnostic run:

```bash
PYTHONPATH=src python3 -m ai_cli.cli --debug --provider echo --prompt "Test"
```

- Diagnostic bundle collection:

```bash
ai-cli collect-diagnostics --output diagnostics.tgz
```

- Use the local web UI to inspect recent traces, logs, and metrics when available

---

## Contributing & Roadmap

- Follow docs/DEVELOPMENT.md for contributor guidelines, tests, and changelog requirements
- Use GitHub flow, sign commits where required, include tests for new behavior

Planned near-term items:
- Expanded cloud-native auth integrations (Workload Identity, IAM Roles Anywhere)
- Enhanced RAG orchestration and end-to-end vector DB pipelines
- Improved multi-region edge routing and autoscaling primitives

---

## Example Files & Locations

- configs/: example YAML profiles
- plugins/: sample provider adapters
- docs/dashboards/: Grafana templates
- deploy/: Dockerfile, Helm chart, k8s manifests, operator samples
- tests/: integration and contract tests

---

## License

MIT License — see LICENSE for details.

---

For full reference, examples, and templates, see the docs/ directory and examples/. Contributions are welcome.
