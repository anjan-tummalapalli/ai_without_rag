# AI CLI Gateway — Enhanced README

Enterprise-grade multi-provider AI CLI gateway for intelligent AI operations, observability, routing, reliability engineering, governance, and infrastructure automation.

---

## Quickstart

1. Clone:
```bash
git clone https://github.com/yourusername/ai-cli.git
cd ai-cli
```
2. Install (Poetry recommended):
```bash
poetry install
```
3. Create `.env` with required keys (see Environment Variables).
4. Run single prompt:
```bash
ai-cli --provider openai --prompt "Explain Kubernetes operators"
```

---

## What’s New / Enhancements Overview

This release expands the original feature set with focused improvements:

- Provider plugin system for add-ons and third-party adapters
- Declarative YAML configuration and profile support
- Secure secrets integration (KMS, HashiCorp Vault, AWS Secrets Manager)
- RBAC, audit logging, and SSO/OIDC support for enterprise access control
- Circuit breakers, bulkheads, and advanced retry policies
- Cost budgets, quotas, and per-team billing meters
- Prometheus + OpenTelemetry full-stack observability + Grafana dashboards
- Helm chart and Kubernetes operator for production deployment
- Async multi-model orchestration and streaming responses
- Local dev container and shell autocompletion
- End-to-end CI workflows and contract tests

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
    routing:
      strategy: cost_latency_balance
      cost_weight: 0.6
      latency_weight: 0.4
    reliability:
      retries: 3
      backoff: exponential
```

Profile usage:
```bash
ai-cli --profile default --prompt "Summarize incident report"
```

---

## Provider Plugin API

- Discoverable plugins (pip-installable) with standardized adapter interface
- Hot-reloadable provider registry
- Example adapter entrypoint:
```python
class ProviderAdapter(BaseProvider):
    def send(self, request): ...
    def health(self): ...
```

---

## Security & Secrets

- Integrations: AWS KMS, GCP KMS, HashiCorp Vault, Azure Key Vault
- API key isolation per profile and per-team
- Prompt sanitization, input validation, response redaction policies
- Audit logs (append-only) and export to SIEM
- SSO/OIDC + SCIM provisioning + role-based access control

---

## Observability & Monitoring

- OpenTelemetry native spans and traces
- Prometheus metrics with example scrape config:
```yaml
scrape_configs:
  - job_name: ai_cli
    static_configs:
      - targets: ['ai-cli:9100']
```
- Key metrics:
  - ai_provider_requests_total
  - ai_provider_errors_total
  - ai_provider_tokens_total
  - ai_provider_request_latency_seconds
  - ai_provider_cost_estimated_total
- Grafana dashboard JSON templates provided in /docs/dashboards

---

## Reliability & Resilience

- Circuit breaker + bulkhead isolation per provider
- Adaptive retry policies (Jitter, exponential backoff)
- Health-aware provider selection and automated failover
- Request validation pipelines and hallucination scoring hooks
- Canary and staged rollout primitives for model changes

---

## Cost & Governance

- Token and cost tracking by request, model, and team
- Budget alerts and auto-throttling when budgets exceeded
- Policy engine (policy-as-code) to enforce model usage, content rules, and data residency
- Data retention and request audit policies configurable per profile

---

## Deployment Options

- Single-binary CLI (local)
- Docker image:
```bash
docker build -t ai-cli:latest .
docker run --env-file .env ai-cli:latest --prompt "Hello"
```
- Kubernetes:
  - Helm chart (charts/ai-cli)
  - Optional operator for autoscaling and multi-region failover
- Cloud-native: OCI images + GitOps (ArgoCD) examples in /deploy

---

## Developer Experience

- Shell completion (bash/zsh/fish)
- Interactive REPL mode and Jupyter-friendly SDK
- Devcontainer and local mock provider for tests
- Auto-formatting and lint hooks: black, ruff, mypy

---

## Testing & CI

- Unit, integration, and contract tests for provider adapters
- Example GitHub Actions workflow (ci.yml) with matrix builds and test caching
- Local test harness that can spin up mock providers and Prometheus

---

## API & SDKs

- Exposes CLI, REST and gRPC endpoints (configurable)
- Minimal SDKs: Python client for embedding the gateway
- Streaming responses supported via gRPC streams and SSE

---

## Troubleshooting

- Common logs and diagnostic commands:
```bash
ai-cli --debug --profile debug
ai-cli health-check --provider openai
```
- Diagnostic bundle collection: `ai-cli collect-diagnostics --output diagnostics.tgz`

---

## Contributing & Roadmap

- Follow docs/DEVELOPMENT.md for contributor guidelines
- Use GitHub flow, sign commits, include tests and changelog entries
- Roadmap highlights: multi-region replication, RAG orchestration, model explainability, workload autoscaling

---

## Example Files & Locations

- configs/: example YAML profiles
- plugins/: sample provider adapters
- docs/dashboards/: Grafana templates
- deploy/: Dockerfile, Helm chart, k8s manifests
- tests/: integration and contract tests

---

## License

MIT License — see LICENSE for details.

---

For full reference, examples, and templates, see the docs/ directory and examples/. Contributions are welcome.
